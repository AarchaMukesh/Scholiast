"""The two ingestion use cases, against fakes (§5.4.2, FR-I5, FR-I7, FR-I9).

Fakes rather than mocks, and every one of them keeps a journal. That is what
lets these tests assert *ordering* — which is where the interesting correctness
lives here: D53 says Postgres is written before vectors and chunks are stamped
after the upsert, and a test that only checks "everything happened" passes for
either ordering while one of them silently loses chunks on a crash.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import NamedTuple

import pytest

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.parsing import DocumentUnparseable, ParsedDocument
from scholiast.application.ports.storage import ObjectNotFound, source_key
from scholiast.application.ports.vector import VectorRecord
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.application.use_cases.ingest_paper import IngestionOutcome, IngestPaperUseCase
from scholiast.domain.chunking import BlockKind, ParsedBlock, ParsedSection
from scholiast.domain.errors import FailureClass, InvariantViolation, WorkspaceNotAvailable
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperVisibility,
    ParseStatus,
    Section,
    SectionKind,
)
from scholiast.domain.upload import UploadRejected
from scholiast.domain.workspace import PaperMembership, PaperState, Workspace

OWNER = "11111111-1111-4111-8111-111111111111"
WORKSPACE = "22222222-2222-4222-8222-222222222222"
PDF = b"%PDF-1.7\n" + b"body " * 100
#: The real digest, so the dedup fake answers the question the use case actually
#: asks rather than a convenient one.
PDF_HASH = hashlib.sha256(PDF).hexdigest()


# ---------------------------------------------------------------------------
# Fakes. Each one records what it was asked to do, in order, on a shared journal.
# ---------------------------------------------------------------------------


@dataclass
class Journal:
    events: list[str] = field(default_factory=list)

    def record(self, event: str) -> None:
        self.events.append(event)

    def index_of(self, event: str) -> int:
        return self.events.index(event)


class FakeWorkspaces:
    def __init__(self, workspace: Workspace | None) -> None:
        self._workspace = workspace

    async def get(self, workspace_id: str) -> Workspace | None:
        if self._workspace is None or self._workspace.id != workspace_id:
            return None
        return self._workspace


class FakePapers:
    def __init__(self, journal: Journal, *, existing: Paper | None = None) -> None:
        self._journal = journal
        self.by_id: dict[str, Paper] = {}
        self.memberships: list[tuple[str, str, PaperState]] = []
        self.states: list[PaperState] = []
        if existing is not None:
            self.by_id[existing.id] = existing
        self._existing = existing

    async def get(self, paper_id: str) -> Paper | None:
        return self.by_id.get(paper_id)

    async def find_by_content_hash(self, content_hash: str) -> Paper | None:
        if self._existing is not None and self._existing.content_hash == content_hash:
            return self._existing
        return None

    async def add(self, paper: Paper) -> Paper:
        self._journal.record("papers.add")
        self.by_id[paper.id] = paper
        return paper

    async def update(self, paper: Paper) -> bool:
        self._journal.record("papers.update")
        self.by_id[paper.id] = paper
        return True

    async def add_to_workspace(
        self, workspace_id: str, paper_id: str, state: PaperState
    ) -> PaperMembership:
        self._journal.record(f"papers.add_to_workspace:{state.value}")
        self.memberships.append((workspace_id, paper_id, state))
        return PaperMembership(
            workspace_id=workspace_id,
            paper_id=paper_id,
            state=state,
            added_at=datetime.now(UTC),
        )

    async def set_membership_state(
        self,
        workspace_id: str,
        paper_id: str,
        state: PaperState,
        *,
        failure_class: str | None = None,
        failure_detail: str | None = None,
    ) -> bool:
        self._journal.record(f"membership:{state.value}")
        self.states.append(state)
        return True


class FakeSections:
    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self.written: list[Section] = []

    async def replace_for_paper(self, paper_id: str, sections: Sequence[Section]) -> int:
        self._journal.record("sections.replace")
        self.written = list(sections)
        return len(sections)

    async def list_for_paper(self, paper_id: str) -> list[Section]:
        return list(self.written)


class FakeChunks:
    def __init__(self, journal: Journal, *, existing: list[Chunk] | None = None) -> None:
        self._journal = journal
        self.written: list[Chunk] = []
        self.stamped: list[str] = []
        self._existing = existing or []

    async def add_many(self, chunks: Sequence[Chunk], *, section_ids: dict[str, str]) -> int:
        self._journal.record("chunks.add_many")
        for chunk in chunks:
            if chunk.provenance.section_path not in section_ids:
                raise KeyError(chunk.provenance.section_path)
        self.written = list(chunks)
        return len(chunks)

    async def stamp_embedded(self, chunk_ids: Sequence[str], version: str) -> int:
        self._journal.record("chunks.stamp")
        self.stamped = list(chunk_ids)
        return len(chunk_ids)

    async def list_for_paper(self, paper_id: str) -> list[Chunk]:
        return list(self._existing)


class FakeStore:
    def __init__(self, journal: Journal, *, missing: bool = False) -> None:
        self._journal = journal
        self.objects: dict[str, bytes] = {}
        self._missing = missing

    async def put(self, key: str, data: bytes, *, content_type: str = "") -> None:
        self._journal.record(f"store.put:{key}")
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        if self._missing or key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    async def exists(self, key: str) -> bool:
        return key in self.objects

    async def delete(self, key: str) -> bool:
        return self.objects.pop(key, None) is not None


class FakeQueue:
    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self.enqueued: list[tuple[str, str]] = []
        self.stages: list[PaperState] = []
        self.completed: list[str] = []

    async def enqueue(self, *, workspace_id: str, paper_id: str) -> str:
        self._journal.record("queue.enqueue")
        self.enqueued.append((workspace_id, paper_id))
        return f"job-{len(self.enqueued)}"

    async def claim(self, *, limit: int = 1, lease_seconds: float = 300.0) -> list[object]:
        return []

    async def heartbeat(
        self, job_id: str, *, state: PaperState, lease_seconds: float = 300.0
    ) -> bool:
        self._journal.record(f"queue.heartbeat:{state.value}")
        self.stages.append(state)
        return True

    async def complete(self, job_id: str) -> bool:
        self.completed.append(job_id)
        return True

    async def fail(
        self,
        job_id: str,
        *,
        failure_class: FailureClass,
        detail: str,
        max_attempts: int = 3,
    ) -> PaperState:
        return PaperState.FAILED


class FakeUnitOfWork:
    def __init__(self, journal: Journal) -> None:
        self._journal = journal
        self.commits = 0

    async def commit(self) -> None:
        self._journal.record("commit")
        self.commits += 1


class FakeParser:
    version = "fake-parser/1.0.0"

    def __init__(self, document: ParsedDocument | Exception) -> None:
        self._document = document

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        if isinstance(self._document, Exception):
            raise self._document
        return self._document


class FakeEmbedder:
    version = "fake-embedder/1.0.0-fp16"
    dimensions = 8

    def __init__(self, journal: Journal, *, short_by: int = 0) -> None:
        self._journal = journal
        self.inputs: list[str] = []
        self._short_by = short_by

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self._journal.record("embedder.embed")
        self.inputs = list(texts)
        vectors = [[float(len(t) % 7) / 7.0] * self.dimensions for t in texts]
        return vectors[: len(vectors) - self._short_by]


class FakeIndex:
    def __init__(self, journal: Journal, *, upsert_raises: Exception | None = None) -> None:
        self._journal = journal
        self.records: list[VectorRecord] = []
        self._upsert_raises = upsert_raises

    async def ensure_ready(self) -> None: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self._journal.record("index.upsert")
        if self._upsert_raises is not None:
            # The vector store refusing the write is the failure D53's ordering
            # exists for. Raising *before* recording the records keeps the fake
            # honest: nothing was indexed.
            raise self._upsert_raises
        self.records = list(records)
        return len(records)

    async def search(self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int):  # type: ignore[no-untyped-def]
        return []

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        return 0

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        return len(self.records)


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def parsed_section(
    path: str, text: str, *, kind: SectionKind = SectionKind.METHOD
) -> ParsedSection:
    return ParsedSection(
        kind=kind,
        path=path,
        blocks=(
            ParsedBlock(
                kind=BlockKind.PARAGRAPH, text=text, page=3, char_start=0, char_end=len(text)
            ),
        ),
        text=text,
        raw_title=path,
    )


def document(*sections: ParsedSection, tei: str | None = "<TEI/>") -> ParsedDocument:
    body = sections or (
        parsed_section("Method", "We propose a retrieval method. " * 20),
        parsed_section(
            "Related Work", "Prior systems did other things. " * 20, kind=SectionKind.RELATED_WORK
        ),
    )
    return ParsedDocument(
        sections=body,
        title="A Real Title",
        authors=("Ada Lovelace",),
        abstract="An abstract.",
        page_count=9,
        parser_version=FakeParser.version,
        raw_tei=tei,
    )


def workspace() -> Workspace:
    return Workspace(id=WORKSPACE, owner_id=OWNER, name="Reading list")


def stored_paper(**overrides: object) -> Paper:
    defaults: dict[str, object] = {
        "id": "33333333-3333-4333-8333-333333333333",
        "content_hash": PDF_HASH,
        "visibility": PaperVisibility.PRIVATE,
        "owner_id": OWNER,
        "title": "paper",
    }
    defaults.update(overrides)
    return Paper(**defaults)  # type: ignore[arg-type]


def make_adder(
    journal: Journal,
    *,
    existing: Paper | None = None,
    existing_chunks: list[Chunk] | None = None,
    ws: Workspace | None = None,
) -> tuple[AddPaperUseCase, FakeStore, FakeQueue, FakePapers]:
    store = FakeStore(journal)
    queue = FakeQueue(journal)
    papers = FakePapers(journal, existing=existing)
    use_case = AddPaperUseCase(
        workspaces=FakeWorkspaces(workspace() if ws is None else ws),
        papers=papers,
        chunks=FakeChunks(journal, existing=existing_chunks),
        store=store,
        queue=queue,
        owner_id=OWNER,
    )
    return use_case, store, queue, papers


def make_ingester(
    journal: Journal,
    *,
    parser: FakeParser,
    paper: Paper | None = None,
    short_by: int = 0,
    upsert_raises: Exception | None = None,
) -> tuple[IngestPaperUseCase, FakeSections, FakeChunks, FakeIndex, FakeStore, FakePapers]:
    subject = paper or stored_paper()
    papers = FakePapers(journal)
    papers.by_id[subject.id] = subject
    sections = FakeSections(journal)
    chunks = FakeChunks(journal)
    index = FakeIndex(journal, upsert_raises=upsert_raises)
    store = FakeStore(journal)
    store.objects[source_key(subject.id)] = PDF
    use_case = IngestPaperUseCase(
        papers=papers,
        sections=sections,
        chunks=chunks,
        store=store,
        parser=parser,
        embedder=FakeEmbedder(journal, short_by=short_by),
        index=index,
        queue=FakeQueue(journal),
        uow=FakeUnitOfWork(journal),
    )
    return use_case, sections, chunks, index, store, papers


async def ingest(use_case: IngestPaperUseCase, paper_id: str):  # type: ignore[no-untyped-def]
    return await use_case.execute(
        job_id="job-1",
        workspace_id=WORKSPACE,
        paper_id=paper_id,
        membership=PaperMembership(
            workspace_id=WORKSPACE,
            paper_id=paper_id,
            state=PaperState.QUEUED,
            added_at=datetime.now(UTC),
        ),
    )


# ---------------------------------------------------------------------------
# AddPaperUseCase
# ---------------------------------------------------------------------------


class TestNothingIsWrittenForAnUploadThatCannotBeIngested:
    async def test_a_non_pdf_is_refused_before_the_blob_is_stored(self) -> None:
        journal = Journal()
        use_case, store, queue, _ = make_adder(journal)
        with pytest.raises(UploadRejected):
            await use_case.execute(workspace_id=WORKSPACE, data=b"PK\x03\x04", filename="x.pdf")
        assert store.objects == {}
        assert queue.enqueued == []

    async def test_an_unreachable_workspace_stores_nothing(self) -> None:
        # A workspace another tenant owns reads as absent (FR-A2). Storing the
        # blob first would let a non-owner fill the object store by guessing ids.
        journal = Journal()
        use_case, store, queue, _ = make_adder(journal, ws=workspace())
        # Its own error class, not a generic invariant violation: the two map to
        # different HTTP statuses, and a missing workspace answered with a 500
        # would be a bug report rather than a 404.
        with pytest.raises(WorkspaceNotAvailable):
            await use_case.execute(workspace_id="44444444-4444-4444-8444-444444444444", data=PDF)
        assert store.objects == {}
        assert queue.enqueued == []


class TestANewUploadBecomesOneJob:
    async def test_the_blob_is_stored_before_the_rows_and_the_job(self) -> None:
        # The worker reads the PDF by key. Rows committed before the blob exists
        # is a race the worker loses; a blob without rows is garbage the next
        # attempt overwrites at the same key.
        journal = Journal()
        use_case, store, queue, _ = make_adder(journal)
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF, filename="bert.pdf")

        assert journal.index_of(f"store.put:{source_key(result.paper_id)}") < journal.index_of(
            "papers.add"
        )
        assert journal.index_of("papers.add") < journal.index_of("queue.enqueue")
        assert store.objects[source_key(result.paper_id)] == PDF
        assert queue.enqueued == [(WORKSPACE, result.paper_id)]
        assert result.state is PaperState.QUEUED
        assert result.job_id is not None
        assert result.deduplicated is False

    async def test_an_upload_is_private_and_owned(self) -> None:
        # §4.4.2's carve-out: an upload is never globally shared, whatever its
        # hash collides with.
        journal = Journal()
        use_case, _, _, papers = make_adder(journal)
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF, filename="draft.pdf")
        paper = papers.by_id[result.paper_id]
        assert paper.visibility is PaperVisibility.PRIVATE
        assert paper.owner_id == OWNER

    async def test_the_title_is_visibly_the_filename_until_parsed(self) -> None:
        journal = Journal()
        use_case, _, _, papers = make_adder(journal)
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF, filename="big-bird.pdf")
        assert papers.by_id[result.paper_id].title == "big bird"


class TestReAddingDoesNoWork:
    async def test_an_already_embedded_paper_joins_indexed_with_no_job(self) -> None:
        # §5.4.2: "EXISTS? → create membership only → 200 (no re-work)". This is
        # FR-I7's whole point — re-adding must not re-parse or re-embed.
        journal = Journal()
        existing = stored_paper(parse_status=ParseStatus.PARSED)
        embedded = Chunk(
            id="c1",
            text="text",
            role=ChunkRole.TABLE,
            ordinal=0,
            provenance=_provenance(existing.id),
            chunker_version="v1",
            token_count=3,
            embedder_version="fake-embedder/1.0.0-fp16",
        )
        use_case, store, queue, _ = make_adder(
            journal, existing=existing, existing_chunks=[embedded]
        )
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF)

        assert result.deduplicated is True
        assert result.state is PaperState.INDEXED
        assert result.job_id is None
        assert queue.enqueued == []
        assert store.objects == {}

    async def test_a_parsed_but_unembedded_paper_is_queued(self) -> None:
        # `parse_status = parsed` says structure was extracted and says nothing
        # about vectors. Trusting it would leave a paper permanently unqueryable
        # while the workspace claimed it was indexed.
        journal = Journal()
        existing = stored_paper(parse_status=ParseStatus.PARSED)
        unembedded = Chunk(
            id="c1",
            text="text",
            role=ChunkRole.TABLE,
            ordinal=0,
            provenance=_provenance(existing.id),
            chunker_version="v1",
            token_count=3,
        )
        use_case, _, queue, _ = make_adder(journal, existing=existing, existing_chunks=[unembedded])
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF)

        assert result.state is PaperState.QUEUED
        assert result.job_id is not None
        assert queue.enqueued == [(WORKSPACE, existing.id)]

    async def test_a_paper_already_in_this_workspace_is_left_alone(self) -> None:
        journal = Journal()
        existing = stored_paper(parse_status=ParseStatus.PARSED)
        ws = workspace()
        ws.memberships.append(
            PaperMembership(
                workspace_id=WORKSPACE,
                paper_id=existing.id,
                state=PaperState.EMBEDDING,
                added_at=datetime.now(UTC),
            )
        )
        use_case, _, queue, papers = make_adder(journal, existing=existing, ws=ws)
        result = await use_case.execute(workspace_id=WORKSPACE, data=PDF)

        # Re-uploading mid-pipeline must not reset the paper or double-queue it.
        assert result.state is PaperState.EMBEDDING
        assert result.job_id is None
        assert queue.enqueued == []
        assert papers.memberships == []


def _provenance(paper_id: str):  # type: ignore[no-untyped-def]
    from scholiast.domain.paper import Provenance

    return Provenance(
        paper_id=paper_id,
        section_path="Method",
        section_kind=SectionKind.METHOD,
        page=1,
        char_start=0,
        char_end=4,
    )


# ---------------------------------------------------------------------------
# IngestPaperUseCase
# ---------------------------------------------------------------------------


class TestPostgresIsWrittenBeforeVectors:
    """D53 — the ordering that decides which inconsistency a crash leaves."""

    async def test_chunks_are_persisted_before_the_upsert(self) -> None:
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        outcome = await ingest(use_case, stored_paper().id)
        assert outcome.succeeded
        assert journal.index_of("chunks.add_many") < journal.index_of("index.upsert")

    async def test_chunks_are_stamped_only_after_the_upsert(self) -> None:
        # Stamping first would produce a chunk that claims to be searchable and
        # is not — a silent recall loss, versus an orphan vector the §5.6.4
        # sweep is built to find.
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)
        assert journal.index_of("index.upsert") < journal.index_of("chunks.stamp")

    async def test_sections_are_replaced_before_chunks_reference_them(self) -> None:
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)
        assert journal.index_of("sections.replace") < journal.index_of("chunks.add_many")


class _FailedIndexRun(NamedTuple):
    """What one ingestion attempt left behind when the vector store refused."""

    journal: Journal
    chunks: FakeChunks
    papers: FakePapers
    queue: FakeQueue
    outcome: IngestionOutcome


class TestIndexedIsNeverReachedWithoutVectors:
    """The consistency claim phase 5 rests on (D53, §5.4.2).

    Every other test here proves the *order* things happen in when they all
    succeed. This one proves what happens when the last one does not: if the
    vector store refuses the write, nothing downstream may claim it worked.
    Without it, `Chunk.is_indexed` and membership INDEXED are asserted by code
    reading alone — and they are what `AddPaperUseCase._attach` trusts when it
    decides a paper needs no re-ingestion at all.
    """

    @staticmethod
    async def _run_with_failing_index() -> _FailedIndexRun:
        journal = Journal()
        use_case, _, chunks, _, _, papers = make_ingester(
            journal,
            parser=FakeParser(document()),
            upsert_raises=RuntimeError("qdrant refused the upsert"),
        )
        outcome: IngestionOutcome = await ingest(use_case, stored_paper().id)
        return _FailedIndexRun(journal, chunks, papers, _queue_of(use_case), outcome)

    async def test_the_membership_never_reaches_indexed(self) -> None:
        _, _, papers, _, outcome = await self._run_with_failing_index()
        assert not outcome.succeeded
        assert PaperState.INDEXED not in papers.states
        # It got as far as EMBEDDING and stopped there, which is what makes the
        # membership honest about where the work actually died.
        assert papers.states == [
            PaperState.PARSING,
            PaperState.CHUNKING,
            PaperState.EMBEDDING,
        ]

    async def test_the_job_never_reaches_indexed(self) -> None:
        _, _, _, queue, _ = await self._run_with_failing_index()
        assert PaperState.INDEXED not in queue.stages
        assert queue.completed == []

    async def test_no_chunk_is_marked_embedded(self) -> None:
        # The invariant in one line: a stamped row with no vector is a chunk
        # that claims to be searchable and is not — and nothing would report it,
        # unlike the orphan vector §5.6.4's sweep is built to find.
        journal, chunks, _, _, _ = await self._run_with_failing_index()
        assert chunks.stamped == []
        assert "chunks.stamp" not in journal.events
        # The chunk rows themselves were written — that is D53's ordering, and
        # it is why a retry converges instead of losing them.
        assert chunks.written

    async def test_the_failure_follows_the_existing_retry_handling(self) -> None:
        # No new failure path: an unclassified store error is TRANSIENT, so the
        # queue's bounded retry and dead-letter apply unchanged (FR-I8, D48).
        _, _, _, _, outcome = await self._run_with_failing_index()
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.TRANSIENT
        assert outcome.failure.retryable is True


class TestWhatGetsEmbedded:
    async def test_parent_windows_are_not_embedded(self) -> None:
        # Embedding parents would double the index for no retrieval gain —
        # parents exist to be returned, not matched.
        journal = Journal()
        use_case, _, chunks, index, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)

        indexed_ids = {record.chunk_id for record in index.records}
        parents = {c.id for c in chunks.written if c.role is ChunkRole.PARENT}
        assert parents
        assert not (indexed_ids & parents)
        assert indexed_ids == {c.id for c in chunks.written if c.role is not ChunkRole.PARENT}

    async def test_the_contextual_header_is_applied_before_embedding(self) -> None:
        # ML-3, and the misattribution defence doc 2 §2.3 opens with: a Related
        # Work chunk must be lexically marked as describing other work.
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)

        embedder_inputs = _embedder_of(use_case).inputs
        assert all(text.startswith("[Paper: A Real Title") for text in embedder_inputs)
        assert any("describes other work" in text for text in embedder_inputs)

    async def test_every_vector_carries_both_version_stamps(self) -> None:
        # NFR-M5: "re-embed only what embedder v1 produced" has to be a query in
        # the store that holds the vectors, not only in Postgres.
        journal = Journal()
        use_case, _, _, index, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)
        assert all(r.embedder_version == FakeEmbedder.version for r in index.records)
        assert all(r.chunker_version for r in index.records)

    async def test_a_short_embedder_response_raises_rather_than_misaligning(self) -> None:
        # zip() would silently drop the tail, giving every later chunk the wrong
        # vector — wrong answers with no error anywhere.
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()), short_by=1)
        with pytest.raises(InvariantViolation):
            await ingest(use_case, stored_paper().id)


class TestTheStateMachineIsDrivenNotAssigned:
    async def test_each_stage_is_persisted_and_committed(self) -> None:
        # §5.4.2: "Each stage: persist state transition." Without the commit the
        # transitions all become visible at the same instant — the end.
        journal = Journal()
        use_case, _, _, _, _, papers = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)

        assert papers.states == [
            PaperState.PARSING,
            PaperState.CHUNKING,
            PaperState.EMBEDDING,
            PaperState.INDEXED,
        ]
        for state in papers.states:
            assert journal.index_of(f"membership:{state.value}") < len(journal.events)
        assert journal.events.count("commit") == 4

    async def test_the_job_stage_tracks_the_membership(self) -> None:
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)
        assert _queue_of(use_case).stages == [
            PaperState.PARSING,
            PaperState.CHUNKING,
            PaperState.EMBEDDING,
            PaperState.INDEXED,
        ]


class TestRetainedTei:
    async def test_the_tei_is_stored_under_a_version_stamped_key(self) -> None:
        # D43: re-chunk ablations must read the bytes that produced today's
        # chunks, not whatever a later GROBID would produce.
        journal = Journal()
        use_case, _, _, _, store, papers = make_ingester(
            journal, parser=FakeParser(document(tei="<TEI>real</TEI>"))
        )
        await ingest(use_case, stored_paper().id)

        paper = papers.by_id[stored_paper().id]
        assert paper.tei_key is not None
        assert FakeParser.version.replace("/", "_") in paper.tei_key
        assert store.objects[paper.tei_key] == b"<TEI>real</TEI>"

    async def test_a_parser_returning_no_tei_leaves_the_key_null(self) -> None:
        journal = Journal()
        use_case, _, _, _, _, papers = make_ingester(journal, parser=FakeParser(document(tei=None)))
        await ingest(use_case, stored_paper().id)
        assert papers.by_id[stored_paper().id].tei_key is None


class TestFailureIsClassifiedAndRecordedOnThePaper:
    async def test_an_unparseable_pdf_is_recorded_on_the_paper_row(self) -> None:
        # D44: parsing happens once per paper, so a second workspace adding this
        # file must see both that it failed and why — without re-parsing, and
        # without reading another tenant's membership row.
        journal = Journal()
        use_case, _, _, _, _, papers = make_ingester(
            journal,
            parser=FakeParser(DocumentUnparseable(FailureClass.SCANNED_IMAGE, "no text layer")),
        )
        outcome = await ingest(use_case, stored_paper().id)

        assert not outcome.succeeded
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.SCANNED_IMAGE
        paper = papers.by_id[stored_paper().id]
        assert paper.parse_status is ParseStatus.FAILED
        assert paper.parse_failure_class is FailureClass.SCANNED_IMAGE

    async def test_a_transient_parse_failure_does_not_mark_the_paper_terminally_failed(
        self,
    ) -> None:
        # `ParseStatus.FAILED` reports itself terminal, and D44's purpose is that
        # a second workspace can trust `papers.parse_status` without re-parsing.
        # Writing it for a GROBID outage would have the paper row claim a final
        # outcome for a job that is only in backoff — the membership's QUEUED
        # already says "retrying" (§4.4.1). A GROBID outage is the absence of an
        # outcome, not one.
        journal = Journal()
        use_case, _, _, _, _, papers = make_ingester(
            journal,
            parser=FakeParser(DocumentUnparseable(FailureClass.TRANSIENT, "GROBID unreachable")),
        )
        outcome = await ingest(use_case, stored_paper().id)

        # The failure is still classified and still retryable — the retry
        # machinery is untouched, only what the paper row claims changed.
        assert not outcome.succeeded
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.TRANSIENT
        assert outcome.failure.retryable is True

        paper = papers.by_id[stored_paper().id]
        assert paper.parse_status is not ParseStatus.FAILED
        assert paper.parse_failure_class is None

    async def test_a_terminal_parse_failure_is_still_recorded(self) -> None:
        # Positive control for the guard above: skipping the record for *every*
        # class would pass that test while deleting D44's reason for existing.
        # Walks the terminal classes rather than sampling one, so a class that
        # changes retryability is caught here rather than in production.
        for failure_class in (FailureClass.UNPARSEABLE, FailureClass.SCANNED_IMAGE):
            journal = Journal()
            use_case, _, _, _, _, papers = make_ingester(
                journal, parser=FakeParser(DocumentUnparseable(failure_class, "hopeless"))
            )
            await ingest(use_case, stored_paper().id)
            paper = papers.by_id[stored_paper().id]
            assert paper.parse_status is ParseStatus.FAILED
            assert paper.parse_failure_class is failure_class

    async def test_a_missing_blob_is_source_unavailable_not_unparseable(self) -> None:
        journal = Journal()
        use_case, _, _, _, store, _ = make_ingester(journal, parser=FakeParser(document()))
        store.objects.clear()
        outcome = await ingest(use_case, stored_paper().id)
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.SOURCE_UNAVAILABLE

    async def test_an_unclassified_error_defaults_to_transient(self) -> None:
        # The safe default: bounded retries then a dead-letter. Declaring the
        # paper permanently unusable would let one bug mark a corpus
        # unsupported with no way back (FR-I8).
        journal = Journal()
        use_case, _, _, _, _, _ = make_ingester(
            journal, parser=FakeParser(RuntimeError("socket exploded"))
        )
        outcome = await ingest(use_case, stored_paper().id)
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.TRANSIENT
        assert outcome.failure.retryable is True

    async def test_a_parse_yielding_no_chunks_is_unparseable(self) -> None:
        # Sections with text but nothing chunkable would otherwise index a paper
        # with zero retrievable content and call it INDEXED.
        journal = Journal()
        empty = ParsedDocument(
            sections=(
                ParsedSection(
                    kind=SectionKind.REFERENCES,
                    path="References",
                    blocks=(
                        ParsedBlock(
                            kind=BlockKind.PARAGRAPH,
                            text="[1] Someone. A paper. 2020.",
                            page=9,
                            char_start=0,
                            char_end=27,
                        ),
                    ),
                    text="[1] Someone. A paper. 2020.",
                    raw_title="References",
                ),
            ),
            parser_version=FakeParser.version,
        )
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(empty))
        outcome = await ingest(use_case, stored_paper().id)
        assert outcome.failure is not None
        assert outcome.failure.failure_class is FailureClass.UNPARSEABLE


class TestDuplicateSectionPathsAreRefused:
    async def test_two_sections_sharing_a_path_raise_rather_than_mis_map(self) -> None:
        # D51. Chunks find their section row by path, so a duplicate would
        # attribute one section's chunks to the other's text — the citation
        # highlights the wrong span and nothing raises. This is the assertion
        # that turns that into a loud failure.
        journal = Journal()
        collided = document(
            parsed_section("Untitled section", "First orphan paragraph. " * 20),
            parsed_section("Untitled section", "Second orphan paragraph. " * 20),
        )
        use_case, _, _, _, _, _ = make_ingester(journal, parser=FakeParser(collided))
        with pytest.raises(InvariantViolation, match="duplicate section path"):
            await ingest(use_case, stored_paper().id)


class TestSectionsCarryWhatCitationResolutionNeeds:
    async def test_section_text_and_page_range_come_from_the_blocks(self) -> None:
        journal = Journal()
        use_case, sections, _, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)

        written = {s.path: s for s in sections.written}
        assert set(written) == {"Method", "Related Work"}
        for section in written.values():
            assert section.text
            assert section.page_start == 3 and section.page_end == 3
            assert section.raw_title == section.path

    async def test_a_breadcrumb_path_resolves_to_a_parent_section(self) -> None:
        # `sections.parent_id` is in the schema, and the breadcrumbs already
        # encode the hierarchy the parser recovered — leaving the column NULL
        # forever would be dead schema with a resolvable answer sitting next to
        # it. Paths are unique within a document (D51), which is what makes the
        # prefix lookup unambiguous.
        journal = Journal()
        nested = document(
            parsed_section("Experiments", "We ran several experiments. " * 20),
            parsed_section("Experiments > Ablations", "We ablate each component. " * 20),
        )
        use_case, sections, _, _, _, _ = make_ingester(journal, parser=FakeParser(nested))
        await ingest(use_case, stored_paper().id)

        by_path = {s.path: s for s in sections.written}
        assert by_path["Experiments"].parent_id is None
        assert by_path["Experiments > Ablations"].parent_id == by_path["Experiments"].id

    async def test_an_unresolvable_parent_stays_unknown_rather_than_invented(self) -> None:
        # A parent whose own section was dropped (a head with no body) simply
        # does not resolve. An unknown parent beats a fabricated one — the same
        # stance the parser takes on unnumbered heads.
        journal = Journal()
        orphaned = document(parsed_section("Missing > Ablations", "We ablate things. " * 20))
        use_case, sections, _, _, _, _ = make_ingester(journal, parser=FakeParser(orphaned))
        await ingest(use_case, stored_paper().id)
        assert sections.written[0].parent_id is None

    async def test_every_chunk_span_indexes_its_own_section_text(self) -> None:
        # FR-I5's round trip, at the boundary where the mapping is made: if a
        # chunk were attached to the wrong section row, its offsets would slice
        # text it never came from.
        journal = Journal()
        use_case, sections, chunks, _, _, _ = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)

        by_path = {s.path: s for s in sections.written}
        for chunk in chunks.written:
            source = by_path[chunk.provenance.section_path]
            span = source.text[chunk.provenance.char_start : chunk.provenance.char_end]
            if chunk.role is ChunkRole.CHILD:
                # `_join` promises a covering range, not verbatim equality, so
                # the exact claim is about the boundaries: the span starts where
                # the chunk's first sentence starts and ends where its last one
                # ends, and it is never shorter than the text it carries.
                assert span.split()[0] == chunk.text.split()[0]
                assert span.split()[-1] == chunk.text.split()[-1]
                assert len(span) >= len(chunk.text)
            else:
                assert span == chunk.text


class TestParsedMetadataReplacesTheProvisionalTitle:
    async def test_the_grobid_title_wins_over_the_filename(self) -> None:
        journal = Journal()
        use_case, _, _, _, _, papers = make_ingester(journal, parser=FakeParser(document()))
        await ingest(use_case, stored_paper().id)
        paper = papers.by_id[stored_paper().id]
        assert paper.title == "A Real Title"
        assert paper.parse_status is ParseStatus.PARSED
        assert paper.parser_version == FakeParser.version

    async def test_a_parse_without_a_title_keeps_the_filename(self) -> None:
        # An empty title would violate the entity's own invariant, and a paper
        # called "untitled" is less useful than one called by its filename.
        journal = Journal()
        titleless = ParsedDocument(
            sections=document().sections, title=None, parser_version=FakeParser.version
        )
        use_case, _, _, _, _, papers = make_ingester(
            journal, parser=FakeParser(titleless), paper=stored_paper(title="bert")
        )
        await ingest(use_case, stored_paper().id)
        assert papers.by_id[stored_paper().id].title == "bert"


def _embedder_of(use_case: IngestPaperUseCase) -> FakeEmbedder:
    # Reaching into the use case for the fake it was given, not for its internals.
    embedder = use_case._embedder
    assert isinstance(embedder, FakeEmbedder)
    return embedder


def _queue_of(use_case: IngestPaperUseCase) -> FakeQueue:
    queue = use_case._queue
    assert isinstance(queue, FakeQueue)
    return queue
