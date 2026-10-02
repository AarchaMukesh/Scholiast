"""The ingestion pipeline: parse → chunk → embed → index (§5.4.2, §5.7.3).

The asynchronous half. It is given a claimed job and ports, and it drives the
§4.4.1 state machine through `PaperMembership.transition_to` rather than
assigning states directly — so an illegal move raises here instead of quietly
persisting a lifecycle the documents forbid.

Four properties this file exists to hold:

**Failure is classified, never swallowed.** Every exit is a `FailureClass`, which
is what decides both the retry and the user's message (FR-I8, FR-I9). An
unclassified exception becomes TRANSIENT — the one default that is safe, because
it retries a bounded number of times and then dead-letters rather than declaring
a paper permanently unusable on the strength of a bug.

**Postgres is written before vectors** (D53). Vectors orphaned by a crash are
what §5.6.4's sweep is built to find and NFR-R6 declares rebuildable; a chunk row
with no vector is a chunk that can never be retrieved, and nothing anywhere
reports it. So chunks are written unstamped, vectors are upserted, and only then
are the rows stamped — which makes `Chunk.is_indexed` mean what it says.

**Re-running converges.** Sections are replaced rather than appended (D52), chunk
ids are deterministic, and D46's point ids upsert. That is FR-I7 holding across a
redelivery, not just across a second upload.

**Section paths are asserted unique** (D51). Chunks find their section row by
path, so a duplicate would attribute a chunk to the wrong text and highlight the
wrong span with nothing raising. The parser guarantees uniqueness; this checks it
rather than trusting it, because the cost of being wrong is a silent citation
defect and the cost of checking is one set comparison.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from scholiast.application.ports.ml import Embedder, GpuBudgetExceeded, TextRole
from scholiast.application.ports.parsing import (
    DocumentParser,
    DocumentUnparseable,
    ParsedDocument,
)
from scholiast.application.ports.queue import Queue
from scholiast.application.ports.repositories import (
    ChunkRepository,
    PaperRepository,
    SectionRepository,
)
from scholiast.application.ports.storage import ObjectNotFound, ObjectStore, source_key, tei_key
from scholiast.application.ports.unit_of_work import UnitOfWork
from scholiast.application.ports.vector import VectorIndex, VectorRecord
from scholiast.domain.chunking import (
    SECTION_PATH_SEPARATOR,
    ChunkingConfig,
    ParsedSection,
    StructureAwareChunker,
)
from scholiast.domain.errors import FailureClass, IngestionFailure, InvariantViolation
from scholiast.domain.paper import Chunk, Paper, ParseStatus, Section
from scholiast.domain.workspace import PaperMembership, PaperState


@dataclass(frozen=True, slots=True)
class IngestionOutcome:
    """Where one attempt ended, and what it produced on the way.

    A failure carries the classified `IngestionFailure` and **not** a terminal
    state. Where a failure lands depends on how many attempts have already been
    spent, which only the queue knows (D48) — so reporting a state here would be
    a guess the worker then has to ignore.
    """

    paper_id: str
    sections: int = 0
    chunks: int = 0
    embedded: int = 0
    failure: IngestionFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure is None


class IngestPaperUseCase:
    """Run one claimed ingestion job to a terminal or retryable state."""

    def __init__(
        self,
        *,
        papers: PaperRepository,
        sections: SectionRepository,
        chunks: ChunkRepository,
        store: ObjectStore,
        parser: DocumentParser,
        embedder: Embedder,
        index: VectorIndex,
        queue: Queue,
        uow: UnitOfWork,
        chunking: ChunkingConfig | None = None,
        embed_batch_size: int | None = None,
    ) -> None:
        self._papers = papers
        self._sections = sections
        self._chunks = chunks
        self._store = store
        self._parser = parser
        self._embedder = embedder
        self._index = index
        self._queue = queue
        self._uow = uow
        self._chunker = StructureAwareChunker(chunking)
        self._embed_batch_size = embed_batch_size

    async def execute(
        self, *, job_id: str, workspace_id: str, paper_id: str, membership: PaperMembership
    ) -> IngestionOutcome:
        try:
            return await self._run(
                job_id=job_id,
                workspace_id=workspace_id,
                paper_id=paper_id,
                membership=membership,
            )
        except DocumentUnparseable as exc:
            return _failed(paper_id, exc.failure_class, exc.detail)
        except ObjectNotFound as exc:
            # The blob the job points at is gone. Not retryable in any useful
            # sense — the bytes are not coming back — and distinct enough from a
            # bad PDF to be worth its own classification.
            return _failed(paper_id, FailureClass.SOURCE_UNAVAILABLE, str(exc))
        except GpuBudgetExceeded as exc:
            # FR-I10 / D98. The embedder already halved to a single item and
            # still could not allocate, so this is capacity rather than sizing —
            # nothing this process can fix by trying again immediately.
            # Retryable because the GPU may well be free in thirty seconds, which
            # is exactly what the backoff is for.
            return _failed(paper_id, FailureClass.RESOURCE_EXHAUSTED, str(exc))
        except InvariantViolation:
            # A domain rule was broken: the same input will break it again, so
            # retrying is pure waste. Re-raised rather than classified, because
            # this is a bug in this code and must be loud.
            raise
        except Exception as exc:
            # Anything unclassified is TRANSIENT: bounded retries then a
            # dead-letter. The alternative default — declaring the paper
            # permanently unusable — would let one bug mark a whole corpus
            # unsupported with no way back (FR-I8).
            return _failed(paper_id, FailureClass.TRANSIENT, f"{type(exc).__name__}: {exc}"[:500])

    # -- the pipeline ------------------------------------------------------

    async def _run(
        self, *, job_id: str, workspace_id: str, paper_id: str, membership: PaperMembership
    ) -> IngestionOutcome:
        paper = await self._papers.get(paper_id)
        if paper is None:
            raise InvariantViolation(f"job {job_id!r} references unreachable paper {paper_id!r}")

        # -- parse ---------------------------------------------------------
        await self._advance(job_id, workspace_id, membership, PaperState.PARSING)
        pdf = await self._store.get(source_key(paper_id))
        try:
            # The stored key's basename, not the paper title: this reaches the
            # parser's error messages, and a title there would read as though
            # a file by that name had been uploaded.
            parsed = await self._parser.parse(pdf, filename=f"{paper_id}.pdf")
        except DocumentUnparseable as exc:
            if not exc.failure_class.retryable:
                # Recorded on the *paper*, not only the membership (D44): parsing
                # happens once per paper, so a second workspace adding this file
                # must learn that it failed and why without re-parsing it.
                #
                # Terminal classes only. `ParseStatus.FAILED` reports itself as
                # terminal, so writing it while the queue still has retries left
                # would have the paper row claim a final outcome for a job that
                # is merely in backoff — and D44's entire purpose is that a
                # second workspace can trust that row without re-parsing. A
                # GROBID outage is not an outcome; it is the absence of one, and
                # the membership's QUEUED already says "retrying" (§4.4.1).
                await self._record_parse_failure(paper, exc.failure_class)
            raise

        _assert_unique_paths(parsed.sections, paper_id=paper_id)
        stored_tei = await self._persist_tei(paper, parsed)

        # -- chunk ---------------------------------------------------------
        await self._advance(job_id, workspace_id, membership, PaperState.CHUNKING)
        result = self._chunker.chunk(paper_id, parsed.sections)
        if not result.chunks:
            raise DocumentUnparseable(
                FailureClass.UNPARSEABLE,
                f"parsed {len(parsed.sections)} sections but produced no chunks",
            )

        sections = _to_sections(parsed, paper_id=paper_id)
        await self._sections.replace_for_paper(paper_id, sections)
        section_ids = {section.path: section.id for section in sections}
        await self._chunks.add_many(result.chunks, section_ids=section_ids)
        await self._papers.update(
            _parsed_metadata(paper, parsed, tei_key=stored_tei),
        )

        # -- embed and index -----------------------------------------------
        await self._advance(job_id, workspace_id, membership, PaperState.EMBEDDING)
        embeddable = result.embeddable
        embedded = await self._embed_and_index(
            paper=paper, parsed=parsed, chunks=embeddable, workspace_id=workspace_id
        )

        await self._advance(job_id, workspace_id, membership, PaperState.INDEXED)
        return IngestionOutcome(
            paper_id=paper_id,
            sections=len(sections),
            chunks=len(result.chunks),
            embedded=embedded,
        )

    async def _embed_and_index(
        self,
        *,
        paper: Paper,
        parsed: ParsedDocument,
        chunks: list[Chunk],
        workspace_id: str,  # noqa: ARG002 - see D36
    ) -> int:
        """Embed children and tables, upsert their vectors, then stamp the rows.

        `workspace_id` is deliberately unused: vectors are filtered by paper id,
        never by workspace (D36), because a public paper is one shared row and a
        workspace-keyed payload would duplicate every vector per tenant.
        """
        if not chunks:
            return 0

        title = parsed.title or paper.title
        # The contextual header is applied here, at the only place that knows
        # both the chunk and its paper (ML-3). It is ~20 tokens and no LLM call,
        # and it is what marks a Related Work chunk as describing other work.
        texts = [chunk.embedding_input(title, paper.year) for chunk in chunks]
        vectors = await self._embedder.embed(
            texts, role=TextRole.PASSAGE, batch_size=self._embed_batch_size
        )
        if len(vectors) != len(chunks):
            raise InvariantViolation(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        await self._index.upsert(
            [
                VectorRecord(
                    chunk_id=chunk.id,
                    paper_id=paper.id,
                    vector=tuple(vector),
                    section_kind=chunk.provenance.section_kind,
                    chunker_version=chunk.chunker_version,
                    embedder_version=self._embedder.version,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )
        # Only now: a stamped row with no vector would be a chunk that claims to
        # be searchable and is not (D53).
        return await self._chunks.stamp_embedded(
            [chunk.id for chunk in chunks], self._embedder.version
        )

    # -- persistence helpers ----------------------------------------------

    async def _persist_tei(self, paper: Paper, parsed: ParsedDocument) -> str | None:
        """Retain the raw TEI so re-chunking ablations are reproducible (D43)."""
        if not parsed.raw_tei:
            return None
        key = tei_key(paper.id, parsed.parser_version or "unknown")
        await self._store.put(key, parsed.raw_tei.encode("utf-8"), content_type="application/xml")
        return key

    async def _record_parse_failure(self, paper: Paper, failure_class: FailureClass) -> None:
        paper.parse_status = ParseStatus.FAILED
        paper.parse_failure_class = failure_class
        await self._papers.update(paper)

    async def _advance(
        self, job_id: str, workspace_id: str, membership: PaperMembership, state: PaperState
    ) -> None:
        """Move the membership and the job together, durably (§5.4.2).

        The domain validates the move, the membership row records what the user
        sees, the queue records the stage and extends the lease, and the commit
        makes all three visible at once. Committing here rather than at the end
        of the job is what makes the lease mean anything: without it every stage
        transition would appear at the same instant — the end — and the job would
        be protected by an open row lock rather than by the visibility timeout
        the design actually specifies.
        """
        membership.transition_to(state)
        await self._papers.set_membership_state(workspace_id, membership.paper_id, state)
        await self._queue.heartbeat(job_id, state=state)
        await self._uow.commit()


def _failed(paper_id: str, failure_class: FailureClass, detail: str) -> IngestionOutcome:
    return IngestionOutcome(
        paper_id=paper_id, failure=IngestionFailure(failure_class=failure_class, detail=detail)
    )


def _assert_unique_paths(sections: tuple[ParsedSection, ...], *, paper_id: str) -> None:
    """D51's guarantee, checked rather than assumed."""
    seen: set[str] = set()
    for section in sections:
        if section.path in seen:
            raise InvariantViolation(
                f"paper {paper_id}: duplicate section path {section.path!r} — chunks are "
                "mapped to section rows by path, so a duplicate would silently attribute "
                "a chunk to another section's text (D51)"
            )
        seen.add(section.path)


def _to_sections(parsed: ParsedDocument, *, paper_id: str) -> list[Section]:
    """Turn parser output into the rows `sections` holds (D42, D52).

    Page range comes from the blocks rather than being carried separately: the
    blocks already know their pages exactly (D40), and a second source for the
    same fact is a second thing to get wrong.
    """
    sections: list[Section] = []
    # Breadcrumb paths already encode the hierarchy the parser could recover
    # ("Results > Ablations"), so `sections.parent_id` is resolvable by looking
    # up the prefix rather than left permanently NULL. Paths are unique within a
    # document (D51), which is what makes the lookup unambiguous; a parent whose
    # own section was dropped simply does not resolve, and an unknown parent is
    # better than an invented one.
    ids_by_path: dict[str, str] = {}
    for order, parsed_section in enumerate(parsed.sections):
        pages = [block.page for block in parsed_section.blocks]
        section_id = str(uuid.uuid4())
        parent_path = (
            parsed_section.path.rsplit(SECTION_PATH_SEPARATOR, 1)[0]
            if SECTION_PATH_SEPARATOR in parsed_section.path
            else None
        )
        sections.append(
            Section(
                id=section_id,
                paper_id=paper_id,
                kind=parsed_section.kind,
                raw_title=parsed_section.raw_title,
                path=parsed_section.path,
                order_index=order,
                page_start=min(pages),
                page_end=max(pages),
                text=parsed_section.text,
                parent_id=ids_by_path.get(parent_path) if parent_path else None,
            )
        )
        ids_by_path[parsed_section.path] = section_id
    return sections


def _parsed_metadata(paper: Paper, parsed: ParsedDocument, *, tei_key: str | None) -> Paper:
    """The paper as the parse leaves it.

    The provisional filename title is replaced only if GROBID found one: an
    empty title would violate the entity's own invariant, and a paper called
    "untitled" is less useful than a paper called by its filename (D55).
    """
    paper.title = parsed.title or paper.title
    paper.authors = parsed.authors or paper.authors
    paper.abstract = parsed.abstract or paper.abstract
    paper.parse_status = ParseStatus.PARSED
    paper.parse_failure_class = None
    paper.parser_version = parsed.parser_version
    paper.tei_key = tei_key
    return paper
