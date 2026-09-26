"""One real PDF, all the way to searchable vectors — phase 5's acceptance test.

Real Postgres, real Qdrant, real GROBID, real chunker. The embedder is a
deterministic stand-in rather than bge-base: loading the encoder is a 419 MB
download on a cold cache and NFR-T3 requires the default run to make zero
network calls, while what this file is actually testing is the *pipeline* — the
ordering, the state machine, the provenance round trip and the idempotency.
`tests/integration/test_vector_index.py` is where the real encoder is exercised.

The criteria under test come from doc 8 §M1:

  * chunks carry full provenance and version stamps  (FR-I5, NFR-M5)
  * a citation resolves to the exact source span     (FR-I5's round trip)
  * tables survive chunking intact                    (FR-I6, D41)
  * re-adding a paper does not duplicate work         (FR-I7)
  * an unusable PDF is reported, not silently indexed (FR-I9)
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.parsing import DocumentUnparseable, ParsedDocument
from scholiast.application.ports.storage import source_key
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import ChunkRole
from scholiast.domain.workspace import PaperState
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlPaperRepository,
    SqlSectionRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

from scholiast.application.ports.tenancy import TenantContext  # isort: skip

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "spike" / "fixtures"
GROBID_URL = "http://localhost:8070"
QDRANT_URL = "http://localhost:6333"
DIMENSIONS = 32


#: Migration 0005 made `users.password_hash` NOT NULL (D85). These rows exist
#: to own data, never to log in, so they get a string that is not a valid
#: Argon2 encoding: `Argon2idHasher.verify` returns False for it against any
#: password, so the account is unreachable by construction rather than by
#: nobody knowing the password.
UNUSABLE_HASH = "not-a-real-argon2-hash"


def _alive(url: str) -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(url).status_code == 200
    except httpx.HTTPError:
        return False


needs_grobid = pytest.mark.skipif(
    not _alive(f"{GROBID_URL}/api/isalive"), reason="GROBID is not running"
)
needs_qdrant = pytest.mark.skipif(
    not _alive(f"{QDRANT_URL}/readyz"), reason="Qdrant is not running"
)
needs_fixtures = pytest.mark.skipif(
    not FIXTURES.exists(), reason="fixture PDFs are gitignored; run spike/scripts/02 to fetch"
)


class DeterministicEmbedder:
    """A stand-in with the shape of the real thing, and none of its weight.

    Hash-based rather than random so a re-embed produces identical vectors —
    which is what lets the idempotency test distinguish "the same point was
    updated" from "a second point appeared".
    """

    version = "test-embedder/1.0.0-fp32"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.calls = 0

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self.calls += 1
        vectors: list[list[float]] = []
        for item in texts:
            seed = abs(hash(item)) % 9973
            vectors.append([((seed + i * 37) % 100) / 100.0 for i in range(DIMENSIONS)])
        return vectors


class UnreachableGrobid:
    """A parser that fails the way a GROBID outage fails.

    A stub rather than a wrong URL: the adapter's own classification of an
    unreachable service is already unit-tested
    (`test_grobid_being_down_is_transient_not_unparseable`), and depending on a
    closed port here would make this test depend on how the host refuses
    connections. What is under test is what the *pipeline and the queue* do with
    a retryable parse failure.
    """

    version = "unreachable/1.0.0"

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        raise DocumentUnparseable(FailureClass.TRANSIENT, "GROBID unreachable: ConnectError")


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "alembic", "-x", f"db_url={url}", "upgrade", "head"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


@pytest_asyncio.fixture
async def sessions(postgres: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(postgres)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as connection:
        await connection.execute(text("DELETE FROM ingestion_jobs"))
        await connection.execute(text("DELETE FROM chunks"))
        await connection.execute(text("DELETE FROM sections"))
        await connection.execute(text("DELETE FROM workspace_papers"))
        await connection.execute(text("DELETE FROM papers"))
        await connection.execute(text("DELETE FROM workspaces"))
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def index() -> AsyncIterator[QdrantVectorIndex]:
    store = QdrantVectorIndex(
        QDRANT_URL, collection=f"pipeline_{uuid.uuid4().hex[:8]}", dimensions=DIMENSIONS
    )
    await store.ensure_ready()
    yield store
    await store._client.delete_collection(store.collection)
    await store.close()


class Harness:
    """Everything phase 5 wires together, assembled once per test."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        index: QdrantVectorIndex,
        objects: Path,
    ) -> None:
        self.sessions = sessions
        self.index = index
        self.store = LocalObjectStore(objects)
        self.embedder = DeterministicEmbedder()
        self.worker = IngestionWorker(
            sessions=sessions,
            store=self.store,
            parser=GrobidParser(GROBID_URL),
            embedder=self.embedder,
            index=index,
            config=WorkerConfig(poll_seconds=0.05),
        )
        self.user_id = ""
        self.workspace_id = ""
        self.scope = TenantContext(user_id="placeholder")

    async def seed_tenant(self) -> None:
        self.user_id = str(uuid.uuid4())
        self.workspace_id = str(uuid.uuid4())
        async with self.sessions() as session:
            await session.execute(
                text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
                {"id": self.user_id, "email": f"{self.user_id}@test.local", "hash": UNUSABLE_HASH},
            )
            await session.execute(
                text(
                    "INSERT INTO workspaces (id, owner_id, name, state) "
                    "VALUES (:id, :owner, 'Pipeline', 'empty')"
                ),
                {"id": self.workspace_id, "owner": self.user_id},
            )
            await session.commit()
        self.scope = TenantContext(user_id=self.user_id)

    async def upload(self, pdf: bytes, filename: str):  # type: ignore[no-untyped-def]
        async with self.sessions() as session:
            use_case = AddPaperUseCase(
                workspaces=SqlWorkspaceRepository(session, self.scope),
                papers=SqlPaperRepository(session, self.scope),
                chunks=SqlChunkRepository(session, self.scope),
                store=self.store,
                queue=PostgresQueue(session, self.scope),
                owner_id=self.user_id,
            )
            result = await use_case.execute(
                workspace_id=self.workspace_id, data=pdf, filename=filename
            )
            await session.commit()
            return result

    async def drain(self, limit: int = 5) -> int:
        processed = 0
        for _ in range(limit):
            ran = await self.worker.run_once()
            if not ran:
                break
            processed += ran
        return processed

    async def membership_state(self, paper_id: str) -> str:
        async with self.sessions() as session:
            return (
                await session.execute(
                    text("SELECT state FROM workspace_papers WHERE paper_id = :p"),
                    {"p": paper_id},
                )
            ).scalar_one()

    async def paper_row(self, paper_id: str):  # type: ignore[no-untyped-def]
        async with self.sessions() as session:
            return (
                await session.execute(
                    text(
                        "SELECT title, parse_status, parse_failure_class, parser_version, "
                        "tei_key FROM papers WHERE id = :p"
                    ),
                    {"p": paper_id},
                )
            ).one()

    async def chunk_rows(self, paper_id: str):  # type: ignore[no-untyped-def]
        async with self.sessions() as session:
            return (
                await session.execute(
                    text(
                        "SELECT c.id, c.role, c.text, c.page, c.char_start, c.char_end, "
                        "c.chunker_version, c.embedder_version, s.path, s.text AS section_text "
                        "FROM chunks c JOIN sections s ON s.id = c.section_id "
                        "WHERE c.paper_id = :p ORDER BY c.ordinal"
                    ),
                    {"p": paper_id},
                )
            ).all()


@pytest_asyncio.fixture
async def harness(
    sessions: async_sessionmaker[AsyncSession], index: QdrantVectorIndex, tmp_path: Path
) -> AsyncIterator[Harness]:
    built = Harness(sessions, index, tmp_path / "objects")
    await built.seed_tenant()
    yield built


def paper_bytes(name: str = "attention-is-all-you-need.pdf") -> bytes:
    return (FIXTURES / name).read_bytes()


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestOnePdfBecomesSearchableEvidence:
    async def test_a_real_paper_reaches_indexed(self, harness: Harness) -> None:
        result = await harness.upload(paper_bytes(), "attention-is-all-you-need.pdf")
        assert result.state is PaperState.QUEUED
        assert result.job_id is not None

        assert await harness.drain() == 1
        assert await harness.membership_state(result.paper_id) == PaperState.INDEXED.value

        row = await harness.paper_row(result.paper_id)
        assert row.parse_status == "parsed"
        assert row.parse_failure_class is None
        assert row.parser_version
        # The provisional filename title is replaced by what GROBID read.
        assert row.title != "attention is all you need"

    async def test_the_retained_tei_is_the_bytes_grobid_returned(self, harness: Harness) -> None:
        # D43: a re-chunk ablation must read the TEI that produced today's
        # chunks. Storing a re-serialised or truncated copy would make the
        # comparison silently non-reproducible.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        row = await harness.paper_row(result.paper_id)
        assert row.tei_key is not None
        tei = await harness.store.get(row.tei_key)
        assert tei.startswith(b"<?xml") or tei.lstrip().startswith(b"<TEI")
        assert b"</TEI>" in tei
        assert row.parser_version.replace("/", "_") in row.tei_key

    async def test_the_original_pdf_is_kept_under_a_paper_scoped_key(
        self, harness: Harness
    ) -> None:
        pdf = paper_bytes()
        result = await harness.upload(pdf, "attention.pdf")
        assert await harness.store.get(source_key(result.paper_id)) == pdf

    async def test_every_chunk_is_version_stamped(self, harness: Harness) -> None:
        # NFR-M5: this is what makes "was it the reranker or the re-chunk?"
        # answerable, and what lets a re-index target only what changed.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        rows = await harness.chunk_rows(result.paper_id)
        assert rows
        assert all(r.chunker_version for r in rows)
        embedded = [r for r in rows if r.role != ChunkRole.PARENT.value]
        assert embedded
        assert all(r.embedder_version == DeterministicEmbedder.version for r in embedded)
        # Parents are context, not candidates: embedding them would double the
        # index for no retrieval gain.
        assert all(r.embedder_version is None for r in rows if r.role == ChunkRole.PARENT.value)

    async def test_a_citation_resolves_to_the_exact_source_span(self, harness: Harness) -> None:
        # FR-I5's round trip, through the real schema: chunk → its section row →
        # slice. This is the acceptance criterion "clicking a citation resolves
        # to the correct paper, section, page and character span".
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        rows = await harness.chunk_rows(result.paper_id)
        checked = 0
        for row in rows:
            span = row.section_text[row.char_start : row.char_end]
            assert row.page >= 1
            if row.role == ChunkRole.CHILD.value:
                # `_join` promises a *covering* range, not verbatim equality:
                # joining normalises inter-sentence whitespace, and a child may
                # bridge two prose blocks with an excluded table between them.
                # What is exactly true is that the span begins where the chunk's
                # first sentence begins and ends where its last one ends — so a
                # one-character drift in either offset changes a boundary word.
                span_words = span.split()
                chunk_words = row.text.split()
                assert span_words[0] == chunk_words[0]
                assert span_words[-1] == chunk_words[-1]
                # Separators are never shorter than the single space the join
                # uses, so a span shorter than its own text means the text
                # contains something the span does not cover.
                assert len(span) >= len(row.text)
            else:
                assert span == row.text
            checked += 1
        assert checked > 50, "too few chunks to call this a round-trip test"

    async def test_tables_survive_chunking_intact(self, harness: Harness) -> None:
        # D41's honest claim: a table is never split mid-row and never loses its
        # caption. Cell fidelity is an M2 measurement, deliberately not asserted.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        tables = [r for r in await harness.chunk_rows(result.paper_id) if r.role == "table"]
        assert tables, "the fixture has tables; finding none means they never entered the index"
        for table in tables:
            rows_in_table = [line for line in table.text.splitlines() if line.startswith("|")]
            assert all(line.endswith("|") for line in rows_in_table), "a row was split"

    async def test_the_vectors_are_searchable_and_scoped_to_the_paper(
        self, harness: Harness
    ) -> None:
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        rows = await harness.chunk_rows(result.paper_id)
        embeddable = [r for r in rows if r.role != ChunkRole.PARENT.value]
        assert await harness.index.count(paper_ids=[result.paper_id]) == len(embeddable)

        probe = (await harness.embedder.embed(["anything"], role=TextRole.QUERY))[0]
        hits = await harness.index.search(probe, paper_ids=[result.paper_id], limit=5)
        assert hits
        assert {h.paper_id for h in hits} == {result.paper_id}
        assert {h.chunk_id for h in hits} <= {r.id for r in rows}

    async def test_section_paths_are_unique_in_a_real_paper(self, harness: Harness) -> None:
        # D51 against real GROBID output rather than hand-written TEI. If the
        # parser ever stops guaranteeing this, the pipeline raises — but this is
        # the test that says whether real papers actually collide.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        async with harness.sessions() as session:
            paths = (
                (
                    await session.execute(
                        text("SELECT path FROM sections WHERE paper_id = :p"),
                        {"p": result.paper_id},
                    )
                )
                .scalars()
                .all()
            )
        assert len(paths) == len(set(paths))


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestIngestionIsIdempotent:
    async def test_re_running_the_same_job_converges(self, harness: Harness) -> None:
        # FR-I7 across a *redelivery*, which is the case an at-least-once queue
        # guarantees will happen. Appending sections or duplicating points would
        # show up here as growth.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        before = await harness.chunk_rows(result.paper_id)
        vectors_before = await harness.index.count(paper_ids=[result.paper_id])

        async with harness.sessions() as session:
            await session.execute(
                text(
                    "UPDATE ingestion_jobs SET state = 'queued', "
                    "visible_at = now() - interval '1 hour' WHERE paper_id = :p"
                ),
                {"p": result.paper_id},
            )
            await session.commit()
        assert await harness.drain() == 1

        # Converged, not merely unchanged. Asserting only that the rows are the
        # same passes when the re-run *crashed* and rolled back — which is what
        # an append-instead-of-replace section write actually does, via a chunk
        # primary-key conflict. The second run has to finish.
        assert await harness.membership_state(result.paper_id) == PaperState.INDEXED.value
        async with harness.sessions() as session:
            job_state = (
                await session.execute(
                    text("SELECT state FROM ingestion_jobs WHERE paper_id = :p"),
                    {"p": result.paper_id},
                )
            ).scalar_one()
        assert job_state == PaperState.INDEXED.value

        after = await harness.chunk_rows(result.paper_id)
        assert [r.id for r in after] == [r.id for r in before]
        assert await harness.index.count(paper_ids=[result.paper_id]) == vectors_before
        async with harness.sessions() as session:
            sections = (
                await session.execute(
                    text("SELECT count(*) FROM sections WHERE paper_id = :p"),
                    {"p": result.paper_id},
                )
            ).scalar_one()
        assert sections == len({r.path for r in after}), (
            "sections accumulated instead of converging"
        )

    async def test_re_uploading_the_same_file_does_no_work(self, harness: Harness) -> None:
        pdf = paper_bytes()
        first = await harness.upload(pdf, "attention.pdf")
        await harness.drain()
        calls = harness.embedder.calls

        second = await harness.upload(pdf, "attention.pdf")
        assert second.paper_id == first.paper_id
        assert second.deduplicated is True
        assert second.job_id is None
        assert await harness.drain() == 0
        assert harness.embedder.calls == calls, "re-upload re-embedded the paper"


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestUnusablePdfsAreReportedNotIndexed:
    async def test_an_encrypted_pdf_ends_unsupported_with_a_reason(self, harness: Harness) -> None:
        # FR-I9: detected and reported as such rather than silently indexing
        # garbage — and recorded on the *paper* (D44), so a second workspace
        # adding the same file learns why without re-parsing it.
        result = await harness.upload(paper_bytes("_broken-encrypted.pdf"), "encrypted.pdf")
        await harness.drain()

        assert await harness.membership_state(result.paper_id) == PaperState.UNSUPPORTED.value
        row = await harness.paper_row(result.paper_id)
        assert row.parse_status == "failed"
        assert row.parse_failure_class == "unparseable"
        assert await harness.chunk_rows(result.paper_id) == []
        assert await harness.index.count(paper_ids=[result.paper_id]) == 0

    async def test_a_scanned_pdf_is_told_apart_from_a_corrupt_one(self, harness: Harness) -> None:
        # A scan is a different failure: OCR would fix it (FR-I12), so the user
        # is told that rather than "corrupted", which is wrong and unactionable.
        result = await harness.upload(paper_bytes("_broken-scanned.pdf"), "scan.pdf")
        await harness.drain()
        row = await harness.paper_row(result.paper_id)
        assert row.parse_failure_class == "scanned_image"

    async def test_a_failed_paper_is_not_queryable(self, harness: Harness) -> None:
        # FR-W5: only INDEXED papers enter the retrieval scope, so an answer
        # cannot cite chunks that do not exist.
        await harness.upload(paper_bytes("_broken-encrypted.pdf"), "encrypted.pdf")
        await harness.drain()
        async with harness.sessions() as session:
            papers = SqlPaperRepository(session, harness.scope)
            assert await papers.queryable_paper_ids(harness.workspace_id) == ()


@needs_qdrant
@needs_fixtures
class TestATransientParseFailureIsNotATerminalOutcome:
    """E-1, end to end: the paper row must not claim an outcome it does not have.

    Needs no GROBID — the parser is stubbed precisely so a *retryable* failure
    can be produced deterministically. Everything else is real: the real worker,
    the real Postgres queue, the real retry policy.
    """

    def _worker(self, harness: Harness) -> IngestionWorker:
        return IngestionWorker(
            sessions=harness.sessions,
            store=harness.store,
            parser=UnreachableGrobid(),
            embedder=harness.embedder,
            index=harness.index,
            config=WorkerConfig(poll_seconds=0.05),
        )

    async def test_the_job_stays_queued_for_retry(self, harness: Harness) -> None:
        # The first half of the requirement: the existing bounded-retry
        # mechanism is untouched — the job goes back to QUEUED with the delivery
        # counted and a backoff applied (D48).
        result = await harness.upload(paper_bytes(), "attention.pdf")
        assert await self._worker(harness).run_once() == 1

        assert await harness.membership_state(result.paper_id) == PaperState.QUEUED.value
        async with harness.sessions() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT state, attempts, last_error, visible_at > now() AS backing_off "
                        "FROM ingestion_jobs WHERE paper_id = :p"
                    ),
                    {"p": result.paper_id},
                )
            ).one()
        assert row.state == PaperState.QUEUED.value
        assert row.attempts == 1
        assert row.backing_off is True
        assert "unreachable" in row.last_error

    async def test_the_paper_is_not_recorded_as_terminally_failed(self, harness: Harness) -> None:
        # The second half: `ParseStatus.FAILED` reports itself terminal, and
        # D44's purpose is that a second workspace can trust `parse_status`
        # without re-parsing. A service outage is not an outcome.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await self._worker(harness).run_once()

        row = await harness.paper_row(result.paper_id)
        assert row.parse_status != "failed"
        assert row.parse_failure_class is None

    async def test_a_later_success_still_indexes_the_paper(self, harness: Harness) -> None:
        # Positive control, and the point of calling it transient: the same job,
        # redelivered once the service is back, completes normally. A guard that
        # broke the retry path would pass both tests above.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await self._worker(harness).run_once()

        async with harness.sessions() as session:
            await session.execute(
                text(
                    "UPDATE ingestion_jobs SET visible_at = now() - interval '1 hour' "
                    "WHERE paper_id = :p"
                ),
                {"p": result.paper_id},
            )
            await session.commit()

        if not _alive(f"{GROBID_URL}/api/isalive"):
            pytest.skip("the recovery half needs a real GROBID")
        assert await harness.drain() == 1
        assert await harness.membership_state(result.paper_id) == PaperState.INDEXED.value
        assert (await harness.paper_row(result.paper_id)).parse_status == "parsed"


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestTheWorkerStaysInsideItsTenant:
    async def test_ingestion_writes_under_the_owners_scope(self, harness: Harness) -> None:
        # D50: only the claim is cross-tenant. Everything the pipeline writes
        # goes through layer 2, so a second tenant sees none of it.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()

        stranger = TenantContext(user_id=str(uuid.uuid4()))
        async with harness.sessions() as session:
            assert await SqlPaperRepository(session, stranger).get(result.paper_id) is None
            assert await SqlChunkRepository(session, stranger).list_for_paper(result.paper_id) == []

    async def test_a_stranger_cannot_wipe_another_tenants_sections(self, harness: Harness) -> None:
        # `replace_for_paper` deletes and cascades to `chunks`, so an unscoped
        # variant would let anyone who guessed a paper id destroy another
        # tenant's parsed content. Verified by attempting it.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()
        before = await harness.chunk_rows(result.paper_id)
        assert before

        stranger = TenantContext(user_id=str(uuid.uuid4()))
        async with harness.sessions() as session:
            written = await SqlSectionRepository(session, stranger).replace_for_paper(
                result.paper_id, []
            )
            await session.commit()

        assert written == 0
        assert [r.id for r in await harness.chunk_rows(result.paper_id)] == [r.id for r in before]

    async def test_the_owner_can_still_replace_its_own_sections(self, harness: Harness) -> None:
        # Positive control: a predicate that refused everything would pass the
        # test above and break every re-ingest.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        await harness.drain()
        async with harness.sessions() as session:
            sections = await SqlSectionRepository(session, harness.scope).list_for_paper(
                result.paper_id
            )
            assert sections
            written = await SqlSectionRepository(session, harness.scope).replace_for_paper(
                result.paper_id, sections
            )
            await session.commit()
        assert written == len(sections)

    async def test_a_deleted_workspace_retires_its_jobs(self, harness: Harness) -> None:
        # Ingesting into a workspace the user has deleted is work nobody will
        # ever see, and the worker cannot resolve an owner for it.
        result = await harness.upload(paper_bytes(), "attention.pdf")
        async with harness.sessions() as session:
            await SqlWorkspaceRepository(session, harness.scope).soft_delete(harness.workspace_id)
            await session.commit()

        assert await harness.drain() == 1
        assert await harness.chunk_rows(result.paper_id) == []
        async with harness.sessions() as session:
            state = (
                await session.execute(
                    text("SELECT state FROM ingestion_jobs WHERE paper_id = :p"),
                    {"p": result.paper_id},
                )
            ).scalar_one()
        assert state == PaperState.INDEXED.value, "the orphaned job was left claimable"
