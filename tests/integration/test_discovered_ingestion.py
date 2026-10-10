"""A discovered paper, added by identifier, all the way to INDEXED (D133, D134, D136).

M4's new criterion, hermetically: "a discovered arXiv paper reaches INDEXED through
the guarded fetcher, and re-adding it by the same identifier creates no new job or
chunks". Real Postgres, the real repositories, queue and worker; a fake source, a
fake fetcher, a stub parser and an in-memory vector index, so this never skips and
never touches the network (NFR-T3). The real fetcher, GROBID and arXiv are phase
6a's live check, which is run by hand and reported, not part of this suite.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping, Sequence
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from structlog.testing import capture_logs
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.fetch import FetchAttempt, FetchedPdf, FetchFailed
from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.parsing import ParsedDocument
from scholiast.application.ports.storage import source_key
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.ports.vector import VectorRecord
from scholiast.application.use_cases.add_discovered import (
    AddDiscoveredPapersUseCase,
    DiscoveredRef,
)
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.domain.chunking import BlockKind, ParsedBlock, ParsedSection
from scholiast.domain.discovery import LinkKind, OpenAccessLink, PaperCandidate
from scholiast.domain.errors import FailureClass
from scholiast.domain.fetch_policy import host_allowlist
from scholiast.domain.paper import SectionKind
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlPaperRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
UNUSABLE_HASH = "not-a-real-argon2-hash"
DIMENSIONS = 8
ARXIV_ID = "2004.04906v3"
LINK = f"https://arxiv.org/pdf/{ARXIV_ID}"
PDF = b"%PDF-1.7\n" + b"dense passage retrieval " * 200 + b"\n%%EOF\n"
#: Not the port's default of 300 s, so a heartbeat that ignored the worker's
#: configured lease would show.
LEASE_SECONDS = 1234.0


class OneArxivRecord:
    """arXiv, knowing exactly one paper."""

    name = "arxiv"

    def is_valid_id(self, source_id: str) -> bool:
        return source_id == ARXIV_ID

    async def search(self, query: str, *, limit: int) -> object:
        raise AssertionError("never searched here")

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        return {
            ARXIV_ID: PaperCandidate(
                source="arxiv",
                source_id=ARXIV_ID,
                rank=1,
                title="Dense Passage Retrieval for Open-Domain Question Answering",
                authors=("Vladimir Karpukhin",),
                year=2020,
                arxiv_id="2004.04906",
                oa_links=(OpenAccessLink(LINK, LinkKind.PDF),),
                adapter_version="arxiv-2",
            )
        }


class RecordingFetcher:
    """The fetch port, answering from memory and recording what it was asked."""

    def __init__(self, result: bytes | FetchFailed = PDF) -> None:
        self.result = result
        self.calls: list[tuple[str, ...]] = []
        #: Run during the fetch, while the job is FETCHING.
        self.during: Callable[[], Awaitable[None]] | None = None

    async def fetch(self, links: Sequence[str]) -> FetchedPdf:
        self.calls.append(tuple(links))
        if self.during is not None:
            await self.during()
        if isinstance(self.result, FetchFailed):
            raise self.result
        return FetchedPdf(self.result, links[0], links[0], (FetchAttempt(links[0], "fetched"),))


class StubParser:
    version = "stub-parser/1.0.0"

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        text_ = "We train a dense retriever over passages. " * 30
        section = ParsedSection(
            kind=SectionKind.METHOD,
            path="Method",
            blocks=(
                ParsedBlock(
                    kind=BlockKind.PARAGRAPH, text=text_, page=2, char_start=0, char_end=len(text_)
                ),
            ),
            text=text_,
            raw_title="Method",
        )
        return ParsedDocument(
            sections=(section,),
            title="GROBID's guess at a title",
            parser_version=self.version,
        )


class FixedEmbedder:
    version = "test-embedder/1.0.0-fp32"
    dimensions = DIMENSIONS

    def __init__(self) -> None:
        self.inputs: list[str] = []

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self.inputs.extend(texts)
        return [[0.1 * (i % 7)] * DIMENSIONS for i, _ in enumerate(texts)]


class MemoryIndex:
    def __init__(self) -> None:
        self.records: dict[str, VectorRecord] = {}

    async def ensure_ready(self) -> None: ...

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self.records.update({r.chunk_id: r for r in records})
        return len(records)

    async def search(self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int):  # type: ignore[no-untyped-def]
        return []

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        doomed = [k for k, r in self.records.items() if r.paper_id in paper_ids]
        for key in doomed:
            del self.records[key]
        return len(doomed)

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        return sum(1 for r in self.records.values() if paper_ids is None or r.paper_id in paper_ids)


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


class Harness:
    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], objects: Path, fetcher: RecordingFetcher
    ) -> None:
        self.sessions = sessions
        self.store = LocalObjectStore(objects)
        self.index = MemoryIndex()
        self.embedder = FixedEmbedder()
        self.fetcher = fetcher
        self.worker = IngestionWorker(
            sessions=sessions,
            store=self.store,
            parser=StubParser(),
            embedder=self.embedder,
            index=self.index,
            config=WorkerConfig(poll_seconds=0.05, lease_seconds=LEASE_SECONDS),
            fetcher=fetcher,
        )
        self.scope = TenantContext(user_id="placeholder")
        self.workspace_id = ""

    async def seed(self) -> None:
        user, workspace = str(uuid.uuid4()), str(uuid.uuid4())
        async with self.sessions() as session:
            await session.execute(
                text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
                {"id": user, "email": f"{user}@test.local", "hash": UNUSABLE_HASH},
            )
            await session.execute(
                text(
                    "INSERT INTO workspaces (id, owner_id, name, state) "
                    "VALUES (:id, :owner, 'Discovered', 'empty')"
                ),
                {"id": workspace, "owner": user},
            )
            await session.commit()
        self.scope, self.workspace_id = TenantContext(user_id=user), workspace

    async def add(self):  # type: ignore[no-untyped-def]
        async with self.sessions() as session:
            use_case = AddDiscoveredPapersUseCase(
                workspaces=SqlWorkspaceRepository(session, self.scope),
                papers=SqlPaperRepository(session, self.scope),
                chunks=SqlChunkRepository(session, self.scope),
                queue=PostgresQueue(session, self.scope),
                sources={"arxiv": OneArxivRecord()},
                allowed_hosts=host_allowlist(["arxiv.org"]),
                owner_id=self.scope.user_id,
            )
            (added,) = await use_case.execute(
                workspace_id=self.workspace_id, refs=[DiscoveredRef("arxiv", ARXIV_ID)]
            )
            await session.commit()
            return added.result

    async def upload(self, data: bytes):  # type: ignore[no-untyped-def]
        async with self.sessions() as session:
            result = await AddPaperUseCase(
                workspaces=SqlWorkspaceRepository(session, self.scope),
                papers=SqlPaperRepository(session, self.scope),
                chunks=SqlChunkRepository(session, self.scope),
                store=self.store,
                queue=PostgresQueue(session, self.scope),
                owner_id=self.scope.user_id,
            ).execute(workspace_id=self.workspace_id, data=data, filename="dpr.pdf")
            await session.commit()
            return result

    async def drain(self) -> int:
        ran = 0
        while processed := await self.worker.run_once():
            ran += processed
        return ran

    async def scalar(self, sql: str, **params: object) -> object:
        async with self.sessions() as session:
            return (await session.execute(text(sql), params)).scalar()


@pytest_asyncio.fixture
async def sessions(postgres: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(postgres)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _harness(
    sessions: async_sessionmaker[AsyncSession], tmp_path: Path, fetcher: RecordingFetcher
) -> Harness:
    built = Harness(sessions, tmp_path / "objects", fetcher)
    await built.seed()
    return built


class TestADiscoveredArxivPaperReachesIndexed:
    async def test_it_is_fetched_parsed_embedded_and_indexed(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        h = await _harness(sessions, tmp_path, RecordingFetcher())
        added = await h.add()

        assert await h.drain() == 1
        assert h.fetcher.calls == [(LINK,)], "the job did not carry the add's links"
        state = await h.scalar(
            "SELECT state FROM workspace_papers WHERE paper_id = :p", p=uuid.UUID(added.paper_id)
        )
        assert state == "indexed"
        stored_hash = await h.scalar(
            "SELECT content_hash FROM papers WHERE id = :p", p=uuid.UUID(added.paper_id)
        )
        assert stored_hash == hashlib.sha256(PDF).hexdigest()
        assert await h.store.get(source_key(added.paper_id)) == PDF
        assert await h.index.count(paper_ids=[added.paper_id]) > 0
        # A discovered paper's job is backfill: an upload is claimed first (D139).
        assert await h.scalar("SELECT priority FROM ingestion_jobs") == 0

    async def test_the_sources_title_is_kept_and_is_the_one_embedded(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        # D135, through the real repository's write-back.
        h = await _harness(sessions, tmp_path, RecordingFetcher())
        added = await h.add()
        await h.drain()

        title = await h.scalar(
            "SELECT title FROM papers WHERE id = :p", p=uuid.UUID(added.paper_id)
        )
        assert title == "Dense Passage Retrieval for Open-Domain Question Answering"
        assert h.embedder.inputs
        assert all("Dense Passage Retrieval" in item for item in h.embedder.inputs)

    async def test_the_fetch_runs_under_the_workers_own_lease(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        # The FETCHING heartbeat must renew the lease the worker claimed with; the
        # fetch deadline is half of it (D134), so a default lease would desynchronise
        # the two and let a slow fetch be claimed twice.
        h = await _harness(sessions, tmp_path, RecordingFetcher())
        remaining: list[float] = []

        async def read_lease() -> None:
            seconds = await h.scalar(
                "SELECT extract(epoch FROM visible_at - now()) FROM ingestion_jobs "
                "WHERE state = 'fetching'"
            )
            remaining.append(float(seconds))  # type: ignore[arg-type]

        h.fetcher.during = read_lease
        await h.add()
        await h.drain()

        assert len(remaining) == 1
        assert LEASE_SECONDS - 60 < remaining[0] <= LEASE_SECONDS

    async def test_re_adding_it_by_the_same_identifier_creates_no_job_or_chunks(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        # D136's criterion, FR-I7's external-ID half.
        h = await _harness(sessions, tmp_path, RecordingFetcher())
        first = await h.add()
        await h.drain()
        jobs = await h.scalar("SELECT count(*) FROM ingestion_jobs")
        chunks = await h.scalar("SELECT count(*) FROM chunks")

        again = await h.add()

        assert again.paper_id == first.paper_id
        assert (again.deduplicated, again.job_id) == (True, None)
        assert await h.scalar("SELECT count(*) FROM ingestion_jobs") == jobs
        assert await h.drain() == 0
        assert await h.scalar("SELECT count(*) FROM chunks") == chunks


class TestAFetchThatCannotSucceedIsReportedNotRetried:
    async def test_a_dead_link_ends_failed_with_a_reason_and_no_url(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        failed = FetchFailed(FailureClass.SOURCE_UNAVAILABLE, (FetchAttempt(LINK, "status:404"),))
        h = await _harness(sessions, tmp_path, RecordingFetcher(failed))
        added = await h.add()

        await h.drain()

        async with h.sessions() as session:
            state, failure, detail = (
                await session.execute(
                    text(
                        "SELECT state, failure_class, failure_detail FROM workspace_papers "
                        "WHERE paper_id = :p"
                    ),
                    {"p": uuid.UUID(added.paper_id)},
                )
            ).one()
        assert (state, failure) == ("failed", "source_unavailable")
        assert detail == "links tried: status:404"
        assert len(h.fetcher.calls) == 1, "a 404 was retried as though it might change"


class TestAPdfTheUserAlreadyUploadedIsNotIngestedTwice:
    async def test_the_workspace_ends_holding_the_upload_once(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        h = await _harness(sessions, tmp_path, RecordingFetcher())
        uploaded = await h.upload(PDF)
        await h.drain()
        placeholder = await h.add()

        with capture_logs() as logs:
            await h.drain()

        events = [entry["event"] for entry in logs]
        assert "ingestion.merged" in events
        # The job went with the placeholder; completing it again would only log a
        # miss that describes nothing wrong.
        assert "ingestion.completion_missed" not in events

        members = await h.scalar(
            "SELECT count(*) FROM workspace_papers WHERE workspace_id = :w",
            w=uuid.UUID(h.workspace_id),
        )
        assert members == 1
        assert (
            await h.scalar(
                "SELECT state FROM workspace_papers WHERE paper_id = :p",
                p=uuid.UUID(uploaded.paper_id),
            )
            == "indexed"
        )
        # The placeholder, and its job with it, are gone (D134).
        assert (
            await h.scalar(
                "SELECT count(*) FROM papers WHERE id = :p", p=uuid.UUID(placeholder.paper_id)
            )
            == 0
        )
        assert (
            await h.scalar(
                "SELECT count(*) FROM ingestion_jobs WHERE paper_id = :p",
                p=uuid.UUID(placeholder.paper_id),
            )
            == 0
        )


class TestAThrottledFetchWaitsAsLongAsTheHostSaid:
    async def test_the_hosts_stated_wait_reaches_the_job_row(
        self, sessions: async_sessionmaker[AsyncSession], tmp_path: Path
    ) -> None:
        # D142, end to end: fetcher -> FETCHING -> worker -> queue. Each half is
        # tested alone elsewhere; this is the test that fails if the worker stops
        # handing the wait to the queue.
        throttled = FetchFailed(
            FailureClass.TRANSIENT, (FetchAttempt(LINK, "status:429", retry_after=600.0),)
        )
        h = await _harness(sessions, tmp_path, RecordingFetcher(throttled))
        added = await h.add()

        await h.drain()

        async with h.sessions() as session:
            state, attempts, wait = (
                await session.execute(
                    text(
                        "SELECT state, attempts, extract(epoch FROM visible_at - now()) "
                        "FROM ingestion_jobs WHERE paper_id = :p"
                    ),
                    {"p": uuid.UUID(added.paper_id)},
                )
            ).one()
        assert (state, attempts) == ("queued", 1)
        assert 590 < float(wait) <= 600, "retried on D48's backoff instead of the host's word"
