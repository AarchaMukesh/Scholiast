"""One real paper in, one cited answer out — phase 6's acceptance test.

Real Postgres, real Qdrant, real GROBID, real Ollama. The papers are ingested
through the phase-5 pipeline rather than hand-inserted, so what is under test is
the whole spine: upload → parse → chunk → embed → index → retrieve → generate →
audit → persist.

**This is also where FR-A4 finally becomes verifiable.** The phase-4 commit
recorded it as debt: "two workspaces holding identical papers spans phases 2 and
4 with nothing joining them until retrieval exists." Retrieval now exists, so
`TestTwoTenantsWithTheSamePaper` builds exactly the situation §5.8 describes and
asserts the boundary holds through the query path.

The embedder is a deterministic stand-in, as in phase 5: what is being tested is
the pipeline, and a 419 MB download on a cold cache would violate NFR-T3.
Generation is real, because a fake model cannot demonstrate that a real one
cites the labels it was given.
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
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.use_cases.add_paper import AddPaperUseCase
from scholiast.application.use_cases.answer_question import (
    DEGRADED_GENERATION_UNAVAILABLE,
    AnswerComplete,
    AnswerQuestionUseCase,
    AnswerToken,
    EvidenceReady,
)
from scholiast.domain.errors import WorkspaceNotQueryable
from scholiast.domain.retrieval import CitationStatus
from scholiast.domain.workspace import PaperState
from scholiast.infrastructure.db.repositories import (
    SqlChunkRepository,
    SqlConversationRepository,
    SqlPaperRepository,
    SqlWorkspaceRepository,
)
from scholiast.infrastructure.llm.ollama import OllamaProvider
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.infrastructure.queue.postgres import PostgresQueue
from scholiast.infrastructure.storage.local import LocalObjectStore
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex
from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "spike" / "fixtures"
GROBID_URL = "http://localhost:8070"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DIMENSIONS = 32
QUESTION = "What problem does this paper address, and what do the authors propose?"


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
needs_ollama = pytest.mark.skipif(
    not _alive(f"{OLLAMA_URL}/api/tags"), reason="Ollama is not running"
)
needs_fixtures = pytest.mark.skipif(
    not FIXTURES.exists(), reason="fixture PDFs are gitignored; run spike/scripts/02 to fetch"
)


class DeterministicEmbedder:
    """Phase 5's stand-in, reused so retrieval is reproducible across runs.

    Queries and passages hash the same way, so a question sharing words with a
    chunk lands near it — crude, but enough for the pipeline assertions here.
    Retrieval *quality* is M2's subject and is measured there, not asserted here.
    """

    version = "test-embedder/1.0.0-fp32"
    dimensions = DIMENSIONS

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        vectors: list[list[float]] = []
        for item in texts:
            seed = abs(hash(item)) % 9973
            vectors.append([((seed + i * 37) % 100) / 100.0 for i in range(DIMENSIONS)])
        return vectors


class UnavailableProvider:
    """Every rung down — §5.11's last one."""

    name = "unavailable"
    model = "none"

    async def stream(self, *, system: str, user: str):  # type: ignore[no-untyped-def]
        from scholiast.application.ports.llm import GenerationUnavailable

        raise GenerationUnavailable("unavailable", "deliberately down")
        yield  # pragma: no cover - unreachable, keeps this an async generator


class FabricatingProvider:
    """Cites a label that was never in the evidence set.

    A stub rather than a prompt trick: layer 2 must hold whatever the model does,
    and waiting for a real model to hallucinate on cue is not a test.
    """

    name = "fabricator"
    model = "stub"

    async def stream(self, *, system: str, user: str):  # type: ignore[no-untyped-def]
        from scholiast.application.ports.llm import GenerationChunk

        for piece in ("The paper reports a result [C1]. ", "It also claims another [C99]."):
            yield GenerationChunk(text=piece)
        yield GenerationChunk(done=True)


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
        for table in (
            "messages",
            "conversations",
            "ingestion_jobs",
            "chunks",
            "sections",
            "workspace_papers",
            "papers",
            "workspaces",
        ):
            await connection.execute(text(f"DELETE FROM {table}"))  # noqa: S608 - fixed list
    yield factory
    await engine.dispose()


@pytest_asyncio.fixture
async def index() -> AsyncIterator[QdrantVectorIndex]:
    store = QdrantVectorIndex(
        QDRANT_URL, collection=f"qa_{uuid.uuid4().hex[:8]}", dimensions=DIMENSIONS
    )
    await store.ensure_ready()
    yield store
    await store._client.delete_collection(store.collection)
    await store.close()


class Tenant:
    """One user, one workspace, and the machinery to fill and query it."""

    def __init__(
        self, sessions: async_sessionmaker[AsyncSession], index: QdrantVectorIndex, objects: Path
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
        self.user_id = str(uuid.uuid4())
        self.workspace_id = str(uuid.uuid4())
        self.scope = TenantContext(user_id=self.user_id)
        self.conversation_id = ""

    async def seed(self) -> None:
        async with self.sessions() as session:
            await session.execute(
                text("INSERT INTO users (id, email) VALUES (:id, :email)"),
                {"id": self.user_id, "email": f"{self.user_id}@test.local"},
            )
            await session.execute(
                text(
                    "INSERT INTO workspaces (id, owner_id, name, state) "
                    "VALUES (:id, :owner, 'QA', 'empty')"
                ),
                {"id": self.workspace_id, "owner": self.user_id},
            )
            await session.commit()
        async with self.sessions() as session:
            created = await SqlConversationRepository(session, self.scope).create(
                self.workspace_id, title="QA"
            )
            await session.commit()
        assert created is not None
        self.conversation_id = created

    async def ingest(self, pdf: bytes, filename: str) -> str:
        async with self.sessions() as session:
            result = await AddPaperUseCase(
                workspaces=SqlWorkspaceRepository(session, self.scope),
                papers=SqlPaperRepository(session, self.scope),
                chunks=SqlChunkRepository(session, self.scope),
                store=self.store,
                queue=PostgresQueue(session, self.scope),
                owner_id=self.user_id,
            ).execute(workspace_id=self.workspace_id, data=pdf, filename=filename)
            await session.commit()
        for _ in range(5):
            if not await self.worker.run_once():
                break
        return result.paper_id

    async def ask(self, question: str = QUESTION, *, providers: Sequence[object] | None = None):  # type: ignore[no-untyped-def]
        chain = providers if providers is not None else [OllamaProvider(OLLAMA_URL)]
        events = []
        async with self.sessions() as session:
            use_case = AnswerQuestionUseCase(
                workspaces=SqlWorkspaceRepository(session, self.scope),
                papers=SqlPaperRepository(session, self.scope),
                chunks=SqlChunkRepository(session, self.scope),
                conversations=SqlConversationRepository(session, self.scope),
                embedder=self.embedder,
                index=self.index,
                providers=chain,  # type: ignore[arg-type]
            )
            stream = await use_case.stream(
                workspace_id=self.workspace_id,
                conversation_id=self.conversation_id,
                question=question,
            )
            async for event in stream:
                events.append(event)
            await session.commit()
        return events


@pytest_asyncio.fixture
async def tenant(
    sessions: async_sessionmaker[AsyncSession], index: QdrantVectorIndex, tmp_path: Path
) -> AsyncIterator[Tenant]:
    built = Tenant(sessions, index, tmp_path / "objects")
    await built.seed()
    yield built


def paper_bytes(name: str = "attention-is-all-you-need.pdf") -> bytes:
    return (FIXTURES / name).read_bytes()


def final(events: Sequence[object]) -> AnswerComplete:
    completes = [e for e in events if isinstance(e, AnswerComplete)]
    assert len(completes) == 1
    return completes[0]


@needs_grobid
@needs_qdrant
@needs_ollama
@needs_fixtures
class TestARealPaperYieldsACitedAnswer:
    async def test_a_question_is_answered_from_the_paper(self, tenant: Tenant) -> None:
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask()

        assert any(isinstance(e, AnswerToken) for e in events)
        answer = final(events)
        assert answer.answer.strip()
        assert answer.provider == "ollama"
        assert answer.degraded == ()

    async def test_every_emitted_citation_exists_in_the_retrieved_context(
        self, tenant: Tenant
    ) -> None:
        # The M1 acceptance criterion verbatim: "every emitted citation ID exists
        # in the retrieved context (validated, not trusted)".
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask()
        answer = final(events)
        labels = {i.label for i in next(e for e in events if isinstance(e, EvidenceReady)).items}

        for citation in answer.citations:
            assert citation.status is not CitationStatus.FABRICATED
            assert citation.label in labels
        assert answer.fabricated_labels == ()

    async def test_a_citation_resolves_to_the_exact_source_span(self, tenant: Tenant) -> None:
        # Phase 5 proved chunk → section → slice. This proves the same round trip
        # is reachable *from a citation*, which is what the UI will do.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask()
        items = {i.label: i for i in next(e for e in events if isinstance(e, EvidenceReady)).items}

        async with tenant.sessions() as session:
            for item in items.values():
                row = (
                    await session.execute(
                        text(
                            "SELECT s.text AS section_text, s.path, p.title, c.role "
                            "FROM chunks c JOIN sections s ON s.id = c.section_id "
                            "JOIN papers p ON p.id = c.paper_id WHERE c.id = :id"
                        ),
                        {"id": item.chunk_id},
                    )
                ).one()
                assert row.path == item.section_path
                assert row.title == item.paper_title
                assert item.page >= 1

                span = row.section_text[item.char_start : item.char_end]
                if row.role == "child":
                    # Phase 5 established the honest contract: `_join` promises a
                    # *covering* range, not verbatim equality — it normalises
                    # inter-sentence whitespace and a child may bridge two prose
                    # blocks with an excluded table between them. What is exact
                    # is the boundaries, so a one-character drift still fails.
                    assert span.split()[0] == item.text.split()[0]
                    assert span.split()[-1] == item.text.split()[-1]
                    assert len(span) >= len(item.text)
                else:
                    assert span == item.text

    async def test_the_answer_and_its_evidence_are_persisted(self, tenant: Tenant) -> None:
        # FR-Q6 and FR-Q9: the evidence panel and the latency breakdown both need
        # the turn to still exist after the stream closed.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask()

        async with tenant.sessions() as session:
            conversation = await SqlConversationRepository(session, tenant.scope).get(
                tenant.conversation_id
            )
        assert conversation is not None
        assert [t.role.value for t in conversation.turns] == ["user", "assistant"]

        answer_turn = conversation.turns[1]
        assert answer_turn.content == final(events).answer
        assert answer_turn.evidence
        assert answer_turn.prompt_version == final(events).prompt_version
        assert answer_turn.provider == "ollama"
        assert answer_turn.timings is not None
        assert answer_turn.timings.total_ms >= 0

    async def test_the_local_model_reports_where_it_actually_ran(self, tenant: Tenant) -> None:
        # D23: a silent CPU fallback at ~7x slower looks exactly like a working
        # GPU path. The adapter reports placement rather than assuming it — and
        # this asserts it answers at all, not which answer, because CI has no GPU.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        await tenant.ask()

        placement = await OllamaProvider(OLLAMA_URL).placement()
        # The model was just used, so it is loaded: "unloaded" would mean the
        # lookup matched nothing and "unknown" that /api/ps could not be read.
        # Either would be the adapter failing to report while looking like it
        # reported — the exact shape of the silent CPU fallback D23 is about.
        # Which of gpu/cpu/partial is not asserted: CI has no GPU.
        assert placement in {"gpu", "cpu", "partial"}


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestCitationIntegrityHoldsWhateverTheModelDoes:
    async def test_a_fabricated_label_is_stripped_and_reported(self, tenant: Tenant) -> None:
        # Layer 2 against a model that invents a label. Needs no Ollama: the
        # point is that the check holds regardless of which model produced the
        # text.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask(providers=[FabricatingProvider()])
        answer = final(events)

        assert "[C99]" not in answer.answer
        assert "[C1]" in answer.answer
        assert answer.fabricated_labels == ("C99",)
        assert any(c.status is CitationStatus.FABRICATED for c in answer.citations)

    async def test_the_fabrication_survives_into_the_stored_turn(self, tenant: Tenant) -> None:
        await tenant.ingest(paper_bytes(), "attention.pdf")
        await tenant.ask(providers=[FabricatingProvider()])

        async with tenant.sessions() as session:
            conversation = await SqlConversationRepository(session, tenant.scope).get(
                tenant.conversation_id
            )
        assert conversation is not None
        stored = conversation.turns[1]
        assert any(c.status is CitationStatus.FABRICATED for c in stored.citations)


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestDegradationReachesTheUser:
    async def test_no_provider_returns_evidence_without_synthesis(self, tenant: Tenant) -> None:
        # §5.11's last rung, end to end: retrieval still has standalone value and
        # says it was served without synthesis.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        events = await tenant.ask(providers=[UnavailableProvider(), UnavailableProvider()])
        answer = final(events)

        assert answer.degraded == (DEGRADED_GENERATION_UNAVAILABLE,)
        assert answer.provider is None
        assert next(e for e in events if isinstance(e, EvidenceReady)).items


@needs_qdrant
class TestAWorkspaceWithNothingIndexedIsNotAnswered:
    async def test_an_empty_workspace_raises_rather_than_improvising(self, tenant: Tenant) -> None:
        # FR-W5. This is the whole thesis of the project in one assertion: with
        # no corpus, the model's own knowledge of the literature is not an answer.
        with pytest.raises(WorkspaceNotQueryable):
            await tenant.ask()


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestTwoTenantsWithTheSamePaper:
    """FR-A4, verifiable at last (the P6 debt from the phase-4 commit).

    §5.8's stated verification is "two tenants with identical papers, every
    retrieval path asserted to return empty for the non-owner". Phases 2 and 4
    built the two halves; until retrieval existed there was nothing to join them.
    """

    @pytest_asyncio.fixture
    async def second(
        self,
        sessions: async_sessionmaker[AsyncSession],
        index: QdrantVectorIndex,
        tmp_path: Path,
    ) -> AsyncIterator[Tenant]:
        built = Tenant(sessions, index, tmp_path / "objects-b")
        await built.seed()
        yield built

    async def test_each_tenant_retrieves_only_its_own_copy(
        self, tenant: Tenant, second: Tenant
    ) -> None:
        # Byte-identical uploads by two tenants: §4.4.2's carve-out gives them
        # two paper rows on purpose, and they must stay apart through the query.
        pdf = paper_bytes()
        first_paper = await tenant.ingest(pdf, "attention.pdf")
        second_paper = await second.ingest(pdf, "attention.pdf")
        assert first_paper != second_paper, "the privacy carve-out collapsed to one row"

        a_events = await tenant.ask(providers=[FabricatingProvider()])
        b_events = await second.ask(providers=[FabricatingProvider()])

        a_papers = {
            i.paper_id for i in next(e for e in a_events if isinstance(e, EvidenceReady)).items
        }
        b_papers = {
            i.paper_id for i in next(e for e in b_events if isinstance(e, EvidenceReady)).items
        }
        assert a_papers == {first_paper}
        assert b_papers == {second_paper}
        assert a_papers.isdisjoint(b_papers)

    async def test_a_tenant_sees_none_of_the_others_conversation(
        self, tenant: Tenant, second: Tenant
    ) -> None:
        await tenant.ingest(paper_bytes(), "attention.pdf")
        await tenant.ask(providers=[FabricatingProvider()])

        async with tenant.sessions() as session:
            # `second`'s scope, `tenant`'s conversation id.
            leaked = await SqlConversationRepository(session, second.scope).get(
                tenant.conversation_id
            )
        assert leaked is None

    async def test_a_tenant_cannot_append_to_the_others_conversation(
        self, tenant: Tenant, second: Tenant
    ) -> None:
        from scholiast.domain.conversation import Turn, TurnRole

        async with tenant.sessions() as session:
            written = await SqlConversationRepository(session, second.scope).append(
                tenant.conversation_id, Turn(role=TurnRole.USER, content="injected")
            )
            await session.commit()
        assert written is None

    async def test_the_vector_filter_is_what_separates_them(
        self, tenant: Tenant, second: Tenant
    ) -> None:
        # The mechanism, not just the outcome: both tenants' vectors live in one
        # collection (AD-4), so the only thing keeping them apart is D36's
        # paper-id filter derived from the tenant-scoped query.
        pdf = paper_bytes()
        await tenant.ingest(pdf, "attention.pdf")
        await second.ingest(pdf, "attention.pdf")

        async with tenant.sessions() as session:
            a_ids = await SqlPaperRepository(session, tenant.scope).queryable_paper_ids(
                tenant.workspace_id
            )
            b_ids = await SqlPaperRepository(session, second.scope).queryable_paper_ids(
                second.workspace_id
            )
        assert a_ids and b_ids
        assert set(a_ids).isdisjoint(b_ids)
        # Both sets really are in the same collection — otherwise the filter is
        # not what is doing the work and this test proves nothing.
        assert await tenant.index.count(paper_ids=[*a_ids, *b_ids]) == await tenant.index.count()


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestAPaperStillIngestingIsDisclosedNotIgnored:
    async def test_an_in_flight_paper_is_reported_as_excluded(self, tenant: Tenant) -> None:
        # FR-W5: queries answer over what is ready and say what was left out.
        # Silently answering over a subset is a correctness bug, not a UX detail.
        await tenant.ingest(paper_bytes(), "attention.pdf")
        second_id = await tenant.ingest(paper_bytes("bert.pdf"), "bert.pdf")
        async with tenant.sessions() as session:
            await SqlPaperRepository(session, tenant.scope).set_membership_state(
                tenant.workspace_id, second_id, PaperState.EMBEDDING
            )
            await session.commit()

        events = await tenant.ask(providers=[FabricatingProvider()])
        panel = next(e for e in events if isinstance(e, EvidenceReady))
        assert panel.excluded_paper_ids == (second_id,)
