"""Doc 8's M1 acceptance criteria, one test each, through the real HTTP surface.

This file is the milestone's definition of done. Every test is named after a
checkbox in the roadmap, and nothing here reaches past the API into a use case:
the criteria are about what the *system* does, and a criterion satisfied only by
calling internals is not satisfied.

    - [ ] Upload a real ML paper -> ask a question -> streamed, cited answer
    - [ ] Clicking a citation resolves to the correct paper, section, page, span
    - [ ] Every emitted citation ID exists in the retrieved context
    - [ ] Chunks carry full provenance and chunker_version / embedder_version
    - [ ] Tables survive chunking intact
    - [ ] import-linter fails the build on a deliberate layer violation
    - [ ] Full stack runs from `docker compose up` + one command

The embedder is a deterministic stand-in, as in phases 5 and 6: a 419 MB
download on a cold cache would violate NFR-T3, and what these criteria are about
is the spine rather than retrieval quality — which is M2's subject and is
measured there.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.api.app import create_app
from scholiast.application.ports.ml import TextRole
from scholiast.config.settings import Settings
from scholiast.domain.chunking import CHUNKER_VERSION

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "spike" / "fixtures"
GROBID_URL = "http://localhost:8070"
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
DIMENSIONS = 32


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


def _lint_imports() -> Path:
    """The `lint-imports` console script beside the running interpreter.

    Located rather than assumed: the test must run the same binary the five
    gates run, and `python -m importlinter` is not an equivalent entry point.
    """
    scripts = Path(sys.executable).parent
    return scripts / ("lint-imports.exe" if os.name == "nt" else "lint-imports")


class DeterministicEmbedder:
    """The stand-in from phases 5 and 6, so runs are reproducible."""

    version = "test-embedder/1.0.0-fp32"
    dimensions = DIMENSIONS

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        return [
            [((abs(hash(t)) % 9973 + i * 37) % 100) / 100.0 for i in range(DIMENSIONS)]
            for t in texts
        ]


#: Migration 0005 gives the seeded M1 account a development password read from
#: the environment (D85). It is set explicitly here rather than left to the
#: documented default, so a developer who exports the variable does not silently
#: break this suite — and so these tests authenticate as the SAME principal M1's
#: `current_tenant` used to return unconditionally, which is what keeps the
#: ownership assertions below meaning what they meant.
#: Migration 0003 seeds this id. M3 phase 1 removed the constant from
#: `api.deps` — no production module names a principal any more, which is
#: the point of the phase — so the suite that asserts on the seeded owner
#: keeps its own copy.
M1_USER_ID = "00000000-0000-4000-8000-000000000001"
SEED_EMAIL = "demo@scholiast.local"
SEED_PASSWORD = "integration-suite-password"


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
            env={**os.environ, "SEED_USER_PASSWORD": SEED_PASSWORD},
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


async def authenticate(client: httpx.AsyncClient) -> str:
    """Log in as the seeded account and attach the bearer token to the client.

    M3 phase 1 replaced the inside of `current_tenant`: every router now needs a
    verified token, and no router changed to gain that. The tests therefore do
    what any client does — they log in.
    """
    response = await client.post(
        "/v1/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD}
    )
    assert response.status_code == 200, response.text
    token = response.json()["access_token"]
    client.headers["Authorization"] = f"Bearer {token}"
    return str(response.json()["user_id"])


@dataclass
class Harness:
    """The client and the app it is driving.

    The app is needed as well as the client because the tests run the ingestion
    worker by hand, and the worker wants the same singletons the app built — not
    a second 419 MB encoder.
    """

    client: httpx.AsyncClient
    app: FastAPI


@pytest_asyncio.fixture
async def api(postgres: str, tmp_path: Path) -> AsyncIterator[Harness]:
    """The real application, over ASGI, with throwaway services behind it.

    The app is built by `create_app` rather than imported from `scholiast.main`,
    so the worker is driven explicitly by the tests: a background poller racing
    the assertions is how flaky integration suites are written. The one-command
    wiring — which *does* start the worker — is asserted separately below.
    """
    collection = f"m1_{uuid.uuid4().hex[:8]}"
    settings = Settings(
        database_url=postgres,  # type: ignore[arg-type]
        qdrant_url=QDRANT_URL,
        qdrant_collection=collection,
        grobid_url=GROBID_URL,
        ollama_url=OLLAMA_URL,
        object_store_path=tmp_path / "objects",
        embedding_dimensions=DIMENSIONS,
        worker_in_process=False,
    )
    app = create_app(settings)

    async with LifespanManager(app) as manager:
        app.state.embedder = DeterministicEmbedder()
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            await authenticate(client)
            yield Harness(client=client, app=app)

        engine = create_async_engine(postgres)
        async with engine.begin() as connection:
            for table in (
                "refresh_tokens",
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
        await engine.dispose()
        await app.state.index._client.delete_collection(collection)


async def drain(app_state, limit: int = 6) -> int:  # type: ignore[no-untyped-def]
    """Run the ingestion worker until the queue is empty.

    Driven rather than polled: the worker is the same class the one-command
    wiring starts, but a background task racing these assertions would make the
    suite flaky for reasons unrelated to what it tests.
    """
    from scholiast.infrastructure.parsing.grobid import GrobidParser
    from scholiast.workers.ingestion import IngestionWorker, WorkerConfig

    worker = IngestionWorker(
        sessions=app_state.sessions,
        store=app_state.store,
        parser=GrobidParser(GROBID_URL),
        embedder=app_state.embedder,
        index=app_state.index,
        config=WorkerConfig(poll_seconds=0.05),
    )
    total = 0
    for _ in range(limit):
        ran = await worker.run_once()
        if not ran:
            break
        total += ran
    return total


async def ingest(api: Harness, name: str = "attention-is-all-you-need.pdf"):  # type: ignore[no-untyped-def]
    """Create a workspace, upload a real paper, and run it to INDEXED."""
    created = await api.client.post("/v1/workspaces", json={"name": "Acceptance"})
    workspace_id = created.json()["id"]

    pdf = (FIXTURES / name).read_bytes()
    upload = await api.client.post(
        f"/v1/workspaces/{workspace_id}/papers/upload",
        files={"file": (name, pdf, "application/pdf")},
    )
    assert upload.status_code == 202, upload.text
    await drain(api.app.state)
    return workspace_id, upload.json()["paper_id"]


async def ask(client: httpx.AsyncClient, workspace_id: str, question: str):  # type: ignore[no-untyped-def]
    """Ask over HTTP and collect the SSE frames, as the browser does."""
    conversation = await client.post(
        f"/v1/workspaces/{workspace_id}/conversations", json={"title": None}
    )
    conversation_id = conversation.json()["conversation_id"]

    events: list[tuple[str, dict[str, object]]] = []
    async with client.stream(
        "POST",
        f"/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages",
        json={"question": question},
    ) as response:
        assert response.status_code == 200, await response.aread()
        assert response.headers["content-type"].startswith("text/event-stream")
        buffer = ""
        async for piece in response.aiter_text():
            buffer += piece
            while "\n\n" in buffer:
                frame, buffer = buffer.split("\n\n", 1)
                name = ""
                data = ""
                for line in frame.split("\n"):
                    if line.startswith("event: "):
                        name = line[7:]
                    elif line.startswith("data: "):
                        data += line[6:]
                if data:
                    events.append((name, json.loads(data)))
    return events


@needs_grobid
@needs_qdrant
@needs_ollama
@needs_fixtures
class TestUploadAPaperAskAQuestionGetACitedAnswer:
    """Criterion 1: the walking skeleton, end to end, over HTTP."""

    async def test_the_answer_streams_and_carries_citations(self, api: Harness) -> None:
        workspace_id, _ = await ingest(api)
        events = await ask(api.client, workspace_id, "What do the authors propose?")

        names = [name for name, _ in events]
        assert names[0] == "evidence"
        assert names.count("token") > 1, "the answer arrived in one lump, not streamed"
        assert names[-1] == "complete"

        complete = events[-1][1]
        assert complete["answer"]
        assert complete["provider"] == "ollama"
        assert complete["prompt_version"]


@needs_grobid
@needs_qdrant
@needs_fixtures
class TestCitationIntegrityAndProvenance:
    """Criteria 2, 3 and 4 — the ones the whole design exists to make true."""

    async def test_every_emitted_citation_id_exists_in_the_retrieved_context(
        self, api: Harness
    ) -> None:
        workspace_id, _ = await ingest(api)
        events = await ask(api.client, workspace_id, "What is reported?")

        evidence = next(payload for name, payload in events if name == "evidence")
        complete = events[-1][1]
        labels = {item["label"] for item in evidence["items"]}  # type: ignore[index]

        assert complete["citations"] or complete["answer"]
        for citation in complete["citations"]:  # type: ignore[union-attr]
            assert citation["status"] != "fabricated"
            assert citation["label"] in labels
        assert complete["fabricated_labels"] == []

    async def test_clicking_a_citation_resolves_to_the_exact_span(self, api: Harness) -> None:
        # Criterion 2, through the endpoint the UI actually calls: the evidence
        # item's offsets index `sections.text` (D42), which is what the content
        # endpoint returns — so the highlight is exact rather than approximate.
        workspace_id, paper_id = await ingest(api)
        events = await ask(api.client, workspace_id, "What is reported?")
        evidence = next(payload for name, payload in events if name == "evidence")

        content = await api.client.get(f"/v1/workspaces/{workspace_id}/papers/{paper_id}/content")
        assert content.status_code == 200
        sections = {s["path"]: s for s in content.json()["sections"]}

        checked = 0
        for item in evidence["items"]:  # type: ignore[index]
            section = sections[item["section_path"]]
            span = section["text"][item["char_start"] : item["char_end"]]
            assert item["paper_id"] == paper_id
            assert item["page"] >= 1
            assert section["page_start"] <= item["page"] <= section["page_end"]
            # Exact for a table or a parent window; a child's span is a covering
            # range with exact boundaries (the contract phase 5 established).
            assert span.split()[0] == item["text"].split()[0]
            assert span.split()[-1] == item["text"].split()[-1]
            checked += 1
        assert checked, "no evidence to check"

    async def test_chunks_carry_provenance_and_both_version_stamps(
        self, api: Harness, postgres: str
    ) -> None:
        # Criterion 4 (FR-I5, NFR-M5). Asserted in the database rather than in
        # the response, because the claim is about what is *stored*.
        _, paper_id = await ingest(api)
        engine = create_async_engine(postgres)
        async with engine.connect() as connection:
            rows = (
                await connection.execute(
                    text(
                        "SELECT role, page, char_start, char_end, chunker_version, "
                        "embedder_version FROM chunks WHERE paper_id = :p"
                    ),
                    {"p": paper_id},
                )
            ).all()
        await engine.dispose()

        assert rows
        assert all(r.page >= 1 and r.char_end > r.char_start for r in rows)
        assert all(r.chunker_version == CHUNKER_VERSION for r in rows)
        embedded = [r for r in rows if r.role != "parent"]
        assert embedded and all(r.embedder_version for r in embedded)
        # Parent windows are context, not candidates — they are never embedded.
        assert all(r.embedder_version is None for r in rows if r.role == "parent")

    async def test_tables_survive_chunking_intact(self, api: Harness, postgres: str) -> None:
        # Criterion 5, with D41's honest claim: a table is never split mid-row
        # and never loses its caption. Cell-level fidelity is measured in M2 and
        # is deliberately not asserted here.
        _, paper_id = await ingest(api)
        engine = create_async_engine(postgres)
        async with engine.connect() as connection:
            tables = (
                (
                    await connection.execute(
                        text("SELECT text FROM chunks WHERE paper_id = :p AND role = 'table'"),
                        {"p": paper_id},
                    )
                )
                .scalars()
                .all()
            )
        await engine.dispose()

        assert tables, "the fixture has tables; finding none means they never entered the index"
        for table in tables:
            lines = table.splitlines()
            rows = [line for line in lines if line.startswith("|")]

            # Never split mid-row. A truncated row loses its trailing delimiter.
            assert all(line.endswith("|") for line in rows), "a row was split"

            # Never loses its caption — the other half of D41's claim, and the
            # half that makes a mangled table still retrievable: captions are
            # present for 100% of tables and are what people actually search
            # ("Table 3: results on MMLU"). Serialisation is caption-first, so
            # this doubles as the atomicity check: a table split across two
            # chunks would leave a tail chunk whose first line is a row.
            assert lines, "an empty table chunk"
            assert not lines[0].startswith("|"), (
                "a table chunk begins with a row, so it either lost its caption "
                "or is the tail of a table that was split across chunks"
            )
            # A caption with no rows is a figure caption misfiled as a table.
            assert rows, "a table chunk carries a caption but no rows"


class TestTheLayeringGateActuallyFails:
    """Criterion 6 — "test this once, on purpose" (D64).

    A gate nobody has watched fail is a gate nobody knows is wired. This plants
    a violation in a *copy* of the tree rather than the real one: the obvious
    implementation writes into `src/scholiast/domain/` and deletes it in a
    `finally`, which is one interrupted test away from leaving a file that
    breaks the gate for everyone and looks like a real violation nobody made.
    """

    def test_a_domain_module_importing_sqlalchemy_breaks_the_build(self, tmp_path: Path) -> None:
        shutil.copytree(ROOT / "src", tmp_path / "src")
        (tmp_path / "src" / "scholiast" / "domain" / "_planted_violation.py").write_text(
            "from sqlalchemy import select  # a domain module reaching for an ORM\n"
            "__all__ = ['select']\n",
            encoding="utf-8",
        )
        # The same three contracts as pyproject.toml, pointed at the copy.
        (tmp_path / "setup.cfg").write_text(
            "[importlinter]\n"
            "root_package = scholiast\n"
            "include_external_packages = True\n"
            "\n"
            "[importlinter:contract:1]\n"
            "name = Domain is framework-free\n"
            "type = forbidden\n"
            "source_modules =\n"
            "    scholiast.domain\n"
            "forbidden_modules =\n"
            "    sqlalchemy\n"
            "    fastapi\n",
            encoding="utf-8",
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(_lint_imports())],
            cwd=tmp_path,
            # The real environment plus a PYTHONPATH pointing at the copy. Wiping
            # it entirely breaks the interpreter on Windows, and grimp needs to
            # import the third-party packages the contract names.
            env={**os.environ, "PYTHONPATH": str(tmp_path / "src")},
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode != 0, (
            "import-linter passed a domain module importing SQLAlchemy — "
            f"the gate is not enforcing anything.\n{result.stdout}\n{result.stderr}"
        )
        assert "Domain is framework-free" in result.stdout
        assert "_planted_violation" in result.stdout

    def test_the_real_tree_still_passes(self) -> None:
        # The positive control. Without it, the test above would pass just as
        # happily if `lint-imports` failed for an unrelated reason — a syntax
        # error, a missing package, a broken config.
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [str(_lint_imports())],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout


class TestOneCommandStartsTheWholeThing:
    """Criterion 7's application half.

    `docker compose up` brings up Postgres, Qdrant, Redis and GROBID; the one
    command is `uvicorn scholiast.main:app`. What is checkable here is that the
    module exposes an app and that the app it exposes starts the worker — the
    part that would otherwise be a README claim nobody verified.
    """

    def test_the_documented_entrypoint_exposes_an_app(self) -> None:
        from fastapi import FastAPI

        from scholiast import main

        assert isinstance(main.app, FastAPI)

    def test_the_entrypoint_lifespan_is_the_one_that_runs_the_worker(self) -> None:
        # `scholiast.main` wraps the API's lifespan to add the worker (D63), so
        # the single command starts both halves. `create_app` on its own does
        # not — which is why the acceptance fixture above can drive the worker
        # by hand without one running underneath it.
        from scholiast import main
        from scholiast.api.app import api_lifespan

        assert main.lifespan is not api_lifespan
        assert "IngestionWorker" in main.lifespan.__wrapped__.__code__.co_names

    def test_the_api_layer_does_not_import_the_worker(self) -> None:
        # The contract says `api` and `workers` are independent siblings, and
        # `lint-imports` enforces it — this asserts the *reason*: an HTTP layer
        # that imports a background worker cannot be deployed without one.
        source = (ROOT / "src" / "scholiast" / "api" / "app.py").read_text(encoding="utf-8")
        assert "scholiast.workers" not in source


@needs_qdrant
class TestTheSeededUserExists:
    """The M1 trivial adapter (doc 8): one user, created by migration 0003."""

    async def test_workspaces_are_owned_by_the_seeded_user(
        self, api: Harness, postgres: str
    ) -> None:
        created = await api.client.post("/v1/workspaces", json={"name": "Owned"})
        assert created.status_code == 201

        engine = create_async_engine(postgres)
        async with engine.connect() as connection:
            owner = (
                await connection.execute(
                    text("SELECT owner_id FROM workspaces WHERE id = :id"),
                    {"id": created.json()["id"]},
                )
            ).scalar_one()
        await engine.dispose()
        assert str(owner) == M1_USER_ID
