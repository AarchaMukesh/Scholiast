"""A paper before its PDF, against a real Postgres (D132, D133, D134).

The repository and queue halves of phase 6a, where an in-memory fake would agree
with whatever the code assumes: identity found per owner and in the right order, a
hash written once, a row lock that really serialises two fetches of one paper, the
links a job carries surviving the claim, and a downgrade that refuses rather than
cascading users' queued papers away.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.tenancy import WORKER_CLAIM_REASON, SystemContext, TenantContext
from scholiast.domain.paper import Paper, PaperOrigin, PaperVisibility
from scholiast.domain.workspace import PaperState
from scholiast.infrastructure.db.repositories import SqlPaperRepository
from scholiast.infrastructure.queue.postgres import PostgresQueue

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
UNUSABLE_HASH = "not-a-real-argon2-hash"
HEAD = "7c1d4e9a2f60"
BEFORE = "e83c5f1a2b64"


def _alembic(url: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-m", "alembic", "-x", f"db_url={url}", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _migrated() -> Iterator[str]:
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = _alembic(url, "upgrade", "head")
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    yield from _migrated()


@pytest_asyncio.fixture
async def sessions(postgres: str) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(postgres)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _tenant(session: AsyncSession) -> tuple[TenantContext, str]:
    """A user with one workspace, committed."""
    user, workspace = uuid.uuid4(), uuid.uuid4()
    await session.execute(
        text("INSERT INTO users (id, email, password_hash) VALUES (:id, :email, :hash)"),
        {"id": user, "email": f"{user}@example.com", "hash": UNUSABLE_HASH},
    )
    await session.execute(
        text(
            "INSERT INTO workspaces (id, owner_id, name, state) VALUES (:id, :owner, 'W', 'active')"
        ),
        {"id": workspace, "owner": user},
    )
    await session.commit()
    return TenantContext(user_id=str(user)), str(workspace)


def _discovered(scope: TenantContext, **overrides: object) -> Paper:
    values: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "content_hash": None,
        "visibility": PaperVisibility.PRIVATE,
        "owner_id": scope.user_id,
        "title": "Dense Passage Retrieval for Open-Domain Question Answering",
        "arxiv_id": "2004.04906",
        "origin": PaperOrigin.DISCOVERY,
    }
    values.update(overrides)
    return Paper(**values)  # type: ignore[arg-type]


async def _held(session: AsyncSession, scope: TenantContext, workspace: str, paper: Paper) -> Paper:
    repo = SqlPaperRepository(session, scope)
    stored = await repo.add(paper)
    await repo.add_to_workspace(workspace, stored.id, PaperState.QUEUED)
    await session.commit()
    return stored


class TestADiscoveredPapersIdentityIsKept:
    async def test_a_paper_awaiting_its_pdf_round_trips_with_its_identity(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            paper = await _held(
                session, scope, workspace, _discovered(scope, doi="10.18653/v1/2020.emnlp-main.550")
            )

        async with sessions() as session:
            read = await SqlPaperRepository(session, scope).get(paper.id)

        assert read is not None
        assert read.content_hash is None
        assert (read.doi, read.arxiv_id) == ("10.18653/v1/2020.emnlp-main.550", "2004.04906")
        assert read.origin is PaperOrigin.DISCOVERY

    async def test_the_arxiv_id_is_tried_before_the_doi(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            by_arxiv = await _held(session, scope, workspace, _discovered(scope))
            by_doi = await _held(
                session, scope, workspace, _discovered(scope, arxiv_id=None, doi="10.1/x")
            )
            repo = SqlPaperRepository(session, scope)

            both = await repo.find_by_identity(arxiv_id="2004.04906", doi="10.1/x")
            only_doi = await repo.find_by_identity(arxiv_id="2110.01599", doi="10.1/x")

        assert both is not None and both.id == by_arxiv.id
        assert only_doi is not None and only_doi.id == by_doi.id

    async def test_another_owners_row_for_the_same_work_is_never_found(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # D131: the same work is two rows, and neither owner can reach the other's.
        async with sessions() as session:
            alice, alice_ws = await _tenant(session)
            bob, _ = await _tenant(session)
            mine = await _held(session, alice, alice_ws, _discovered(alice))

            seen_by_bob = await SqlPaperRepository(session, bob).find_by_identity(
                arxiv_id="2004.04906", doi=None
            )
            seen_by_alice = await SqlPaperRepository(session, alice).find_by_identity(
                arxiv_id="2004.04906", doi=None
            )

        assert seen_by_bob is None
        assert seen_by_alice is not None and seen_by_alice.id == mine.id

    async def test_a_system_scope_has_no_owner_to_find_by(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            system = SystemContext(reason=WORKER_CLAIM_REASON)
            with pytest.raises(PermissionError):
                await SqlPaperRepository(session, system).find_by_identity(
                    arxiv_id="2004.04906", doi=None
                )

    async def test_a_hash_is_written_once_and_only_by_its_owner(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            alice, workspace = await _tenant(session)
            bob, _ = await _tenant(session)
            paper = await _held(session, alice, workspace, _discovered(alice))

            assert not await SqlPaperRepository(session, bob).set_content_hash(paper.id, "b" * 64)
            assert await SqlPaperRepository(session, alice).set_content_hash(paper.id, "a" * 64)
            assert not await SqlPaperRepository(session, alice).set_content_hash(paper.id, "c" * 64)
            await session.commit()
            read = await SqlPaperRepository(session, alice).get(paper.id)

        assert read is not None and read.content_hash == "a" * 64

    async def test_a_parse_write_back_never_moves_the_identity(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # D132: identity sits under a unique index, where `update` cannot move it.
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            paper = await _held(
                session, scope, workspace, _discovered(scope, content_hash="d" * 64)
            )
            paper.arxiv_id = "2110.01599"
            paper.doi = "10.1/elsewhere"
            await SqlPaperRepository(session, scope).update(paper)
            await session.commit()
            read = await SqlPaperRepository(session, scope).get(paper.id)

        assert read is not None
        assert (read.arxiv_id, read.doi) == ("2004.04906", None)


class TestTheRowLockSerialisesTwoFetchesOfOnePaper:
    """D134: the second job to finish a fetch must find the first one's hash."""

    async def test_a_second_locker_waits_while_a_plain_read_does_not(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as setup:
            scope, workspace = await _tenant(setup)
            paper = await _held(setup, scope, workspace, _discovered(scope))

        async with sessions() as first, sessions() as second:
            assert await SqlPaperRepository(first, scope).get_for_update(paper.id) is not None

            # A reader is never blocked, so the API keeps listing the paper...
            assert await SqlPaperRepository(second, scope).get(paper.id) is not None
            # ...but a second locker waits; bounded, so a lock that is not taken
            # fails this test instead of hanging it.
            await second.execute(text("SET lock_timeout = '300ms'"))
            with pytest.raises(DBAPIError, match="lock"):
                await SqlPaperRepository(second, scope).get_for_update(paper.id)
            await second.rollback()

            await first.commit()
            await second.execute(text("SET lock_timeout = '300ms'"))
            assert await SqlPaperRepository(second, scope).get_for_update(paper.id) is not None
            await second.rollback()


class TestAJobCarriesTheLinksItWillFetch:
    async def test_the_links_an_add_queued_come_back_with_the_claim(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        links = ["https://arxiv.org/pdf/2004.04906v3", "https://www.aclweb.org/a.pdf"]
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            paper = await _held(session, scope, workspace, _discovered(scope))
            job_id = await PostgresQueue(session, scope).enqueue(
                workspace_id=workspace, paper_id=paper.id, source_links=links
            )
            await session.commit()

        claimed = await self._claim(sessions, job_id)
        assert claimed.source_links == tuple(links)

    async def test_an_upload_job_carries_none(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            paper = await _held(
                session, scope, workspace, _discovered(scope, content_hash="e" * 64)
            )
            job_id = await PostgresQueue(session, scope).enqueue(
                workspace_id=workspace, paper_id=paper.id
            )
            await session.commit()
            stored = (
                await session.execute(
                    text("SELECT source_links FROM ingestion_jobs WHERE id = :id"),
                    {"id": uuid.UUID(job_id)},
                )
            ).scalar_one()

        assert stored is None
        assert (await self._claim(sessions, job_id)).source_links == ()

    @staticmethod
    async def _claim(sessions: async_sessionmaker[AsyncSession], job_id: str):  # type: ignore[no-untyped-def]
        system = SystemContext(reason=WORKER_CLAIM_REASON)
        # Claim everything due: other tests' jobs share this database.
        async with sessions() as session:
            jobs = await PostgresQueue(session, system).claim(limit=100)
            await session.commit()
        (claimed,) = [job for job in jobs if job.id == job_id]
        return claimed


@pytest.fixture
def fresh_postgres() -> Iterator[str]:
    """Its own database: this test moves the schema, and nothing else may see that."""
    yield from _migrated()


class TestTheDowngradeRefusesWhileAPaperAwaitsItsPdf:
    async def test_it_refuses_with_one_waiting_and_runs_once_there_is_none(
        self, fresh_postgres: str
    ) -> None:
        engine = create_async_engine(fresh_postgres)
        sessions = async_sessionmaker(engine, expire_on_commit=False)
        async with sessions() as session:
            scope, workspace = await _tenant(session)
            paper = await _held(session, scope, workspace, _discovered(scope))

        refused = _alembic(fresh_postgres, "downgrade", BEFORE)
        assert refused.returncode != 0
        assert "await their PDF" in refused.stderr
        assert await self._version(engine) == HEAD, "a refused downgrade left the schema moved"

        async with sessions() as session:
            await session.execute(
                text("DELETE FROM papers WHERE id = :id"), {"id": uuid.UUID(paper.id)}
            )
            await session.commit()
        ran = _alembic(fresh_postgres, "downgrade", BEFORE)
        assert ran.returncode == 0, ran.stderr
        assert await self._version(engine) == BEFORE
        async with engine.connect() as conn:
            nullable = (
                await conn.execute(
                    text(
                        "SELECT is_nullable FROM information_schema.columns "
                        "WHERE table_name = 'papers' AND column_name = 'content_hash'"
                    )
                )
            ).scalar_one()
        assert nullable == "NO"

        assert _alembic(fresh_postgres, "upgrade", "head").returncode == 0
        await engine.dispose()

    @staticmethod
    async def _version(engine) -> str:  # type: ignore[no-untyped-def]
        async with engine.connect() as conn:
            return str(
                (await conn.execute(text("SELECT version_num FROM alembic_version"))).scalar_one()
            )
