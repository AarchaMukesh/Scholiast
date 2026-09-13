"""The Postgres queue, against a real Postgres (D47, D48, §5.7.4).

`SKIP LOCKED` and lease expiry cannot be tested against a fake: both are
properties of the database's locking and clock, and an in-memory stand-in would
assert that the test double behaves, which nobody doubted. Every test here names
the failure it prevents, because a queue that looks like it works is the normal
state of a broken queue.

Migrations create the schema, never `metadata.create_all()` — the migration is
what runs against a real database, so it is what must be exercised.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.application.ports.tenancy import (
    WORKER_CLAIM_REASON,
    SystemContext,
    TenantContext,
    TenantIsolationViolation,
)
from scholiast.domain.errors import FailureClass
from scholiast.domain.workspace import PaperState
from scholiast.infrastructure.queue.postgres import PostgresQueue, retry_delay

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = SystemContext(reason=WORKER_CLAIM_REASON)


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
        # Each test starts from an empty queue: a claim test that passes because
        # it picked up a previous test's job proves nothing.
        await connection.execute(text("DELETE FROM ingestion_jobs"))
        await connection.execute(text("DELETE FROM workspace_papers"))
        await connection.execute(text("DELETE FROM papers"))
        await connection.execute(text("DELETE FROM workspaces"))
    yield factory
    await engine.dispose()


async def seed_paper(factory: async_sessionmaker[AsyncSession]) -> tuple[str, str, str]:
    """A user, a workspace and a paper — the rows a job must point at."""
    user_id, workspace_id, paper_id = (str(uuid.uuid4()) for _ in range(3))
    async with factory() as session:
        await session.execute(
            text("INSERT INTO users (id, email) VALUES (:id, :email)"),
            {"id": user_id, "email": f"{user_id}@test.local"},
        )
        await session.execute(
            text(
                "INSERT INTO workspaces (id, owner_id, name, state) "
                "VALUES (:id, :owner, 'Queue test', 'populating')"
            ),
            {"id": workspace_id, "owner": user_id},
        )
        await session.execute(
            text(
                "INSERT INTO papers (id, visibility, owner_id, content_hash, title) "
                "VALUES (:id, 'private', :owner, :hash, 'Paper')"
            ),
            {"id": paper_id, "owner": user_id, "hash": uuid.uuid4().hex * 2},
        )
        await session.commit()
    return user_id, workspace_id, paper_id


async def enqueue(factory: async_sessionmaker[AsyncSession]) -> tuple[str, str, str]:
    user_id, workspace_id, paper_id = await seed_paper(factory)
    async with factory() as session:
        job_id = await PostgresQueue(session, SYSTEM).enqueue(
            workspace_id=workspace_id, paper_id=paper_id
        )
        await session.commit()
    return job_id, workspace_id, user_id


async def expire_lease(factory: async_sessionmaker[AsyncSession], job_id: str) -> None:
    """Push a job's visibility back into the past — a worker that died."""
    async with factory() as session:
        await session.execute(
            text("UPDATE ingestion_jobs SET visible_at = now() - interval '1 hour' WHERE id = :id"),
            {"id": job_id},
        )
        await session.commit()


async def job_row(factory: async_sessionmaker[AsyncSession], job_id: str):  # type: ignore[no-untyped-def]
    async with factory() as session:
        return (
            await session.execute(
                text(
                    "SELECT state, attempts, last_error, visible_at > now() AS leased "
                    "FROM ingestion_jobs WHERE id = :id"
                ),
                {"id": job_id},
            )
        ).one()


class TestAClaimIsALeaseNotARemoval:
    async def test_an_enqueued_job_is_claimable(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id, workspace_id, _ = await enqueue(sessions)
        async with sessions() as session:
            (job,) = await PostgresQueue(session, SYSTEM).claim()
            await session.commit()
        assert job.id == job_id
        assert job.workspace_id == workspace_id
        assert job.state is PaperState.QUEUED

    async def test_a_claimed_job_is_invisible_until_its_lease_expires(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # Without the lease predicate the same job is handed out on every poll,
        # so one paper is ingested over and over while the rest of the queue
        # starves.
        await enqueue(sessions)
        async with sessions() as session:
            assert len(await PostgresQueue(session, SYSTEM).claim()) == 1
            await session.commit()
        async with sessions() as session:
            assert await PostgresQueue(session, SYSTEM).claim() == []

    async def test_a_dead_workers_job_comes_back(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # NFR-R2: no accepted ingestion job is lost on worker restart. The job is
        # left mid-stage, exactly as an OOM-killed worker would leave it.
        job_id, _, _ = await enqueue(sessions)
        async with sessions() as session:
            (job,) = await PostgresQueue(session, SYSTEM).claim()
            await PostgresQueue(session, SYSTEM).heartbeat(job.id, state=PaperState.CHUNKING)
            await session.commit()
        await expire_lease(sessions, job_id)

        async with sessions() as session:
            (redelivered,) = await PostgresQueue(session, SYSTEM).claim()
        assert redelivered.id == job_id
        # The stage survives the redelivery, which is what tells an operator
        # where the paper was when its worker died.
        assert redelivered.state is PaperState.CHUNKING

    async def test_a_heartbeat_extends_the_lease(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # A job that outlives its lease while healthy would be claimed by a
        # second worker and ingested twice concurrently.
        job_id, _, _ = await enqueue(sessions)
        async with sessions() as session:
            await PostgresQueue(session, SYSTEM).claim(lease_seconds=1)
            await session.commit()
        await expire_lease(sessions, job_id)
        async with sessions() as session:
            await PostgresQueue(session, SYSTEM).heartbeat(
                job_id, state=PaperState.PARSING, lease_seconds=600
            )
            await session.commit()

        assert job_row_leased(await job_row(sessions, job_id))
        async with sessions() as session:
            assert await PostgresQueue(session, SYSTEM).claim() == []


def job_row_leased(row) -> bool:  # type: ignore[no-untyped-def]
    return bool(row.leased)


class TestConcurrentWorkersNeverShareAJob:
    """Two properties, and mutation testing showed they are carried separately.

    *No double delivery* comes from the lease predicate plus row locking:
    removing `SKIP LOCKED` leaves it intact, because Postgres re-evaluates the
    WHERE clause after the lock is granted and the first claimer has by then
    pushed `visible_at` into the future. That is worth knowing rather than
    assuming — the obvious story ("SKIP LOCKED is what stops double delivery")
    is not what the code depends on.

    *Not waiting* is what `SKIP LOCKED` actually buys, and it is a real
    requirement: without it a second worker blocks on the first worker's row for
    as long as that worker's transaction is open, which with per-stage commits
    is the length of a pipeline stage.
    """

    async def test_two_simultaneous_claims_get_different_jobs(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # Two workers ingesting one paper would race on the same section rows,
        # where one worker's `replace_for_paper` deletes the other's chunks
        # mid-write.
        first_id, _, _ = await enqueue(sessions)
        second_id, _, _ = await enqueue(sessions)

        async def claim_one() -> list[str]:
            async with sessions() as session:
                jobs = await PostgresQueue(session, SYSTEM).claim()
                # Hold the transaction open so the two genuinely overlap.
                await asyncio.sleep(0.2)
                await session.commit()
                return [j.id for j in jobs]

        left, right = await asyncio.gather(claim_one(), claim_one())
        assert sorted(left + right) == sorted([first_id, second_id])
        assert set(left).isdisjoint(right)

    async def test_a_claimer_does_not_wait_on_another_workers_row(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # The property SKIP LOCKED is actually for. The holder keeps its
        # transaction open — exactly what a worker does while parsing — and a
        # second claimer must step over that row rather than queue behind it.
        # Without SKIP LOCKED this call blocks until the holder commits and the
        # timeout below fires.
        first, second = await enqueue(sessions), await enqueue(sessions)
        holder = sessions()
        try:
            held = await PostgresQueue(holder, SYSTEM).claim()
            assert len(held) == 1, "the holder should have taken exactly one job"

            async with sessions() as other:
                claimed = await asyncio.wait_for(PostgresQueue(other, SYSTEM).claim(), timeout=10.0)
            assert claimed, "the second claimer came back empty instead of stepping over"
            assert claimed[0].id != held[0].id
            assert {held[0].id, claimed[0].id} == {first[0], second[0]}
        finally:
            await holder.rollback()
            await holder.close()

    async def test_a_lone_job_goes_to_exactly_one_claimer(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id, _, _ = await enqueue(sessions)

        async def claim_one() -> list[str]:
            async with sessions() as session:
                jobs = await PostgresQueue(session, SYSTEM).claim()
                await asyncio.sleep(0.2)
                await session.commit()
                return [j.id for j in jobs]

        results = await asyncio.gather(*(claim_one() for _ in range(3)))
        assert sorted(job for batch in results for job in batch) == [job_id]


class TestDeliveriesAreCounted:
    async def test_claiming_increments_attempts(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # Counting only *reported* failures would redeliver a poison job forever:
        # a worker killed mid-job reports nothing at all.
        job_id, _, _ = await enqueue(sessions)
        for expected in (1, 2, 3):
            async with sessions() as session:
                (job,) = await PostgresQueue(session, SYSTEM).claim()
                await session.commit()
            assert job.attempts == expected
            await expire_lease(sessions, job_id)

    async def test_a_crash_looping_job_eventually_dead_letters(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id, _, _ = await enqueue(sessions)
        landed = PaperState.QUEUED
        for _ in range(5):
            await expire_lease(sessions, job_id)
            async with sessions() as session:
                queue = PostgresQueue(session, SYSTEM)
                claimed = await queue.claim()
                if not claimed:
                    break
                landed = await queue.fail(
                    claimed[0].id, failure_class=FailureClass.TRANSIENT, detail="worker died"
                )
                await session.commit()
        assert landed is PaperState.FAILED
        await expire_lease(sessions, job_id)
        async with sessions() as session:
            assert await PostgresQueue(session, SYSTEM).claim() == []


class TestRetryPolicy:
    async def test_a_transient_failure_is_requeued_with_a_delay(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id, _, _ = await enqueue(sessions)
        async with sessions() as session:
            queue = PostgresQueue(session, SYSTEM)
            (job,) = await queue.claim()
            landed = await queue.fail(
                job.id, failure_class=FailureClass.TRANSIENT, detail="GROBID 503"
            )
            await session.commit()

        assert landed is PaperState.QUEUED
        row = await job_row(sessions, job_id)
        assert row.state == PaperState.QUEUED.value
        # Backed off, not immediately re-runnable: without the delay a failing
        # dependency is hammered at poll speed.
        assert row.leased is True
        assert "GROBID 503" in row.last_error

    async def test_an_unparseable_pdf_is_unsupported_immediately(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # Retrying cannot supply a password, and §4.4.1 says UNSUPPORTED exists
        # to avoid giving "false hope of retry".
        job_id, _, _ = await enqueue(sessions)
        async with sessions() as session:
            queue = PostgresQueue(session, SYSTEM)
            (job,) = await queue.claim()
            landed = await queue.fail(
                job.id, failure_class=FailureClass.UNPARSEABLE, detail="encrypted"
            )
            await session.commit()

        assert landed is PaperState.UNSUPPORTED
        await expire_lease(sessions, job_id)
        async with sessions() as session:
            assert await PostgresQueue(session, SYSTEM).claim() == []

    async def test_a_missing_source_is_failed_not_unsupported(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        await enqueue(sessions)
        async with sessions() as session:
            queue = PostgresQueue(session, SYSTEM)
            (job,) = await queue.claim()
            landed = await queue.fail(
                job.id, failure_class=FailureClass.SOURCE_UNAVAILABLE, detail="404"
            )
            await session.commit()
        assert landed is PaperState.FAILED

    def test_backoff_grows_and_stays_within_its_jitter_band(self) -> None:
        # Jitter must decorrelate retries without turning the schedule into
        # noise — a delay that could land anywhere is not a backoff.
        for attempts, base in ((1, 5.0), (2, 5.0), (3, 5.0)):
            span = base * 2 ** (attempts - 1)
            samples = [retry_delay(attempts, base=base) for _ in range(50)]
            assert all(span * 0.8 <= s <= span * 1.2 for s in samples)
            assert len(set(samples)) > 1, "jitter is not varying"


class TestCompletionIsTerminal:
    async def test_a_completed_job_is_never_claimed_again(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        job_id, _, _ = await enqueue(sessions)
        async with sessions() as session:
            queue = PostgresQueue(session, SYSTEM)
            (job,) = await queue.claim()
            await queue.complete(job.id)
            await session.commit()

        await expire_lease(sessions, job_id)
        async with sessions() as session:
            assert await PostgresQueue(session, SYSTEM).claim() == []
        assert (await job_row(sessions, job_id)).state == PaperState.INDEXED.value


class TestClaimingIsTheOnlyCrossTenantStep:
    async def test_a_tenant_scope_cannot_claim(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # D50. The claim reads other tenants' jobs by necessity; letting a
        # request-path scope do it would make that a general-purpose way to read
        # the whole queue.
        await enqueue(sessions)
        async with sessions() as session:
            queue = PostgresQueue(session, TenantContext(user_id=str(uuid.uuid4())))
            with pytest.raises(TenantIsolationViolation):
                await queue.claim()

    async def test_the_system_scope_still_can(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # Positive control: a guard that refused everything would pass the test
        # above and stop all ingestion.
        await enqueue(sessions)
        async with sessions() as session:
            assert len(await PostgresQueue(session, SYSTEM).claim()) == 1


class TestTheJobRowIsCommittedWithItsPaper:
    async def test_a_rollback_leaves_neither_the_job_nor_the_membership(
        self, sessions: async_sessionmaker[AsyncSession]
    ) -> None:
        # D47's reason for existing: with the queue inside the source of truth,
        # §5.4.2's "commit the job before enqueueing" has no window. A broker
        # would leave the paper queued forever or the job pointing at nothing.
        _, workspace_id, paper_id = await seed_paper(sessions)
        async with sessions() as session:
            await session.execute(
                text(
                    "INSERT INTO workspace_papers (workspace_id, paper_id, state) "
                    "VALUES (:w, :p, 'queued')"
                ),
                {"w": workspace_id, "p": paper_id},
            )
            await PostgresQueue(session, SYSTEM).enqueue(
                workspace_id=workspace_id, paper_id=paper_id
            )
            await session.rollback()

        async with sessions() as session:
            jobs = (await session.execute(text("SELECT count(*) FROM ingestion_jobs"))).scalar_one()
            members = (
                await session.execute(text("SELECT count(*) FROM workspace_papers"))
            ).scalar_one()
        assert (jobs, members) == (0, 0)
