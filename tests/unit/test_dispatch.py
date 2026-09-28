"""Dispatch semantics (§5.4.2, D95, D96).

M3's acceptance criterion — *killing a worker mid-ingestion loses no accepted
job* — is already proven at the queue level by M1's lease tests, and D95 is the
reason Celery does not disturb that: the job lives in Postgres, and the broker is
told about it afterwards. What is new in phase 3, and therefore what is tested
here, is the *telling*:

  after the commit    a dispatch published inside the transaction would announce
                      a job a rollback then erased, and a worker quick enough to
                      claim it would find nothing.

  never fatal         by the time it runs the paper is accepted; a broker that is
                      down must cost latency, not the upload.

  by name             the API addresses `scholiast.ingest_paper` as a string,
                      because importing the task would couple two deployables the
                      layering keeps apart (D96).
"""

from __future__ import annotations

from typing import Any

import pytest

from scholiast.api.deps import _dispatch_committed_jobs
from scholiast.infrastructure.queue.celery_app import INGEST_QUEUE, INGEST_TASK
from scholiast.infrastructure.queue.dispatch import CeleryDispatcher, NullDispatcher
from scholiast.infrastructure.queue.postgres import PENDING_DISPATCH


class FakeCelery:
    """Records what would have gone on the wire."""

    def __init__(self, *, fails: bool = False) -> None:
        self.sent: list[tuple[str, list[Any], str]] = []
        self._fails = fails

    def send_task(self, name: str, args: list[Any], queue: str) -> None:
        if self._fails:
            raise ConnectionError("broker unreachable")
        self.sent.append((name, args, queue))


class RecordingDispatcher:
    def __init__(self) -> None:
        self.dispatched: list[str] = []

    async def dispatch(self, job_id: str) -> bool:
        self.dispatched.append(job_id)
        return True


class FakeSession:
    """Just the `info` dict the real session exposes."""

    def __init__(self, pending: list[str] | None = None) -> None:
        self.info: dict[str, Any] = {PENDING_DISPATCH: pending} if pending is not None else {}


class FakeRequest:
    def __init__(self, dispatcher: Any) -> None:
        self.app = type("App", (), {"state": type("State", (), {"dispatcher": dispatcher})()})()


class TestTheCeleryDispatcher:
    async def test_it_addresses_the_task_by_name(self) -> None:
        # D96. The string is the contract between two deployables; the API never
        # imports the function, which `lint-imports` enforces separately.
        app = FakeCelery()
        assert await CeleryDispatcher(app).dispatch("job-1") is True
        assert app.sent == [(INGEST_TASK, ["job-1"], INGEST_QUEUE)]

    async def test_a_broker_failure_is_reported_not_raised(self) -> None:
        # The upload already succeeded. Raising here would fail a request whose
        # work is committed and will run regardless.
        assert await CeleryDispatcher(FakeCelery(fails=True)).dispatch("job-1") is False

    async def test_the_null_dispatcher_does_nothing_and_says_so(self) -> None:
        # Single-process topology: the in-process worker already polls the same
        # database, so there is nothing to wake.
        assert await NullDispatcher().dispatch("job-1") is False


class TestDispatchHappensAfterTheCommit:
    async def test_committed_jobs_are_dispatched(self) -> None:
        dispatcher = RecordingDispatcher()
        session = FakeSession(["job-1", "job-2"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.dispatched == ["job-1", "job-2"]

    async def test_the_pending_list_is_consumed(self) -> None:
        # Drained rather than read: a session reused for a second unit of work
        # must not re-announce the first one's jobs.
        dispatcher = RecordingDispatcher()
        session = FakeSession(["job-1"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.dispatched == ["job-1"]

    async def test_nothing_is_dispatched_when_nothing_was_enqueued(self) -> None:
        dispatcher = RecordingDispatcher()
        await _dispatch_committed_jobs(FakeRequest(dispatcher), FakeSession())  # type: ignore[arg-type]
        assert dispatcher.dispatched == []

    async def test_a_dispatch_failure_does_not_stop_the_rest(self) -> None:
        # One unreachable publish must not strand the other jobs in the same
        # transaction; each is independently recoverable by the lease anyway.
        class OneBadDispatcher(RecordingDispatcher):
            async def dispatch(self, job_id: str) -> bool:
                if job_id == "job-1":
                    return False
                return await super().dispatch(job_id)

        dispatcher = OneBadDispatcher()
        session = FakeSession(["job-1", "job-2"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.dispatched == ["job-2"]


class TestTheTaskNameIsNotAccidental:
    def test_the_api_side_and_the_worker_side_agree(self) -> None:
        """Both ends import the same constant, so a rename cannot desynchronise.

        If the API sent one string and the worker registered another, every
        dispatch would vanish into the broker and every job would still run —
        just at the poll interval instead of immediately. A silent latency
        regression with nothing failing, which is why the name is a constant.
        """
        from scholiast.workers import tasks

        assert tasks.ingest_paper.name == INGEST_TASK

    def test_the_api_does_not_import_the_worker_package(self) -> None:
        # The architectural claim of D96, asserted here as well as by
        # `lint-imports`: this one fails with a readable message pointing at the
        # module that did it.
        import scholiast.api.deps as deps_module

        offenders = [
            name
            for name in dir(deps_module)
            if getattr(getattr(deps_module, name, None), "__module__", "").startswith(
                "scholiast.workers"
            )
        ]
        assert offenders == []


@pytest.mark.parametrize("job_id", ["", "not-a-uuid"])
async def test_the_dispatcher_does_not_validate_the_id(job_id: str) -> None:
    """Deliberately dumb: the id came from a row this transaction just wrote.

    Validating here would duplicate a guarantee the database already made, and
    would turn a malformed id — which cannot happen — into a failed upload.
    """
    app = FakeCelery()
    assert await CeleryDispatcher(app).dispatch(job_id) is True
    assert app.sent[0][1] == [job_id]
