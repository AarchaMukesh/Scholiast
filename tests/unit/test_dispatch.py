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
from scholiast.infrastructure.db.pending import PENDING_DELETION_DISPATCH, PENDING_DISPATCH
from scholiast.infrastructure.queue.celery_app import (
    DELETE_TASK,
    INGEST_QUEUE,
    INGEST_TASK,
    MAINTENANCE_QUEUE,
    RECONCILE_TASK,
)
from scholiast.infrastructure.queue.dispatch import CeleryDispatcher, NullDispatcher


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
    #: True, because this fake stands in for a real broker. A False here would
    #: silence the warning the composition root logs on a failed publish.
    publishes = True

    def __init__(self) -> None:
        self.dispatched: list[str] = []
        self.deleted: list[str] = []

    async def dispatch(self, job_id: str) -> bool:
        self.dispatched.append(job_id)
        return True

    async def dispatch_deletion(self, workspace_id: str) -> bool:
        self.deleted.append(workspace_id)
        return True


class FakeSession:
    """Just the `info` dict the real session exposes."""

    def __init__(
        self, pending: list[str] | None = None, deletions: list[str] | None = None
    ) -> None:
        self.info: dict[str, Any] = {}
        if pending is not None:
            self.info[PENDING_DISPATCH] = pending
        if deletions is not None:
            self.info[PENDING_DELETION_DISPATCH] = deletions


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

    async def test_a_deletion_is_addressed_by_name_to_the_maintenance_queue(self) -> None:
        # D99 + D100: same by-name contract as ingestion, different queue,
        # because a cascade must not occupy the slot a user's upload wants.
        app = FakeCelery()
        assert await CeleryDispatcher(app).dispatch_deletion("ws-1") is True
        assert app.sent == [(DELETE_TASK, ["ws-1"], MAINTENANCE_QUEUE)]

    async def test_a_broker_failure_on_a_deletion_is_reported_not_raised(self) -> None:
        # The 202 has already gone out and the workspace already stopped serving
        # reads. The backstop is `list_deleting`, not the caller.
        assert await CeleryDispatcher(FakeCelery(fails=True)).dispatch_deletion("ws-1") is False

    async def test_the_null_dispatcher_declines_deletions_too(self) -> None:
        assert await NullDispatcher().dispatch_deletion("ws-1") is False

    def test_only_a_real_broker_claims_to_publish(self) -> None:
        """FR-O1's log line depends on this distinction being in the contract.

        `dispatch` returns False both when a broker was unreachable and when
        there is no broker by design (D63's single-process topology). Without
        `publishes` to tell them apart, the composition root's warning would fire
        on EVERY upload in the default development configuration — which is the
        mistake D106 names: a warning on the normal path carries no information.
        """
        assert CeleryDispatcher(FakeCelery()).publishes is True
        assert NullDispatcher().publishes is False


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
        assert dispatcher.deleted == []

    async def test_committed_deletions_are_dispatched(self) -> None:
        dispatcher = RecordingDispatcher()
        session = FakeSession(deletions=["ws-1"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.deleted == ["ws-1"]

    async def test_jobs_and_deletions_never_cross_over(self) -> None:
        # Two keys rather than one tagged list (D99): they go to different tasks
        # on different queues, and a single list of bare strings would let a
        # workspace id be published as an ingestion job with nothing to catch it.
        dispatcher = RecordingDispatcher()
        session = FakeSession(["job-1"], deletions=["ws-1"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.dispatched == ["job-1"]
        assert dispatcher.deleted == ["ws-1"]

    async def test_the_deletion_list_is_consumed_too(self) -> None:
        dispatcher = RecordingDispatcher()
        session = FakeSession(deletions=["ws-1"])
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        await _dispatch_committed_jobs(FakeRequest(dispatcher), session)  # type: ignore[arg-type]
        assert dispatcher.deleted == ["ws-1"]

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

    def test_the_deletion_task_name_agrees_too(self) -> None:
        """D96's failure mode, for the cascade (D99).

        A desynchronised deletion name is *worse* than a desynchronised ingest
        name. An undispatched job still runs at the poll interval; an
        undispatched cascade waits for the reconciliation sweep, which runs
        every five minutes by default — so the workspace the user deleted keeps
        its rows, vectors and blobs for that long with nothing reporting why.
        """
        from scholiast.workers import tasks

        assert tasks.delete_workspace.name == DELETE_TASK
        assert tasks.reconcile.name == RECONCILE_TASK

    def test_the_scheduled_sweep_names_a_task_that_exists(self) -> None:
        # The beat schedule is configuration, so a typo in it fails at no import
        # and no test — the sweep simply never runs, and the only symptom is
        # residue nobody is looking for.
        from scholiast.workers import tasks

        scheduled = {entry["task"] for entry in tasks.app.conf.beat_schedule.values()}
        assert scheduled == {RECONCILE_TASK}
        assert RECONCILE_TASK in tasks.app.tasks

    def test_maintenance_work_is_routed_off_the_ingest_queue(self) -> None:
        # D100. With `worker_prefetch_multiplier=1` a worker holds one message
        # at a time, so a cascade sharing `ingest` would make a user's upload
        # wait behind somebody else's delete.
        from scholiast.workers import tasks

        routes = tasks.app.conf.task_routes
        assert routes[DELETE_TASK]["queue"] == MAINTENANCE_QUEUE
        assert routes[RECONCILE_TASK]["queue"] == MAINTENANCE_QUEUE
        assert routes[INGEST_TASK]["queue"] == INGEST_QUEUE

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
