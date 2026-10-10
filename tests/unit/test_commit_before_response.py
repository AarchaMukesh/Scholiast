"""A 2xx means the write is committed (D166).

Until D166 both session dependencies committed in FastAPI's request-scoped
teardown, which runs after the response has been sent. A client could hold a 2xx
before its write was durable, and its next request read the state from before it:
the UI's workspace counter stuck at "0/1 ready" after adding a paper (its
follow-up list request came back empty, so it never polled again), and a commit
that failed after the response left the client holding a success.

These run the real dependencies in a minimal app over a recording session, and
decide by the ORDER of events (the commit, the ASGI messages that send the
response, the worker's wake-up), not by timing, so no server and no database is
needed: the in-process transport hides the race in time, never in order. The
route guard at the end holds the real app to the rule.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute

from scholiast.api import deps
from scholiast.api.app import create_app
from scholiast.application.ports.queue import JobPriority
from scholiast.config.settings import Settings
from scholiast.infrastructure.db import pending

ANSWER_ROUTE = ("POST", "/v1/workspaces/{workspace_id}/conversations/{conversation_id}/messages")


class RecordingSession:
    """Just enough of an AsyncSession: it records writes, commits and rollbacks."""

    def __init__(self, log: list[str], *, fail_commit: bool = False) -> None:
        self.log, self.info, self._fail = log, {}, fail_commit

    def write(self, what: str) -> None:
        self.log.append(f"write:{what}")

    async def commit(self) -> None:
        if self._fail:
            raise RuntimeError("the database refused the commit")
        self.log.append("commit")

    async def rollback(self) -> None:
        self.log.append("rollback")


class RecordingDispatcher:
    publishes = False

    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def dispatch(
        self, job_id: str, *, priority: JobPriority = JobPriority.INTERACTIVE
    ) -> bool:
        self.log.append(f"dispatch:{job_id}")
        return True

    async def dispatch_deletion(self, workspace_id: str) -> bool:
        self.log.append(f"dispatch-deletion:{workspace_id}")
        return True


def recording_app(log: list[str], *, fail_commit: bool = False) -> FastAPI:
    app = FastAPI()

    @asynccontextmanager
    async def factory() -> AsyncIterator[RecordingSession]:
        yield RecordingSession(log, fail_commit=fail_commit)

    app.state.sessions = factory
    app.state.dispatcher = RecordingDispatcher(log)

    @app.post("/write")
    async def write(session: deps.SessionDep) -> dict[str, str]:
        session.write("row")  # type: ignore[attr-defined]
        return {"ok": "yes"}

    @app.post("/write-a-job")
    async def write_a_job(session: deps.SessionDep) -> dict[str, str]:
        session.write("job")  # type: ignore[attr-defined]
        pending.record(session, pending.PENDING_DISPATCH, "job-1")
        return {"ok": "yes"}

    @app.post("/auth-write")
    async def auth_write(session: deps.AuthSessionDep) -> dict[str, str]:
        session.write("token")  # type: ignore[attr-defined]
        return {"ok": "yes"}

    @app.post("/stream")
    async def stream(session: deps.StreamSessionDep) -> StreamingResponse:
        async def body() -> AsyncIterator[str]:
            yield "event: token\n\n"
            # As AnswerQuestionUseCase._persist does: the turns are written while
            # the body streams, after this handler has returned.
            session.write("turns")  # type: ignore[attr-defined]
            yield "event: done\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    return app


def recorded(app: FastAPI, log: list[str]) -> Callable[..., Any]:
    """The app, with every ASGI message it sends written to the log as it goes."""

    async def asgi(scope: Any, receive: Any, send: Any) -> None:
        async def logging_send(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                log.append("response:start")
            elif message["type"] == "http.response.body" and not message.get("more_body", False):
                log.append("response:end")
            await send(message)

        await app(scope, receive, logging_send)

    return asgi


async def call(app: FastAPI, log: list[str], path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=recorded(app, log), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as client:
        return await client.post(path)


class TestAWriteIsCommittedBeforeItsResponse:
    async def test_a_write_is_committed_before_the_response_starts(self) -> None:
        log: list[str] = []

        response = await call(recording_app(log), log, "/write")

        assert response.status_code == 200
        assert log.index("commit") < log.index("response:start"), log

    async def test_an_authentication_write_is_committed_before_the_response_starts(self) -> None:
        # Register, then log in at once: the login must find the account.
        log: list[str] = []

        response = await call(recording_app(log), log, "/auth-write")

        assert response.status_code == 200
        assert log.index("commit") < log.index("response:start"), log

    async def test_a_commit_that_fails_is_an_error_not_a_success(self) -> None:
        # Committed after the response, the client already held a 200 for a
        # write that was never stored.
        log: list[str] = []

        response = await call(recording_app(log, fail_commit=True), log, "/write")

        assert response.status_code == 500
        assert "commit" not in log


class TestTheWorkerIsWokenAfterTheResponse:
    async def test_a_committed_job_is_still_dispatched(self) -> None:
        log: list[str] = []

        await call(recording_app(log), log, "/write-a-job")

        assert "dispatch:job-1" in log

    async def test_the_wake_up_waits_for_the_response_not_the_other_way_round(self) -> None:
        # A broker that is down makes `send_task` wait out its retries; that wait
        # must not be any user's request (D95, D166).
        log: list[str] = []

        await call(recording_app(log), log, "/write-a-job")

        assert log.index("commit") < log.index("response:end") < log.index("dispatch:job-1"), log


class TestAStreamedAnswerKeepsItsSessionUntilTheStreamEnds:
    async def test_writes_made_while_streaming_are_committed(self) -> None:
        # A session closed when the handler returned would be gone before the
        # turns are written, and they would be lost without an error.
        log: list[str] = []

        response = await call(recording_app(log), log, "/stream")

        assert response.status_code == 200
        assert log.index("write:turns") < log.index("commit"), log


def _sessions_in(dependant: Any) -> list[tuple[Any, str | None]]:
    """Every session dependency in a route's dependency tree, with its scope."""
    found = []
    stack = [dependant]
    while stack:
        node = stack.pop()
        if node.call in (deps.get_session, deps.auth_session, deps.stream_session):
            found.append((node.call, node.scope))
        stack.extend(node.dependencies)
    return found


def _routes() -> list[APIRoute]:
    """Every API route, including those inside included routers (`original_router`)."""
    found: list[APIRoute] = []
    stack: list[Any] = list(create_app(Settings()).routes)
    while stack:
        route = stack.pop()
        if isinstance(route, APIRoute):
            found.append(route)
        inner = getattr(route, "original_router", None)
        stack.extend(getattr(inner, "routes", None) or getattr(route, "routes", None) or ())
    return found


class TestEveryRouteCommitsBeforeItRespondsExceptTheStreamedAnswer:
    def test_the_guard_sees_the_routes_it_guards(self) -> None:
        with_sessions = [r for r in _routes() if _sessions_in(r.dependant)]

        assert len(with_sessions) > 10, "the guard found almost no routes; has the wiring moved?"
        assert any((m, r.path) == ANSWER_ROUTE for r in with_sessions for m in r.methods)

    def test_every_ordinary_and_authentication_session_is_function_scoped(self) -> None:
        late = [
            (sorted(r.methods), r.path, call.__name__)
            for r in _routes()
            for call, scope in _sessions_in(r.dependant)
            if call in (deps.get_session, deps.auth_session) and scope != "function"
        ]

        assert late == []

    def test_only_streaming_routes_take_the_session_that_outlives_the_handler(self) -> None:
        streaming = {
            (m, r.path)
            for r in _routes()
            for m in r.methods
            if inspect.signature(r.endpoint).return_annotation
            in (StreamingResponse, "StreamingResponse")
        }
        takes_stream_session = {
            (m, r.path)
            for r in _routes()
            for m in r.methods
            if any(call is deps.stream_session for call, _ in _sessions_in(r.dependant))
        }

        assert streaming == takes_stream_session == {ANSWER_ROUTE}

    def test_the_streamed_answer_holds_one_session(self) -> None:
        # Its repositories built from the ordinary session would put two sessions
        # in one request, and the one the turns were written to would close first.
        [route] = [r for r in _routes() for m in r.methods if (m, r.path) == ANSWER_ROUTE]

        assert {call for call, _ in _sessions_in(route.dependant)} == {deps.stream_session}
