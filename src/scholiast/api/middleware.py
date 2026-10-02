"""Per-request correlation, logging and RED metrics (FR-O1, FR-O3, D110).

**Pure ASGI, not `BaseHTTPMiddleware`.** Starlette's convenience base wraps the
response body in an anyio stream, which adds buffering across a long-lived
response — and this API's most important endpoint is an SSE stream (D61, FR-Q5)
whose whole contract is that tokens arrive as they are produced. A middleware
that broke streaming to record that streaming happened would be self-defeating.

**An inbound correlation id is attacker-controlled and is validated, not
trusted.** Echoing an arbitrary header value back into a response and into every
log line is how a caller writes into someone else's logs. Anything that is not a
short, plain token is replaced by a fresh id rather than rejected: the request is
not wrong, only its label, and failing an upload over a malformed debug header
would be a worse outcome than relabelling it.

**The route LABEL is the template, never the path** (D110). A raw path would put
workspace and paper ids into an unauthenticated endpoint — the existence oracle
FR-A2 exists to close — and would grow the series count once per id ever seen. An
unmatched request is labelled `unmatched`, which is the whole point: a 404 probe
must not be able to mint a label.
"""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

import structlog
from starlette.datastructures import MutableHeaders

from scholiast.infrastructure.observability.logging import (
    CORRELATION_HEADER,
    bind_correlation_id,
    new_correlation_id,
)
from scholiast.infrastructure.observability.metrics import record_request

log = structlog.get_logger(__name__)

Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]

#: Bounded and plain. Long enough for a uuid4 hex or a trace id, narrow enough
#: that nothing in it can break a log record or an HTTP header.
_SAFE_CORRELATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

#: What a request that matched no route is labelled. A constant, so probing for
#: routes cannot create series.
UNMATCHED_ROUTE = "unmatched"


class ObservabilityMiddleware:
    """Binds a correlation id, times the request, and records the outcome."""

    def __init__(self, app: Callable[[Scope, Receive, Send], Awaitable[None]]) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            # Lifespan and websocket scopes pass through untouched: there is no
            # request to correlate and no status to record.
            await self._app(scope, receive, send)
            return

        correlation_id = _inbound_id(scope)
        bind_correlation_id(correlation_id)

        status = 500
        started = time.perf_counter()

        async def send_wrapper(message: MutableMapping[str, Any]) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message).append(CORRELATION_HEADER, correlation_id)
            await send(message)

        try:
            await self._app(scope, receive, send_wrapper)
        except Exception:
            # The status stays 500: an exception that escaped the app never sent
            # a response start, so there is no real status to record, and the
            # client is about to get a 500 from the server anyway.
            elapsed = time.perf_counter() - started
            route = _route_label(scope)
            record_request(method=_method(scope), route=route, status=status, seconds=elapsed)
            log.exception(
                "http.unhandled", method=_method(scope), route=route, duration_ms=elapsed * 1000
            )
            raise

        # For a streaming response this is the duration of the WHOLE stream, not
        # time to first byte. That is the honest reading of "how long did the
        # request take", and it is deliberately not the NFR-P3 number: time to
        # first token is measured inside the use case, which is the only place
        # that knows which byte was the first token rather than the first frame.
        elapsed = time.perf_counter() - started
        route = _route_label(scope)
        record_request(method=_method(scope), route=route, status=status, seconds=elapsed)
        log.info(
            "http.request",
            method=_method(scope),
            route=route,
            status=status,
            duration_ms=round(elapsed * 1000, 2),
        )


def _method(scope: Scope) -> str:
    method = scope.get("method")
    return method if isinstance(method, str) else "UNKNOWN"


def _inbound_id(scope: Scope) -> str:
    """The caller's id if it is safe to reuse, otherwise a fresh one."""
    headers: list[tuple[bytes, bytes]] = scope.get("headers") or []
    for raw_name, raw_value in headers:
        if raw_name.decode("latin-1").lower() != CORRELATION_HEADER.lower():
            continue
        candidate = raw_value.decode("latin-1")
        if _SAFE_CORRELATION_ID.match(candidate):
            return candidate
        break
    return new_correlation_id()


def _route_label(scope: Scope) -> str:
    """The matched route's template, or `unmatched`.

    Starlette puts the matched route on the scope during routing, so this is read
    AFTER the app has run. A request that matched nothing has no route, and its
    path is deliberately not substituted (D110).
    """
    path: object = getattr(scope.get("route"), "path", None)
    if isinstance(path, str):
        return path
    return UNMATCHED_ROUTE
