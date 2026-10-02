"""Structured logging and the correlation id (FR-O1, §5.12, D107).

**JSON by default, because the consumer is a machine.** structlog's own default
renderer is built for a human reading a terminal; FR-O1 asks for structured logs
that a collector can index, and the two are different products. A `console`
format stays available for local work, so the choice is configuration rather
than a fork.

**The correlation id is ambient, not a parameter.** It lives in a `ContextVar`
that the API middleware sets per request and the worker sets per claimed job, so
nothing in `application/` has to carry an observability concern through its
signature — a use case that took a `correlation_id` would be describing logging
rather than the domain. Adapters read it where they need it; that is what
contextvars are for.

**Redaction happens here, at the boundary, and not at every call site** (FR-O1's
"redaction at the boundary"). A processor that strips known-sensitive keys is the
only version of this that holds: relying on every future `log.info` to remember
not to include an email is relying on the thing that will eventually be
forgotten. D17 and D30 are why content is on the list and not just credentials —
this system ingests possibly-unpublished manuscripts, so a logged chunk of paper
text is a disclosure, not a debugging aid.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

import structlog

#: Set per request by the API middleware, per claimed job by the worker (D107).
#: None outside both — a CLI or a test that never bound one, where "no id" is
#: the truth and a fabricated one would imply a request that never happened.
_CORRELATION_ID: ContextVar[str | None] = ContextVar("correlation_id", default=None)

#: The header the API reads an inbound id from and echoes on the way out.
CORRELATION_HEADER = "X-Request-ID"

#: Any key CONTAINING one of these is redacted, because credentials arrive under
#: many names and a denylist of exact spellings is the one that goes stale.
_SECRET_SUBSTRINGS = (
    "password",
    "token",
    "secret",
    "api_key",
    "apikey",
    "authorization",
    "credential",
)

#: Exact keys whose *values* are user or paper content rather than metadata.
#: Substring matching is wrong for these — "text" would swallow "context" and
#: "next_visible_at" — so they are named individually and deliberately.
#:
#: `system` and `user` are the two kwargs of `LLMProvider.stream`, so a log line
#: that included them would carry the whole prompt: the question plus every
#: retrieved chunk, which is the single largest piece of paper content the
#: process ever holds in one string.
_CONTENT_KEYS = frozenset(
    {
        "answer",
        "chunk_text",
        "email",
        "prompt",
        "query",
        "question",
        "system",
        "text",
        "user",
    }
)

REDACTED = "[redacted]"


def new_correlation_id() -> str:
    """A fresh id. Hex without dashes: it is read in log lines and grepped."""
    return uuid.uuid4().hex


def bind_correlation_id(value: str | None) -> None:
    """Make `value` the id for everything logged in this context."""
    _CORRELATION_ID.set(value)


def current_correlation_id() -> str | None:
    """The id in force, or None outside a request and outside a claimed job.

    Read by `PostgresQueue.enqueue` so the id reaches the job row (D107), and by
    the log processor below so it reaches every line without being passed.
    """
    return _CORRELATION_ID.get()


def _add_correlation_id(
    _logger: object, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Attach the ambient id, when there is one.

    Absent rather than null when unbound. A `correlation_id: null` field on every
    line from the eval harness would be noise in every record to describe a
    property of none of them.
    """
    correlation_id = current_correlation_id()
    if correlation_id is not None:
        event_dict["correlation_id"] = correlation_id
    return event_dict


def _redact(_logger: object, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Replace sensitive values, keeping the keys.

    The key survives on purpose: `email: "[redacted]"` tells a reader that the
    line was about a specific user, which is most of what the field was for,
    while dropping the field entirely would make a redacted record
    indistinguishable from one that never had the field.

    Nested dicts are walked; nothing else is. A secret smuggled inside a list of
    objects would survive this, and saying so is better than implying a
    guarantee — the contract is "known keys at any depth of mapping", which is
    the shape every logger in this codebase actually emits.
    """
    return {key: _redact_value(key, value) for key, value in event_dict.items()}


def _redact_value(key: str, value: Any) -> Any:
    if _is_sensitive(key):
        return REDACTED
    if isinstance(value, dict):
        return {inner: _redact_value(inner, item) for inner, item in value.items()}
    return value


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    if lowered in _CONTENT_KEYS:
        return True
    return any(marker in lowered for marker in _SECRET_SUBSTRINGS)


class _StdoutLogger:
    """Writes one rendered line to `sys.stdout`, looked up at call time.

    **Why not `structlog.PrintLoggerFactory(file=sys.stdout)`**, which is the
    obvious spelling: it captures the stream object when the logger is built, and
    `cache_logger_on_first_use` means that is once per process. Anything that
    later replaces `sys.stdout` — a supervisor redirecting output, a reload, a
    `contextlib.redirect_stdout`, or a test harness capturing per test — leaves
    every subsequent write going to the old object, and a *closed* old object
    raises.

    That is not hypothetical: it surfaced as an unrelated 500 in the rate-limit
    suite, because `create_app` configures logging and the first test to build an
    app bound the stream for the whole process. Resolving per write costs one
    attribute lookup and removes the class of bug.
    """

    def msg(self, message: str) -> None:
        stream = sys.stdout
        stream.write(f"{message}\n")
        stream.flush()

    log = debug = info = warn = warning = msg
    error = critical = exception = fatal = msg


class _StdoutLoggerFactory:
    """structlog's factory protocol: called with the logger name, returns a logger."""

    def __call__(self, *args: object) -> _StdoutLogger:  # noqa: ARG002 - protocol shape
        # structlog passes the logger name positionally. Ignored: the name is
        # already in the event dict via `add_logger_name`-style processors and the
        # renderer, so a logger that also embedded it would print it twice.
        return _StdoutLogger()


_configured = False


def configure_logging(*, level: str = "INFO", json_format: bool = True) -> None:
    """Install the processor chain. Idempotent.

    Idempotent because two composition roots call it and, with
    `worker_in_process` true (D63), one process is both of them. Re-running the
    configuration would be harmless today and is exactly the kind of thing that
    stops being harmless once a processor holds state.

    **The stream is resolved per write, not bound here** — see
    `_StdoutLoggerFactory`. Binding `sys.stdout` at configuration time is the
    obvious implementation and it is wrong for any process that later replaces the
    stream.

    `stdlib` logging is pointed at the same renderer so that uvicorn's and
    SQLAlchemy's records come out in the same shape. A log stream that is JSON
    for our lines and plain text for the framework's is not a structured log
    stream — it is two streams sharing a file descriptor.
    """
    global _configured
    if _configured:
        return

    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        _add_correlation_id,
        # Last before rendering, so it also covers keys other processors added.
        _redact,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderer: Any = (
        structlog.processors.JSONRenderer()
        if json_format
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[*shared, structlog.processors.format_exc_info, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=_StdoutLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            processor=renderer,
            foreign_pre_chain=shared,
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    _configured = True


def reset_for_tests() -> None:
    """Allow a test to reconfigure. Never called by production code."""
    global _configured
    _configured = False
