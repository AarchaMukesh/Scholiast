"""The metric set, and where each number comes from (FR-O3, §5.12, D108, D111).

**Two kinds of metric, because there are two kinds of fact.**

  in-process     things that happen *here*: HTTP requests, retrieval stage
                 timings, tokens spent, degradations disclosed. Counters and
                 histograms, incremented as they occur.

  read at scrape  things that are true of the *work*: how deep the queue is, how
                 old its oldest job is, how many papers are in which state.
                 Those live in Postgres because D95 put them there, and the
                 database is the authority both the API and the worker read.

D111 is the reasoning. `prometheus_client` counters are per-process and phase 3
moved ingestion into a different process, so an `ingestion_outcomes` counter
incremented in the Celery worker is invisible from the API's `/metrics`. Reading
the rows is not a workaround for missing plumbing — it is reading the same
source of truth. It also fails in the right direction: a gauge the worker pushed
would freeze at its last value when the worker died, reporting a healthy queue
exactly when nothing was draining it.

**A dedicated registry, not the global default.** The default is process-wide
mutable state: importing this module twice, or importing it in a test that also
builds an app, raises `Duplicated timeseries`. An explicit registry makes the
metric set a value this module owns.

**No label carries an identifier** (D110). Route templates, not paths; failure
classes, not paper ids. That is a privacy property — `/metrics` is
unauthenticated — and a cardinality one: a uuid in a label grows the series
count without bound.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import AbstractAsyncContextManager

from prometheus_client import CollectorRegistry, Counter, Gauge, generate_latest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.llm import TokenUsage

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

REGISTRY = CollectorRegistry()

# -- in-process: the API's own behaviour -----------------------------------

HTTP_REQUESTS = Counter(
    "scholiast_http_requests_total",
    "HTTP requests by method, route template and status class.",
    ("method", "route", "status"),
    registry=REGISTRY,
)

HTTP_DURATION = Gauge(
    "scholiast_http_request_duration_seconds_last",
    "Duration of the most recent request for this route. A gauge, not a "
    "histogram: NFR-P2's budget is checked by the eval harness against "
    "percentiles it computes itself, and a histogram here would invite reading "
    "latency conclusions off a dashboard that cross-run drift makes "
    "incomparable (CLAUDE.md's latency trap).",
    ("method", "route"),
    registry=REGISTRY,
)

RETRIEVAL_STAGE = Gauge(
    "scholiast_retrieval_stage_duration_seconds_last",
    "Most recent duration of each retrieval stage (§5.12's per-stage timings).",
    ("stage",),
    registry=REGISTRY,
)

GENERATION_TOKENS = Counter(
    "scholiast_generation_tokens_total",
    "Tokens spent, by provider and direction. Quota is the scarce resource "
    "here, not money (D14) — so this is the number that replaces cost.",
    ("provider", "direction"),
    registry=REGISTRY,
)

DEGRADATIONS = Counter(
    "scholiast_degradations_total",
    "Answers served in a degraded mode, by mode (§5.11). Expected to stay near "
    "zero; every increment is a rung of the ladder that fired.",
    ("mode",),
    registry=REGISTRY,
)

BREAKER_STATE = Gauge(
    "scholiast_provider_breaker_state",
    "0 closed, 1 half-open, 2 open. Not in §5.12 — added because it is the one "
    "signal that says WHY a provider_failover is happening rather than only "
    "that it did (D104, D108).",
    ("provider",),
    registry=REGISTRY,
)

_BREAKER_VALUES = {"closed": 0, "half_open": 1, "open": 2}

# -- read at scrape time: facts about the work (D111) ----------------------

QUEUE_DEPTH = Gauge(
    "scholiast_ingestion_queue_depth",
    "Ingestion jobs by state, read from `ingestion_jobs` at scrape time.",
    ("state",),
    registry=REGISTRY,
)

OLDEST_JOB_AGE = Gauge(
    "scholiast_ingestion_oldest_job_age_seconds",
    "Age of the oldest claimable job. The queue-is-not-draining alarm: depth "
    "alone cannot distinguish a busy queue from a stalled one.",
    registry=REGISTRY,
)

PAPERS_BY_STATE = Gauge(
    "scholiast_papers_by_state",
    "Workspace memberships by paper state — ingestion outcomes, as the source "
    "of truth records them.",
    ("state",),
    registry=REGISTRY,
)

PAPERS_BY_FAILURE_CLASS = Gauge(
    "scholiast_papers_by_failure_class",
    "Failed and unsupported papers by §4.4.1 failure class.",
    ("failure_class",),
    registry=REGISTRY,
)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


def record_request(*, method: str, route: str, status: int, seconds: float) -> None:
    """RED for one request. `route` must be a template, never a raw path (D110)."""
    HTTP_REQUESTS.labels(method=method, route=route, status=_status_class(status)).inc()
    HTTP_DURATION.labels(method=method, route=route).set(seconds)


def _status_class(status: int) -> str:
    """`2xx`, `4xx`, `5xx`. Bucketed because the alert is on the class.

    Keeping the exact code would multiply the series for a distinction nothing
    alerts on — and a 404 and a 403 differing in the metric would reintroduce
    the existence oracle FR-A2 closes at the HTTP layer.
    """
    return f"{status // 100}xx"


def record_stages(timings: Mapping[str, float]) -> None:
    """Per-stage retrieval timings, in seconds."""
    for stage, seconds in timings.items():
        RETRIEVAL_STAGE.labels(stage=stage).set(seconds)


def record_generation(*, provider: str | None, usage: TokenUsage | None) -> None:
    """Token spend. Skipped when the provider did not report it.

    `TokenUsage.reported` is what makes that distinguishable: adding zeros for a
    provider that returns no usage would report a provider costing nothing, when
    what happened is that nobody counted (see the port's docstring).
    """
    if provider is None or usage is None or not usage.reported:
        return
    GENERATION_TOKENS.labels(provider=provider, direction="prompt").inc(usage.prompt_tokens)
    GENERATION_TOKENS.labels(provider=provider, direction="completion").inc(usage.completion_tokens)


def record_degradations(modes: tuple[str, ...]) -> None:
    for mode in modes:
        DEGRADATIONS.labels(mode=mode).inc()


def refresh_breaker_states(providers: object) -> None:
    """Read each breaker's state. Silent about providers that are not breakers.

    Duck-typed rather than isinstance-checked against `CircuitBreaker`: a test
    app wires bare fakes, and a metrics refresh that raised on them would make
    `/metrics` a liability in exactly the configuration used to test it.
    """
    if not isinstance(providers, list):
        return
    for provider in providers:
        state = getattr(provider, "state", None)
        name = getattr(provider, "name", None)
        if state is None or not isinstance(name, str):
            continue
        value = _BREAKER_VALUES.get(str(state))
        if value is not None:
            BREAKER_STATE.labels(provider=name).set(value)


async def refresh_work_gauges(sessions: SessionFactory) -> None:
    """Read the queue and membership tables (D111).

    One aggregate query per gauge family, each a grouped count over an indexed
    column. **Gauges are cleared first**, because a label that stops appearing in
    the result would otherwise keep its last value forever: an empty queue would
    go on reporting the depth it had when it drained.
    """
    async with sessions() as session:
        jobs = (
            await session.execute(
                text("SELECT state, count(*) AS n FROM ingestion_jobs GROUP BY state")
            )
        ).all()
        oldest = (
            await session.execute(
                text(
                    "SELECT COALESCE(MAX(EXTRACT(EPOCH FROM (now() - created_at))), 0) AS age "
                    "FROM ingestion_jobs WHERE state IN "
                    "('queued', 'fetching', 'parsing', 'chunking', 'embedding')"
                )
            )
        ).scalar_one()
        papers = (
            await session.execute(
                text("SELECT state, count(*) AS n FROM workspace_papers GROUP BY state")
            )
        ).all()
        failures = (
            await session.execute(
                text(
                    "SELECT failure_class, count(*) AS n FROM workspace_papers "
                    "WHERE failure_class IS NOT NULL GROUP BY failure_class"
                )
            )
        ).all()

    QUEUE_DEPTH.clear()
    for row in jobs:
        QUEUE_DEPTH.labels(state=row.state).set(row.n)
    OLDEST_JOB_AGE.set(float(oldest or 0.0))
    PAPERS_BY_STATE.clear()
    for row in papers:
        PAPERS_BY_STATE.labels(state=row.state).set(row.n)
    PAPERS_BY_FAILURE_CLASS.clear()
    for row in failures:
        PAPERS_BY_FAILURE_CLASS.labels(failure_class=row.failure_class).set(row.n)


def render() -> bytes:
    """The exposition format for everything in the registry."""
    return generate_latest(REGISTRY)
