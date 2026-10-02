"""The ops surface: liveness, readiness and metrics (FR-O2, FR-O3, D110, D111).

**Readiness reports four things and gates on two**, and the split is what makes
the degradation ladder possible. FR-O2 asks for "dependency status (DB, vector
store, queue, model availability)" — it asks for *reporting*, not for requiring —
and §5.11 says a generation outage degrades to evidence-only rather than failing.
A readiness probe that went red when Ollama stopped would pull the process out of
rotation for a condition the system is designed to serve through: the ladder
could never fire in a load-balanced deployment, because no instance would ever be
routed to while degraded.

  Postgres      **required.** §5.11: "no meaningful degradation for the source of
                truth". Without it there is no auth, no workspace and no queue.
  Qdrant        **required.** Losing it loses retrieval entirely — both arms live
                there (D105) — so an instance without it cannot answer.
  the queue     **covered by the Postgres check**, because under D95 Postgres IS
                the queue. Redis is the transport, and a broker outage costs
                latency rather than work: the lease makes an unclaimed job
                claimable and the sweep finds it. So it is deliberately not
                probed — pinging it every few seconds to report a non-blocking
                condition would add load to say nothing actionable.
  providers     **reported, never required**, and read from the circuit breakers
                rather than dialled. The breaker already knows what this process
                thinks of each provider (D104), so this costs no network call and
                is a better signal than a fresh probe: it reports the state the
                next request will actually get.

**Unauthenticated, and phase 8 owns that** (D110). `/health/ready` naming an
unhealthy dependency tells an anonymous caller something about the deployment;
`/metrics` exposes aggregate counters with no identifier in any label.
"""

from __future__ import annotations

from hmac import compare_digest
from http import HTTPStatus
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from sqlalchemy import text

from scholiast.infrastructure.observability import metrics

log = structlog.get_logger(__name__)

router = APIRouter(tags=["health"])


def require_scrape_token(request: Request) -> None:
    """Gate the ops endpoints on a static token, when one is configured (D114).

    **Optional by design.** Requiring it unconditionally would break
    `docker compose up` plus one command and every local `curl`, for an endpoint
    that on a development machine is exposed to nobody. Off by default means a
    deployment has something to switch on, which is what closing D110 was for.

    A static token, not phase 1's bearer JWT: a scraper polls every fifteen
    seconds forever and cannot refresh something that expires. The comparison is
    constant-time, because a token checked with `==` leaks its prefix to anyone
    willing to measure — the same reasoning phase 1 applied to password
    verification, for a secret that travels on every scrape.

    `/health/live` is NOT gated: an orchestrator's restart decision must not
    require a credential, and it answers the same thing whatever is wrong.
    """
    configured: str | None = request.app.state.settings.metrics_token
    if not configured:
        return
    header = request.headers.get("authorization", "")
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not compare_digest(presented, configured):
        # 401 with `WWW-Authenticate`, per RFC 9110 — the same shape phase 1's
        # unauthenticated handler uses, so a scraper sees one convention.
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/health/live")
async def live() -> dict[str, str]:
    """The process is running and the event loop is turning."""
    return {"status": "ok"}


@router.get("/health/ready", dependencies=[Depends(require_scrape_token)])
async def ready(request: Request, response: Response) -> dict[str, Any]:
    """Dependency status, with a 503 when something required is down (FR-O2).

    M1's docstring said a readiness probe that reports ready without checking
    anything "is worse than no probe at all: a green light nobody verified". This
    verifies, and it reports each dependency by name so the green light is
    accountable.
    """
    checks: dict[str, str] = {
        "postgres": await _postgres(request),
        "qdrant": await _qdrant(request),
    }
    # `not_configured` is not a failure. A composition root that wired no index
    # — the auth-only and isolation lifespans do exactly that — is not an
    # unhealthy deployment, and gating on it would make those apps permanently
    # not-ready. Only a dependency that is present AND unreachable is a veto.
    required_ok = all(status in {"ok", "not_configured"} for status in checks.values())

    # Reported below the gate: their state is information, not a veto.
    checks.update(_providers(request))

    if not required_ok:
        response.status_code = HTTPStatus.SERVICE_UNAVAILABLE
    return {"status": "ready" if required_ok else "not_ready", "checks": checks}


@router.get("/metrics", dependencies=[Depends(require_scrape_token)])
async def prometheus_metrics(request: Request) -> Response:
    """The exposition format (FR-O3).

    Work gauges are refreshed from Postgres on the way in (D111), because they
    are facts about the job table rather than counters this process kept. A
    database that will not answer costs those gauges their freshness and nothing
    else: the scrape still returns the in-process metrics, since a monitoring
    endpoint that fails completely during an incident is a monitoring endpoint
    that is absent during every incident.
    """
    metrics.refresh_breaker_states(getattr(request.app.state, "providers", None))
    try:
        await metrics.refresh_work_gauges(request.app.state.sessions)
    except Exception:
        # Debug, not error. The readiness probe and every other endpoint are
        # already shouting about a Postgres outage, so a second alarm from the
        # monitoring endpoint would add noise to an incident rather than
        # information — but it IS recorded, for whoever later asks why the queue
        # gauges have a gap.
        log.debug("metrics.work_gauges_unavailable", exc_info=True)
    return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE)


async def _postgres(request: Request) -> str:
    try:
        async with request.app.state.sessions() as session:
            await session.execute(text("SELECT 1"))
    except Exception:
        return "unavailable"
    return "ok"


async def _qdrant(request: Request) -> str:
    index = getattr(request.app.state, "index", None)
    if index is None:
        # A composition root that wired no index — the auth-only and
        # isolation test lifespans do exactly this. Absent is not unhealthy, and
        # reporting it as such would make those apps permanently not-ready.
        return "not_configured"
    return "ok" if await index.ping() else "unavailable"


def _providers(request: Request) -> dict[str, str]:
    """Each provider's breaker state, free of charge (D104)."""
    providers = getattr(request.app.state, "providers", None)
    if not isinstance(providers, list):
        return {}
    reported: dict[str, str] = {}
    for provider in providers:
        name = getattr(provider, "name", None)
        state = getattr(provider, "state", None)
        if isinstance(name, str):
            reported[f"provider:{name}"] = "ok" if state is None else str(state)
    return reported
