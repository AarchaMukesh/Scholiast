"""Discovery search (FR-D1, FR-D6, §5.4.1, D122).

`POST`, not `GET`, for D61's reason: a query in a URL ends up in access logs and
proxies, and a research topic can say more about unpublished work than its author
would like logged. The body carries it instead, and the log redaction already
covers a field called `query`.

**What leaves and what does not.** Each source's status, result count and
`retry_after` reach the client; why a source failed does not. That text comes from
a third party, and it is logged here instead, with the traceback when the failure
was a bug rather than the source saying no.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends

from scholiast.api import schemas
from scholiast.api.deps import DiscoverDep, DiscoveryRateLimit, current_tenant
from scholiast.application.ports.discovery import SourceUnavailable
from scholiast.application.use_cases.discover_papers import (
    NoSourceAnswered,
    SourceOutcome,
    SourceStatus,
)
from scholiast.domain.discovery import PaperCandidate
from scholiast.infrastructure.observability import metrics

router = APIRouter(prefix="/v1/discovery", tags=["discovery"])
log = structlog.get_logger(__name__)


def _candidate(candidate: PaperCandidate) -> schemas.CandidateOut:
    return schemas.CandidateOut(
        source=candidate.source,
        source_id=candidate.source_id,
        source_rank=candidate.rank,
        title=candidate.title,
        authors=list(candidate.authors),
        year=candidate.year,
        published=candidate.published,
        venue=candidate.venue,
        abstract=candidate.abstract,
        citation_count=candidate.citation_count,
        doi=candidate.doi,
        arxiv_id=candidate.arxiv_id,
        open_access=[
            schemas.OpenAccessLinkOut(url=link.url, kind=link.kind.value, licence=link.licence)
            for link in candidate.oa_links
        ],
    )


def _source_status(outcome: SourceOutcome) -> schemas.SourceStatusOut:
    return schemas.SourceStatusOut(
        source=outcome.source,
        status=outcome.status.value,
        results=len(outcome.candidates),
        total_hits=outcome.total_hits,
        retry_after_seconds=outcome.retry_after,
    )


def _observe(outcomes: tuple[SourceOutcome, ...]) -> None:
    """Metrics for every source, and a log line for every one that did not answer."""
    metrics.record_discovery((o.source, o.status.value, o.seconds) for o in outcomes)
    for outcome in outcomes:
        if outcome.status is SourceStatus.OK:
            continue
        error = outcome.error
        if error is not None and not isinstance(error, (SourceUnavailable, TimeoutError)):
            # Not the source saying no: a bug, isolated to this source (FR-D6).
            log.error(
                "discovery.source_crashed",
                source=outcome.source,
                exc_info=(type(error), error, error.__traceback__),
            )
        else:
            log.warning(
                "discovery.source_unavailable",
                source=outcome.source,
                status=outcome.status.value,
                detail=str(error)[:300] if error is not None else None,
                retry_after=outcome.retry_after,
            )


# Login is its own dependency, not a side effect of the limiter's: the limiter needs
# the principal too, but a route whose authentication depended on its rate limit
# would lose both to one edit. FastAPI resolves `current_tenant` once per request.
@router.post(
    "/search",
    response_model=schemas.DiscoveryResults,
    dependencies=[Depends(current_tenant), DiscoveryRateLimit],
)
async def search(body: schemas.DiscoverySearch, discover: DiscoverDep) -> schemas.DiscoveryResults:
    try:
        result = await discover.execute(body.query)
    except NoSourceAnswered as exc:
        _observe(exc.sources)
        raise
    _observe(result.sources)
    return schemas.DiscoveryResults(
        results=[_candidate(c) for c in result.candidates],
        source_status=[_source_status(o) for o in result.sources],
        partial=result.partial,
    )
