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

from typing import Literal

import structlog
from fastapi import APIRouter

from scholiast.api import schemas
from scholiast.api.deps import DiscoverDep, DiscoveryRateLimit, TenantDep
from scholiast.application.ports.discovery import SourceUnavailable
from scholiast.application.use_cases.discover_papers import (
    NoSourceAnswered,
    SourceOutcome,
    SourceStatus,
)
from scholiast.domain.ranking import RANKER_VERSION, RankedWork, SearchFilters, SortOrder
from scholiast.infrastructure.observability import metrics

router = APIRouter(prefix="/v1/discovery", tags=["discovery"])
log = structlog.get_logger(__name__)


def _work(ranked: RankedWork) -> schemas.WorkOut:
    work, parts = ranked.work, ranked.parts
    imputed: list[Literal["citation_impact", "recency"]] = []
    if parts.citation_imputed:
        imputed.append("citation_impact")
    if parts.recency_imputed:
        imputed.append("recency")
    return schemas.WorkOut(
        score=ranked.score,
        score_parts=schemas.ScorePartsOut(
            relevance=parts.relevance,
            citation_impact=parts.citation,
            recency=parts.recency,
            imputed=imputed,
        ),
        members=[
            schemas.WorkMemberOut(source=m.source, source_id=m.source_id, source_rank=m.rank)
            for m in work.members
        ],
        add=schemas.PaperRefOut(source=work.add.source, source_id=work.add.source_id),
        title=work.title,
        authors=list(work.authors),
        year=work.year,
        published=work.published,
        venue=work.venue,
        abstract=work.abstract,
        citation_count=work.citation_count,
        doi=work.doi,
        arxiv_id=work.arxiv_id,
        open_access=[
            schemas.OpenAccessLinkOut(url=link.url, kind=link.kind.value, licence=link.licence)
            for link in work.oa_links
        ],
    )


def _source_status(outcome: SourceOutcome) -> schemas.SourceStatusOut:
    return schemas.SourceStatusOut(
        source=outcome.source,
        status=outcome.status.value,
        results=len(outcome.candidates),
        total_hits=outcome.total_hits,
        retry_after_seconds=outcome.retry_after,
        cached=outcome.cached,
        age_seconds=outcome.age_seconds,
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


# The user is a parameter, not a side effect of the limiter's: the cache is per user
# (D127), so the search needs it, and a route whose authentication depended on its
# rate limit would lose both to one edit. FastAPI resolves it once per request.
@router.post("/search", response_model=schemas.DiscoveryResults, dependencies=[DiscoveryRateLimit])
async def search(
    body: schemas.DiscoverySearch, discover: DiscoverDep, tenant: TenantDep
) -> schemas.DiscoveryResults:
    filters = SearchFilters(
        year_from=body.year_from,
        year_to=body.year_to,
        min_citations=body.min_citations,
        venue=body.venue,
        sort=SortOrder(body.sort),
    )
    try:
        result = await discover.execute(body.query, scope=tenant.user_id, filters=filters)
    except NoSourceAnswered as exc:
        _observe(exc.sources)
        raise
    _observe(result.sources)
    return schemas.DiscoveryResults(
        results=[_work(r) for r in result.ranked],
        source_status=[_source_status(o) for o in result.sources],
        partial=result.partial,
        ordering=body.sort,
        ranking=schemas.RankingOut(
            version=RANKER_VERSION,
            relevance=result.relevance_signal.value,
            weights=schemas.RankingWeightsOut(
                relevance=result.weights.relevance,
                citation_impact=result.weights.citation,
                recency=result.weights.recency,
            ),
            year=result.ranking_year,
        ),
        excluded=schemas.ExcludedOut(
            **result.excluded, unknown_citations=result.excluded_unknown_citations
        ),
    )
