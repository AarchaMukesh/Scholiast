"""Topic in, candidates out: every source asked at once, under one deadline (FR-D1, FR-D6, D122).

**One deadline for the whole fan-out.** The user waits for the fan-out, not for any
one source, so what matters is when it stops waiting. Every source starts together
and every one is cut off at the same moment, 2.0 s by default: D122 gives the
sources 2.0 of NFR-P1's 3.0 s and keeps 1.0 s for phases 7 and 8. A source still
working at the deadline is reported `timeout`; the others' results are not held
hostage to it.

**Each source's failure is its own** (FR-D6). A source that raises anything at all,
not only `SourceUnavailable`, costs that source and not the search: a KeyError on
one malformed record is a bug in one adapter, and independence has to cover the
failures nobody anticipated as well as the ones the port types. The exception rides
on the outcome so the caller can log it; it never reaches the client, because a
third party's error text is not ours to show.

**No source answered is not "no papers".** If every source failed, the search
raises `NoSourceAnswered` and the API says 503 (D119's rule, lifted to the
endpoint). An empty result list means the sources searched and found nothing.

**Each source's page is cached per user for a day** (D127). The cache is looked up
inside the deadline, before the source is asked; a hit answers for that source and
nothing is sent. Only answers are stored, never failures, so the next search asks
again any source that failed. The cache is isolated like a source: if it raises,
the search carries on as a miss.

**The order is the sources' own, interleaved** (D122): each source's first result,
then each one's second. Duplicates across sources stay until phase 7 merges them,
and phase 8 replaces the order with a ranking. Nothing here is written anywhere:
discovery results are not papers until they are added (§5.4.1).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum, unique

from scholiast.application.ports.discovery import (
    AcademicSource,
    CachedPage,
    CacheKey,
    DiscoveryCache,
    SourcePage,
    SourceUnavailable,
)
from scholiast.domain.discovery import (
    PaperCandidate,
    UnsearchableQuery,
    interleave,
    is_searchable,
    normalise_query,
)

#: D122: the sources' share of NFR-P1's 3.0 s.
DEFAULT_DEADLINE_SECONDS = 2.0


@unique
class SourceStatus(StrEnum):
    """What happened to one source in one search: a fixed set, so a UI can enumerate it."""

    OK = "ok"
    TIMEOUT = "timeout"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"


@dataclass(frozen=True, slots=True)
class ConfiguredSource:
    """A source and the page size to ask it for (D125: 25 for OpenAlex, 50 for arXiv)."""

    source: AcademicSource
    page_size: int
    #: The adapter build behind it (NFR-M5). Part of the cache key, so a page parsed
    #: by an older build is never served as this one's (D127).
    version: str = ""


@dataclass(frozen=True, slots=True)
class SourceOutcome:
    """One source's part in one search."""

    source: str
    status: SourceStatus
    candidates: tuple[PaperCandidate, ...] = ()
    #: How many works the source said matched, when it answered and said.
    total_hits: int | None = None
    #: When a refused source said, or our pacing knows, when it will accept work again.
    retry_after: float | None = None
    #: From the search's start to this source's outcome.
    seconds: float = 0.0
    #: Whether the page came from the cache rather than the source (D127).
    cached: bool = False
    #: How old a cached page is. None when the source was asked.
    age_seconds: float | None = None
    #: What went wrong, for the caller's log. Never sent to a client.
    error: BaseException | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    candidates: tuple[PaperCandidate, ...]
    sources: tuple[SourceOutcome, ...]

    @property
    def partial(self) -> bool:
        """FR-D6's flag: true whenever any source did not answer."""
        return any(outcome.status is not SourceStatus.OK for outcome in self.sources)


class NoSourceAnswered(Exception):
    """Every configured source failed, so there is no result to give, not an empty one."""

    def __init__(self, sources: tuple[SourceOutcome, ...]) -> None:
        super().__init__(
            "no discovery source answered: " + ", ".join(f"{o.source}={o.status}" for o in sources)
        )
        self.sources = sources


class DiscoverPapersUseCase:
    """Ask every configured source, give each the same deadline, report each one's fate."""

    def __init__(
        self,
        sources: Sequence[ConfiguredSource],
        *,
        cache: DiscoveryCache | None = None,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._sources = tuple(sources)
        self._cache = cache
        self._deadline = deadline_seconds
        self._clock = clock
        self._now = now

    async def execute(self, query: str, *, scope: str) -> DiscoveryResult:
        """Search every source for `query` on behalf of the user `scope` names."""
        if not is_searchable(query):
            raise UnsearchableQuery("the query has no letter or digit to search for")
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._deadline
        started = self._clock()
        outcomes = await asyncio.gather(
            *(
                self._ask(configured, query, scope, deadline, started)
                for configured in self._sources
            )
        )
        unsearchable = [o for o in outcomes if isinstance(o.error, ValueError)]
        if outcomes and len(unsearchable) == len(outcomes):
            # Every source said there is nothing in this query it can search for.
            # That is the caller's mistake, not an outage (the port's rule).
            raise UnsearchableQuery(str(unsearchable[0].error))
        if not any(o.status is SourceStatus.OK for o in outcomes):
            raise NoSourceAnswered(tuple(outcomes))
        return DiscoveryResult(
            candidates=interleave([o.candidates for o in outcomes]),
            sources=tuple(outcomes),
        )

    async def _ask(
        self,
        configured: ConfiguredSource,
        query: str,
        scope: str,
        deadline: float,
        started: float,
    ) -> SourceOutcome:
        key = CacheKey(
            scope=scope,
            source=configured.source.name,
            version=configured.version,
            page_size=configured.page_size,
            query=normalise_query(query),
        )
        page: SourcePage | None = None
        hit: CachedPage | None = None
        status = SourceStatus.OK
        retry_after: float | None = None
        error: BaseException | None = None
        try:
            async with asyncio.timeout_at(deadline):
                hit = await self._lookup(key)
                if hit is not None:
                    page = hit.page
                else:
                    page = await configured.source.search(query, limit=configured.page_size)
        except TimeoutError as exc:
            status, error = SourceStatus.TIMEOUT, exc
        except SourceUnavailable as exc:
            status, retry_after, error = SourceStatus(exc.kind.value), exc.retry_after, exc
        except Exception as exc:
            # One source's bug must not fail the search (FR-D6).
            status, error = SourceStatus.ERROR, exc
        if page is not None and hit is None:
            # After the deadline's block, not inside it: a page that arrived must not
            # be lost to a slow cache write. The cache bounds its own calls (D127).
            await self._store(key, page)
        return SourceOutcome(
            configured.source.name,
            status,
            candidates=page.candidates if page is not None else (),
            total_hits=page.total_hits if page is not None else None,
            retry_after=retry_after,
            seconds=self._clock() - started,
            error=error,
            cached=hit is not None,
            age_seconds=(
                max(0.0, (self._now() - hit.fetched_at).total_seconds())
                if hit is not None
                else None
            ),
        )

    async def _lookup(self, key: CacheKey) -> CachedPage | None:
        if self._cache is None:
            return None
        try:
            return await self._cache.get(key)
        except Exception:
            # The port forbids raising; a cache with a bug must still only miss (D127).
            return None

    async def _store(self, key: CacheKey, page: SourcePage) -> None:
        if self._cache is None:
            return
        try:
            await self._cache.put(key, page, self._now())
        except Exception:
            # A cache that cannot store costs the next search a miss, nothing more.
            return
