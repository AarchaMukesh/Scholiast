"""The sources' breakers, arXiv's pacer, and how the wiring stacks them (D104, D122, D123).

The claims worth testing are about calls NOT made and slots NOT spent: an open
breaker that asks anyway, a pacing refusal the breaker holds against arXiv, a search
that waits for arXiv's slot, a gap measured from the wrong end, a probe the deadline
cut off that wedges the breaker. Clocks are injected, so no test sleeps through a
window to check it.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from scholiast.application.ports.discovery import (
    AcademicSource,
    SourceFailure,
    SourcePage,
    SourceUnavailable,
)
from scholiast.config.settings import Settings
from scholiast.infrastructure.circuit import BreakerState
from scholiast.infrastructure.discovery.breaker import SourceBreaker
from scholiast.infrastructure.discovery.pacing import PacedSource
from scholiast.infrastructure.discovery.sources import build_discovery_sources


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class Scripted:
    """A source that does what it is told next, and counts the calls that reached it."""

    name = "scripted"

    def __init__(self) -> None:
        self.calls = 0
        self.raises: BaseException | None = None
        self.hang = False
        self.release = asyncio.Event()
        self.on_call: object = None

    async def search(self, query: str, *, limit: int) -> SourcePage:
        self.calls += 1
        if callable(self.on_call):
            self.on_call()
        if self.hang:
            await self.release.wait()
        if self.raises is not None:
            raise self.raises
        return SourcePage(self.name, ())


def failure(kind: SourceFailure = SourceFailure.ERROR, **kw: object) -> SourceUnavailable:
    return SourceUnavailable("scripted", kind, "down", **kw)  # type: ignore[arg-type]


def breaker(
    threshold: int = 3, cooldown: float = 30.0
) -> tuple[SourceBreaker, Scripted, FakeClock]:
    inner, clock = Scripted(), FakeClock()
    return (
        SourceBreaker(inner, failure_threshold=threshold, cooldown_seconds=cooldown, clock=clock),
        inner,
        clock,
    )


async def fail(guard: SourceBreaker, times: int) -> None:
    for _ in range(times):
        with pytest.raises(SourceUnavailable):
            await guard.search("q", limit=5)


class TestTheSourceBreaker:
    async def test_it_satisfies_the_port_and_keeps_the_sources_name(self) -> None:
        guard, _, _ = breaker()
        assert isinstance(guard, AcademicSource)
        assert guard.name == "scripted"

    async def test_after_the_threshold_it_asks_nothing(self) -> None:
        # Not "does it raise": "does it raise WITHOUT asking". The count decides.
        guard, inner, _ = breaker(threshold=3)
        inner.raises = failure()
        await fail(guard, 3)
        assert guard.state is BreakerState.OPEN

        with pytest.raises(SourceUnavailable) as caught:
            await guard.search("q", limit=5)

        assert inner.calls == 3, "the breaker asked a source it had given up on"
        assert caught.value.kind is SourceFailure.CIRCUIT_OPEN
        assert caught.value.sent is False
        assert caught.value.retry_after == pytest.approx(30.0)

    async def test_a_stated_retry_after_opens_it_at_once_for_exactly_that_long(self) -> None:
        guard, inner, clock = breaker(threshold=3, cooldown=30.0)
        inner.raises = failure(SourceFailure.RATE_LIMITED, retry_after=7.0)
        await fail(guard, 1)

        clock.advance(6.9)
        assert guard.state is BreakerState.OPEN
        clock.advance(0.2)
        assert guard.state is BreakerState.HALF_OPEN

    async def test_our_own_pacing_is_never_held_against_the_source(self) -> None:
        # D123: a burst of searches is refused by OUR pacer; arXiv did nothing.
        guard, inner, _ = breaker(threshold=1)
        inner.raises = failure(SourceFailure.RATE_LIMITED, retry_after=2.0, sent=False)
        await fail(guard, 10)

        assert guard.state is BreakerState.CLOSED
        assert guard.failures == 0

    async def test_a_query_the_source_cannot_express_is_not_a_failure(self) -> None:
        # Otherwise three stopword-only queries would open arXiv's breaker for everyone.
        guard, inner, _ = breaker(threshold=1)
        inner.raises = ValueError("only stopwords")
        for _ in range(5):
            with pytest.raises(ValueError):
                await guard.search("the of and", limit=5)

        assert guard.state is BreakerState.CLOSED

    async def test_the_deadline_cutting_a_call_off_counts_as_a_failure(self) -> None:
        # D122: a source that cannot answer inside the budget is down for this purpose.
        guard, inner, _ = breaker(threshold=3)
        inner.hang = True
        for _ in range(3):
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(guard.search("q", limit=5), timeout=0.01)

        assert guard.state is BreakerState.OPEN

    async def test_a_crash_counts_as_a_failure(self) -> None:
        guard, inner, _ = breaker(threshold=2)
        inner.raises = KeyError("results")
        for _ in range(2):
            with pytest.raises(KeyError):
                await guard.search("q", limit=5)

        assert guard.state is BreakerState.OPEN

    async def test_a_probe_the_deadline_cuts_off_reopens_rather_than_wedging(self) -> None:
        guard, inner, clock = breaker(threshold=1, cooldown=30.0)
        inner.raises = failure()
        await fail(guard, 1)
        clock.advance(31.0)

        inner.raises, inner.hang = None, True
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(guard.search("q", limit=5), timeout=0.01)
        assert guard.state is BreakerState.OPEN, "a probe that did not answer must re-open"

        clock.advance(31.0)
        inner.hang = False
        await guard.search("q", limit=5)
        assert guard.state is BreakerState.CLOSED, "the breaker wedged on an abandoned probe"

    async def test_a_probe_our_pacer_refused_frees_the_probe_slot(self) -> None:
        # Nothing reached the source, so there is no verdict; the next call probes.
        guard, inner, clock = breaker(threshold=1, cooldown=30.0)
        inner.raises = failure()
        await fail(guard, 1)
        clock.advance(31.0)

        inner.raises = failure(SourceFailure.RATE_LIMITED, sent=False)
        await fail(guard, 1)
        inner.raises = None
        await guard.search("q", limit=5)

        assert guard.state is BreakerState.CLOSED


def paced(gap: float = 3.0) -> tuple[PacedSource, Scripted, FakeClock]:
    inner, clock = Scripted(), FakeClock()
    return PacedSource(inner, min_gap_seconds=gap, clock=clock), inner, clock


class TestArxivsPacer:
    async def test_a_second_search_while_one_is_in_flight_is_refused_not_queued(self) -> None:
        source, inner, _ = paced()
        inner.hang = True
        first = asyncio.create_task(source.search("q", limit=5))
        await asyncio.sleep(0)  # the first is now in flight

        # Bounded, so a pacer that let it through fails here instead of hanging on
        # a request that never answers.
        with pytest.raises(SourceUnavailable) as caught:
            await asyncio.wait_for(source.search("q", limit=5), timeout=1.0)

        assert inner.calls == 1, "two requests reached arXiv at once"
        assert caught.value.kind is SourceFailure.RATE_LIMITED
        assert caught.value.sent is False
        assert caught.value.retry_after == pytest.approx(3.0)
        inner.release.set()
        await first

    async def test_the_gap_runs_from_the_end_of_the_previous_request(self) -> None:
        # arXiv's terms as the spike read them: three seconds after the previous
        # response ENDED. Measured from the start, a 10 s request would leave no gap.
        source, inner, clock = paced(gap=3.0)
        inner.on_call = lambda: clock.advance(10.0)  # the request takes 10 s
        await source.search("q", limit=5)  # starts at 1000, ends at 1010
        inner.on_call = None

        clock.advance(2.9)
        with pytest.raises(SourceUnavailable) as caught:
            await source.search("q", limit=5)
        assert caught.value.retry_after == pytest.approx(0.1)

        clock.advance(0.1)
        await source.search("q", limit=5)
        assert inner.calls == 2

    async def test_a_search_never_waits_for_the_slot(self) -> None:
        # D123 as amended: waiting would make arXiv's pace the user's latency.
        source, _, _ = paced()
        await source.search("q", limit=5)

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(SourceUnavailable):
            await source.search("q", limit=5)

        assert loop.time() - started < 0.05

    async def test_a_query_arxiv_cannot_express_uses_no_slot(self) -> None:
        source, inner, _ = paced()
        inner.raises = ValueError("only stopwords")
        with pytest.raises(ValueError):
            await source.search("the of", limit=5)

        inner.raises = None
        await source.search("graph networks", limit=5)  # not refused: nothing was sent
        assert inner.calls == 2

    async def test_a_failed_request_still_used_its_slot(self) -> None:
        # A 429 or a 500 was a request arXiv received; the next one still waits.
        source, inner, _ = paced()
        inner.raises = failure(SourceFailure.RATE_LIMITED)
        with pytest.raises(SourceUnavailable):
            await source.search("q", limit=5)

        inner.raises = None
        with pytest.raises(SourceUnavailable) as caught:
            await source.search("q", limit=5)
        assert caught.value.sent is False

    async def test_a_request_cut_off_counts_as_ended_when_it_was_cut(self) -> None:
        source, inner, clock = paced(gap=3.0)
        inner.hang = True
        task = asyncio.create_task(source.search("q", limit=5))
        await asyncio.sleep(0)
        clock.advance(5.0)  # cut off 5 s in
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        inner.hang = False
        clock.advance(2.9)
        with pytest.raises(SourceUnavailable):
            await source.search("q", limit=5)
        clock.advance(0.1)
        await source.search("q", limit=5)


class TestTheWiring:
    def test_openalex_is_absent_without_a_key_not_broken(self) -> None:
        # A client around an empty key would turn every search into a 401 that the
        # fan-out reads as an outage.
        sources = build_discovery_sources(Settings(openalex_api_key=None))
        assert [c.source.name for c in sources] == ["arxiv"]

    def test_openalex_leads_with_its_measured_page_size(self) -> None:
        sources = build_discovery_sources(Settings(openalex_api_key="k"))
        assert [(c.source.name, c.page_size) for c in sources] == [("openalex", 25), ("arxiv", 50)]

    def test_each_source_names_its_adapter_build_for_the_cache_key(self) -> None:
        # Without it, a page parsed by an older build would be served as this one's
        # for up to a day after a parser change (D127, NFR-M5).
        from scholiast.infrastructure.discovery import arxiv, openalex

        sources = build_discovery_sources(Settings(openalex_api_key="k"))
        assert [c.version for c in sources] == [openalex.ADAPTER_VERSION, arxiv.ADAPTER_VERSION]

    def test_the_app_hands_its_cache_to_every_search(self) -> None:
        # A cache built and never passed on would leave every search uncached,
        # which looks exactly like a cache that is merely cold.
        from starlette.requests import Request

        from scholiast.api import deps
        from scholiast.api.app import create_app
        from scholiast.infrastructure.discovery.cache import RedisDiscoveryCache

        app = create_app(Settings())
        search = deps.discover_use_case(Request({"type": "http", "app": app}))

        assert isinstance(app.state.discovery_cache, RedisDiscoveryCache)
        assert search._cache is app.state.discovery_cache

    def test_each_source_has_its_own_breaker(self) -> None:
        # One shared breaker would let a rate-limited OpenAlex stop the fan-out
        # asking a healthy arXiv.
        sources = build_discovery_sources(Settings(openalex_api_key="k"))
        assert all(isinstance(c.source, SourceBreaker) for c in sources)
        assert sources[0].source is not sources[1].source

    def test_arxivs_pacer_sits_inside_its_breaker(self) -> None:
        """Structure, asserted directly, because the behaviour takes 9 s of real pacing.

        Inside, an open breaker refuses before the pacer is consulted and spends no
        slot. Outside, an open breaker would still claim a slot, and arXiv would be
        skipped for three seconds after a refusal that sent nothing.
        """
        arxiv = build_discovery_sources(Settings(openalex_api_key=None))[0].source
        assert isinstance(arxiv, SourceBreaker)
        assert isinstance(arxiv._inner, PacedSource)

    def test_arxivs_pace_cannot_be_configured_below_its_terms(self) -> None:
        with pytest.raises(ValidationError):
            Settings(discovery_arxiv_min_gap_seconds=1.0)
