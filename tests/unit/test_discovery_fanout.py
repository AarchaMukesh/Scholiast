"""The fan-out: every source at once, one deadline, a status each (FR-D6, D122).

What fails silently here is the search answering when it should not, or the wrong
way: a slow source holding the others hostage, a failure read as "no papers", one
adapter's bug failing everybody's search, a failure kind reported as another. The
first half drives the use case with fake sources, so each rule is pinned on its own;
the second half drives the REAL adapters through the real wiring (breakers and
arXiv's pacer included) under injected HTTP faults, so the rules hold end to end.

No network: `respx` refuses any request it was not told about (NFR-T3).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
import respx

from scholiast.application.ports.discovery import SourceFailure, SourcePage, SourceUnavailable
from scholiast.application.use_cases.discover_papers import (
    ConfiguredSource,
    DiscoverPapersUseCase,
    NoSourceAnswered,
    SourceOutcome,
    SourceStatus,
)
from scholiast.config.settings import Settings
from scholiast.domain.discovery import PaperCandidate, UnsearchableQuery
from scholiast.infrastructure.discovery import openalex as openalex_adapter
from scholiast.infrastructure.discovery.sources import build_discovery_sources

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "discovery"
OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_QUERY = "https://export.arxiv.org/api/query"
KEY = "test-key-0123456789"


def candidate(source: str, rank: int) -> PaperCandidate:
    return PaperCandidate(
        source=source,
        source_id=f"{source}-{rank}",
        rank=rank,
        title=f"{source} paper {rank}",
        adapter_version="fake-1",
    )


class FakeSource:
    """Answers, fails, crashes or dawdles on command, and records each page size asked for."""

    def __init__(
        self,
        name: str,
        *,
        results: int = 3,
        delay: float = 0.0,
        raises: BaseException | None = None,
    ) -> None:
        self._name = name
        self.results = results
        self.delay = delay
        self.raises = raises
        self.calls: list[int] = []

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, *, limit: int) -> SourcePage:
        self.calls.append(limit)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        page = tuple(candidate(self._name, r) for r in range(1, self.results + 1))
        return SourcePage(self._name, page, total_hits=1000)


def use_case(*sources: FakeSource, deadline: float = 2.0) -> DiscoverPapersUseCase:
    return DiscoverPapersUseCase(
        [ConfiguredSource(s, 25) for s in sources], deadline_seconds=deadline
    )


def status_of(outcomes: tuple[SourceOutcome, ...], name: str) -> SourceOutcome:
    return next(o for o in outcomes if o.source == name)


class TestEverySourceAnswering:
    async def test_results_interleave_each_sources_own_order(self) -> None:
        # D122's order until phase 8 ranks: favours no source, claims no relevance.
        result = await use_case(FakeSource("a", results=3), FakeSource("b", results=2)).execute(
            "graph neural networks"
        )

        assert [c.source_id for c in result.candidates] == ["a-1", "b-1", "a-2", "b-2", "a-3"]
        assert result.partial is False
        assert [o.status for o in result.sources] == [SourceStatus.OK, SourceStatus.OK]
        assert {o.total_hits for o in result.sources} == {1000}

    async def test_an_empty_page_is_an_answer_not_a_failure(self) -> None:
        # The other half of "a failure is not no papers": no papers IS an answer,
        # and must not become a 503 or a partial flag.
        result = await use_case(FakeSource("a", results=0), FakeSource("b", results=0)).execute(
            "an obscure topic"
        )

        assert result.candidates == ()
        assert result.partial is False

    async def test_each_source_is_asked_for_its_own_page_size(self) -> None:
        # D125 measured OpenAlex at 25; arXiv stays at 50. One page size for all
        # would undo the measurement for one of them.
        a, b = FakeSource("a"), FakeSource("b")
        await DiscoverPapersUseCase([ConfiguredSource(a, 25), ConfiguredSource(b, 50)]).execute("x")

        assert (a.calls, b.calls) == ([25], [50])

    async def test_sources_are_asked_together_not_one_after_another(self) -> None:
        # Sequential calls would make the wait the SUM of the sources' latencies,
        # and the deadline a budget only the first source could use.
        a, b = FakeSource("a", delay=0.3), FakeSource("b", delay=0.3)

        started = time.perf_counter()
        await use_case(a, b).execute("x")

        assert time.perf_counter() - started < 0.5


class TestTheDeadline:
    async def test_a_slow_source_is_cut_off_and_the_others_answer(self) -> None:
        started = time.perf_counter()
        result = await use_case(
            FakeSource("slow", delay=30.0), FakeSource("fast"), deadline=0.2
        ).execute("x")
        elapsed = time.perf_counter() - started

        assert elapsed < 0.6, f"the search waited {elapsed:.2f}s for a source past its deadline"
        assert status_of(result.sources, "slow").status is SourceStatus.TIMEOUT
        assert status_of(result.sources, "fast").status is SourceStatus.OK
        assert [c.source for c in result.candidates] == ["fast"] * 3
        assert result.partial is True

    async def test_every_source_is_cut_off_at_the_same_moment(self) -> None:
        # One deadline for the fan-out, not one per source: two slow sources cost
        # one deadline, not two.
        started = time.perf_counter()
        result = await use_case(
            FakeSource("s1", delay=30.0),
            FakeSource("s2", delay=30.0),
            FakeSource("ok"),
            deadline=0.2,
        ).execute("x")

        assert time.perf_counter() - started < 0.6
        assert [o.status for o in result.sources] == [
            SourceStatus.TIMEOUT,
            SourceStatus.TIMEOUT,
            SourceStatus.OK,
        ]


class TestEachFailureIsItsSourcesAlone:
    @pytest.mark.parametrize("kind", list(SourceFailure))
    async def test_each_failure_kind_becomes_that_sources_status(self, kind: SourceFailure) -> None:
        failing = FakeSource("down", raises=SourceUnavailable("down", kind, "x", retry_after=7.0))
        result = await use_case(failing, FakeSource("up")).execute("x")

        down = status_of(result.sources, "down")
        assert down.status.value == kind.value
        assert down.retry_after == 7.0
        assert [c.source for c in result.candidates] == ["up"] * 3
        assert result.partial is True

    async def test_an_unexpected_exception_costs_only_its_source(self) -> None:
        # Not a `SourceUnavailable`: a bug in one adapter. FR-D6's independence
        # has to cover what nobody anticipated, not only what the port types.
        crash = KeyError("display_name")
        result = await use_case(FakeSource("buggy", raises=crash), FakeSource("up")).execute("x")

        buggy = status_of(result.sources, "buggy")
        assert buggy.status is SourceStatus.ERROR
        assert buggy.error is crash, "the caller cannot log a crash it is not given"
        assert [c.source for c in result.candidates] == ["up"] * 3

    async def test_no_source_answering_raises_rather_than_returning_no_papers(self) -> None:
        down = SourceUnavailable("a", SourceFailure.ERROR, "x")
        with pytest.raises(NoSourceAnswered) as caught:
            await use_case(
                FakeSource("a", raises=down), FakeSource("b", delay=30.0), deadline=0.1
            ).execute("x")

        assert [o.status for o in caught.value.sources] == [
            SourceStatus.ERROR,
            SourceStatus.TIMEOUT,
        ]

    async def test_a_deployment_with_no_sources_says_so_rather_than_finding_nothing(self) -> None:
        with pytest.raises(NoSourceAnswered):
            await use_case().execute("x")


class TestAQueryWithNothingToSearch:
    async def test_punctuation_is_refused_before_any_source_is_asked(self) -> None:
        a = FakeSource("a")
        with pytest.raises(UnsearchableQuery):
            await use_case(a).execute(" ?!? ... ")

        assert a.calls == []

    async def test_every_source_refusing_the_query_is_the_callers_mistake_not_an_outage(
        self,
    ) -> None:
        # The port's rule: ValueError means the query, not the source. Reporting a
        # 503 here would tell the user the sources are down.
        refuse = ValueError("nothing searchable")
        with pytest.raises(UnsearchableQuery):
            await use_case(FakeSource("a", raises=refuse), FakeSource("b", raises=refuse)).execute(
                "the of and"
            )

    async def test_one_source_refusing_the_query_does_not_fail_the_search(self) -> None:
        # arXiv cannot search a query of only its stopwords; OpenAlex can.
        result = await use_case(
            FakeSource("arxiv", raises=ValueError("only stopwords")), FakeSource("openalex")
        ).execute("what is the")

        assert status_of(result.sources, "arxiv").status is SourceStatus.ERROR
        assert result.partial is True
        assert len(result.candidates) == 3


# ---------------------------------------------------------------------------
# The real adapters, through the real wiring, under injected faults.
# ---------------------------------------------------------------------------


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def real_use_case(*, deadline: float = 2.0) -> DiscoverPapersUseCase:
    """Built exactly as the app builds it: breakers, arXiv's pacer, page sizes."""
    settings = Settings(openalex_api_key=KEY, discovery_deadline_seconds=deadline)
    return DiscoverPapersUseCase(build_discovery_sources(settings), deadline_seconds=deadline)


def healthy_openalex() -> respx.Route:
    return respx.get(OPENALEX_WORKS).mock(
        return_value=httpx.Response(200, text=fixture("openalex_search.json"))
    )


def healthy_arxiv() -> respx.Route:
    return respx.get(ARXIV_QUERY).mock(
        return_value=httpx.Response(200, text=fixture("arxiv_search.xml"))
    )


#: (label, how the route fails, the status it must become, the retry_after it carries)
OPENALEX_FAULTS = [
    ("a server error", {"return_value": httpx.Response(500, text="oops")}, "error", None),
    (
        "a 429 with Retry-After",
        {"return_value": httpx.Response(429, headers={"retry-after": "7"})},
        "rate_limited",
        7.0,
    ),
    (
        "its own wildcard 400",
        {"return_value": httpx.Response(400, text=fixture("openalex_400_wildcard.json"))},
        "error",
        None,
    ),
    ("a read timeout", {"side_effect": httpx.ReadTimeout("slow")}, "timeout", None),
    ("a refused connection", {"side_effect": httpx.ConnectError("refused")}, "error", None),
    (
        "a 200 that is not JSON",
        {"return_value": httpx.Response(200, text="<html/>")},
        "error",
        None,
    ),
]

ARXIV_FAULTS = [
    ("a server error", {"return_value": httpx.Response(500, text="oops")}, "error", None),
    (
        "its 503 throttle",
        {"return_value": httpx.Response(503, headers={"retry-after": "15"})},
        "rate_limited",
        15.0,
    ),
    (
        "an error feed",
        {"return_value": httpx.Response(200, text=fixture("arxiv_error.xml"))},
        "error",
        None,
    ),
    ("a read timeout", {"side_effect": httpx.ReadTimeout("slow")}, "timeout", None),
    ("a refused connection", {"side_effect": httpx.ConnectError("refused")}, "error", None),
]


class TestTheRealAdaptersUnderFaultInjection:
    @respx.mock
    async def test_both_healthy_is_a_full_interleaved_answer_openalex_first(self) -> None:
        healthy_openalex()
        healthy_arxiv()

        result = await real_use_case().execute("dense passage retrieval")

        assert result.partial is False
        sources = [c.source for c in result.candidates]
        assert sources[:2] == ["openalex", "arxiv"], "OpenAlex is the primary, so it leads (D118)"
        assert sources.count("openalex") == 3
        assert sources.count("arxiv") == 3

    @pytest.mark.parametrize(
        ("label", "fault", "status", "retry_after"),
        OPENALEX_FAULTS,
        ids=[f[0] for f in OPENALEX_FAULTS],
    )
    @respx.mock
    async def test_openalex_failing_leaves_arxiv_intact(
        self, label: str, fault: dict[str, object], status: str, retry_after: float | None
    ) -> None:
        respx.get(OPENALEX_WORKS).mock(**fault)
        healthy_arxiv()

        result = await real_use_case().execute("dense passage retrieval")

        openalex = status_of(result.sources, "openalex")
        assert (openalex.status.value, openalex.retry_after) == (status, retry_after)
        assert [c.source for c in result.candidates] == ["arxiv"] * 3
        assert result.partial is True

    @pytest.mark.parametrize(
        ("label", "fault", "status", "retry_after"),
        ARXIV_FAULTS,
        ids=[f[0] for f in ARXIV_FAULTS],
    )
    @respx.mock
    async def test_arxiv_failing_leaves_openalex_intact(
        self, label: str, fault: dict[str, object], status: str, retry_after: float | None
    ) -> None:
        healthy_openalex()
        respx.get(ARXIV_QUERY).mock(**fault)

        result = await real_use_case().execute("dense passage retrieval")

        arxiv = status_of(result.sources, "arxiv")
        assert (arxiv.status.value, arxiv.retry_after) == (status, retry_after)
        assert [c.source for c in result.candidates] == ["openalex"] * 3
        assert result.partial is True

    @respx.mock
    async def test_a_crash_inside_one_adapter_costs_only_that_source(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def crash(payload: object) -> SourcePage:
            raise KeyError("results")

        monkeypatch.setattr(openalex_adapter, "parse_works", crash)
        healthy_openalex()
        healthy_arxiv()

        result = await real_use_case().execute("dense passage retrieval")

        assert status_of(result.sources, "openalex").status is SourceStatus.ERROR
        assert [c.source for c in result.candidates] == ["arxiv"] * 3

    @respx.mock
    async def test_a_source_slower_than_the_deadline_is_cut_off_there(self) -> None:
        async def dawdle(request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(10.0)
            return httpx.Response(200, text=fixture("openalex_search.json"))

        respx.get(OPENALEX_WORKS).mock(side_effect=dawdle)
        healthy_arxiv()

        started = time.perf_counter()
        result = await real_use_case(deadline=0.3).execute("dense passage retrieval")

        assert time.perf_counter() - started < 1.0
        assert status_of(result.sources, "openalex").status is SourceStatus.TIMEOUT
        assert [c.source for c in result.candidates] == ["arxiv"] * 3

    @respx.mock
    async def test_both_failing_is_no_answer_with_both_statuses(self) -> None:
        respx.get(OPENALEX_WORKS).mock(return_value=httpx.Response(500))
        respx.get(ARXIV_QUERY).mock(side_effect=httpx.ConnectError("refused"))

        with pytest.raises(NoSourceAnswered) as caught:
            await real_use_case().execute("dense passage retrieval")

        assert {o.source: o.status for o in caught.value.sources} == {
            "openalex": SourceStatus.ERROR,
            "arxiv": SourceStatus.ERROR,
        }
