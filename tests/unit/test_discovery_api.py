"""`POST /v1/discovery/search` (FR-D1, FR-D5's groundwork, FR-D6, D122).

The endpoint's claims are about what a client is told: every source's status beside
a partial result, a 503 rather than an empty list when nobody answered, a 400 for a
query with nothing in it, and never a third party's error text or our OpenAlex key.
The sources are fakes, except where the claim is about a real adapter's output.
"""

from __future__ import annotations

from datetime import date
from http import HTTPStatus

import httpx
import pytest
import respx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from scholiast.api import deps
from scholiast.api.errors import PROBLEM_CONTENT_TYPE, register_error_handlers
from scholiast.api.routers import discovery
from scholiast.application.ports.discovery import SourceFailure, SourcePage, SourceUnavailable
from scholiast.application.ports.tenancy import TenantContext
from scholiast.application.use_cases.discover_papers import ConfiguredSource
from scholiast.config.settings import Settings
from scholiast.domain.discovery import LinkKind, OpenAccessLink, PaperCandidate
from scholiast.domain.ranking import RANKER_VERSION, SHIPPED_WEIGHTS
from scholiast.infrastructure.discovery.openalex import OpenAlexSource
from scholiast.infrastructure.observability.metrics import REGISTRY
from scholiast.infrastructure.security.ratelimit import FixedWindowLimiter

KEY = "test-key-0123456789"
SEARCH = "/v1/discovery/search"


def paper(source: str, rank: int, **fields: object) -> PaperCandidate:
    base: dict[str, object] = {
        "source": source,
        "source_id": f"{source}-{rank}",
        "rank": rank,
        "title": f"{source} paper {rank}",
        "adapter_version": "fake-1",
    }
    base.update(fields)
    return PaperCandidate(**base)  # type: ignore[arg-type]


class Fake:
    def __init__(
        self,
        name: str,
        *,
        papers: tuple[PaperCandidate, ...] | None = None,
        raises: BaseException | None = None,
    ) -> None:
        self._name = name
        self.papers = papers if papers is not None else (paper(name, 1), paper(name, 2))
        self.raises = raises
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    async def search(self, query: str, *, limit: int) -> SourcePage:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return SourcePage(self._name, self.papers, total_hits=1000)


def build(
    *sources: object, settings: Settings | None = None, authenticated: bool = True
) -> TestClient:
    app = FastAPI()
    register_error_handlers(app)
    app.include_router(discovery.router)
    app.state.settings = settings or Settings()
    app.state.rate_limiter = FixedWindowLimiter()
    app.state.discovery_sources = [ConfiguredSource(s, 25) for s in sources]  # type: ignore[arg-type]
    if authenticated:
        app.dependency_overrides[deps.current_tenant] = lambda: TenantContext(user_id="u-1")
    else:
        app.state.tokens = object()  # never reached: no credentials are presented
    return TestClient(app, raise_server_exceptions=False)


def down(
    kind: SourceFailure, retry_after: float | None = None, detail: str = "x"
) -> SourceUnavailable:
    return SourceUnavailable("s", kind, detail, retry_after=retry_after)


class TestAPartialResult:
    def test_every_source_is_named_with_its_status(self) -> None:
        client = build(Fake("openalex"), Fake("arxiv", raises=down(SourceFailure.TIMEOUT)))

        response = client.post(SEARCH, json={"query": "dense passage retrieval"})

        assert response.status_code == HTTPStatus.OK
        body = response.json()
        assert body["partial"] is True
        assert body["ordering"] == "score"
        assert body["source_status"] == [
            {
                "source": "openalex",
                "status": "ok",
                "results": 2,
                "total_hits": 1000,
                "retry_after_seconds": None,
                "cached": False,
                "age_seconds": None,
            },
            {
                "source": "arxiv",
                "status": "timeout",
                "results": 0,
                "total_hits": None,
                "retry_after_seconds": None,
                "cached": False,
                "age_seconds": None,
            },
        ]
        assert [r["add"]["source_id"] for r in body["results"]] == ["openalex-1", "openalex-2"]

    def test_every_source_answering_is_not_partial(self) -> None:
        body = build(Fake("openalex"), Fake("arxiv")).post(SEARCH, json={"query": "x y"}).json()

        assert body["partial"] is False
        assert [r["add"]["source_id"] for r in body["results"]] == [
            "openalex-1",
            "arxiv-1",
            "openalex-2",
            "arxiv-2",
        ]

    def test_a_result_carries_what_fr_d1_asks_for_and_unknown_is_null_not_zero(self) -> None:
        rich = paper(
            "arxiv",
            1,
            authors=("Vladimir Karpukhin", "Barlas Oguz"),
            year=2020,
            published=date(2020, 4, 10),
            venue="EMNLP",
            abstract="Open-domain question answering relies on efficient passage retrieval.",
            citation_count=None,
            doi="10.18653/v1/2020.emnlp-main.550",
            arxiv_id="2004.04906",
            oa_links=(OpenAccessLink("https://arxiv.org/pdf/2004.04906v3", LinkKind.PDF),),
        )

        result = build(Fake("arxiv", papers=(rich,))).post(SEARCH, json={"query": "dpr"}).json()

        [work] = result["results"]
        score, parts = work.pop("score"), work.pop("score_parts")
        age = result["ranking"]["year"] - 2020
        # arXiv reports no citation count: the part is imputed, never zero (D152).
        assert parts["imputed"] == ["citation_impact"]
        assert parts["recency"] == pytest.approx(0.5 ** (age / 3))
        weights = result["ranking"]["weights"]
        assert score == pytest.approx(
            weights["relevance"] * parts["relevance"]
            + weights["citation_impact"] * parts["citation_impact"]
            + weights["recency"] * parts["recency"]
        ), "the score is not the weighted sum of the parts it shows"
        assert result["results"] == [
            {
                "members": [{"source": "arxiv", "source_id": "arxiv-1", "source_rank": 1}],
                "add": {"source": "arxiv", "source_id": "arxiv-1"},
                "title": "arxiv paper 1",
                "authors": ["Vladimir Karpukhin", "Barlas Oguz"],
                "year": 2020,
                "published": "2020-04-10",
                "venue": "EMNLP",
                "abstract": "Open-domain question answering relies on efficient passage retrieval.",
                "citation_count": None,
                "doi": "10.18653/v1/2020.emnlp-main.550",
                "arxiv_id": "2004.04906",
                "open_access": [
                    {"url": "https://arxiv.org/pdf/2004.04906v3", "kind": "pdf", "licence": None}
                ],
            }
        ]


class TestOnePaperOneEntry:
    """FR-D3 at the endpoint: one entry per paper, naming every record of it (D147)."""

    @staticmethod
    def dpr(source: str, **fields: object) -> PaperCandidate:
        return paper(
            source,
            1,
            title="Dense Passage Retrieval for Open-Domain Question Answering",
            authors=("Vladimir Karpukhin", "Barlas Oguz"),
            **fields,
        )

    def test_a_paper_both_sources_return_is_one_entry_naming_both(self) -> None:
        published = self.dpr(
            "openalex",
            year=2020,
            doi="10.18653/v1/2020.emnlp-main.550",
            citation_count=4100,
            oa_links=(
                OpenAccessLink("https://aclanthology.org/2020.emnlp-main.550", LinkKind.LANDING),
            ),
        )
        preprint = self.dpr(
            "arxiv",
            year=2020,
            arxiv_id="2004.04906",
            oa_links=(OpenAccessLink("https://arxiv.org/pdf/2004.04906v3", LinkKind.PDF),),
        )

        body = (
            build(Fake("openalex", papers=(published,)), Fake("arxiv", papers=(preprint,)))
            .post(SEARCH, json={"query": "dpr"})
            .json()
        )

        [work] = body["results"]
        assert work["members"] == [
            {"source": "openalex", "source_id": "openalex-1", "source_rank": 1},
            {"source": "arxiv", "source_id": "arxiv-1", "source_rank": 1},
        ]
        # The record add can fetch, not merely the first one shown.
        assert work["add"] == {"source": "arxiv", "source_id": "arxiv-1"}
        assert (work["doi"], work["arxiv_id"], work["citation_count"]) == (
            "10.18653/v1/2020.emnlp-main.550",
            "2004.04906",
            4100,
        )
        assert [link["url"] for link in work["open_access"]] == [
            "https://aclanthology.org/2020.emnlp-main.550",
            "https://arxiv.org/pdf/2004.04906v3",
        ]

    def test_the_offered_record_is_one_the_configured_allowlist_lets_add_fetch(self) -> None:
        # Search must judge "fetchable" by the allowlist add enforces, or it could
        # offer a record that add then refuses (D147).
        on_acl = self.dpr(
            "openalex",
            oa_links=(
                OpenAccessLink("https://aclanthology.org/2020.emnlp-main.550.pdf", LinkKind.PDF),
            ),
        )
        on_arxiv = self.dpr(
            "arxiv",
            arxiv_id="2004.04906",
            oa_links=(OpenAccessLink("https://arxiv.org/pdf/2004.04906v3", LinkKind.PDF),),
        )
        only_acl = Settings(fetch_allowed_hosts=["aclanthology.org"])

        body = (
            build(
                Fake("arxiv", papers=(on_arxiv,)),
                Fake("openalex", papers=(on_acl,)),
                settings=only_acl,
            )
            .post(SEARCH, json={"query": "dpr"})
            .json()
        )

        [work] = body["results"]
        assert work["add"] == {"source": "openalex", "source_id": "openalex-1"}


class TestRankingAndFilters:
    """FR-D4, FR-D5 and FR-D8 at the endpoint (D151-D153)."""

    @staticmethod
    def papers() -> tuple[PaperCandidate, ...]:
        return (
            paper(
                "openalex",
                1,
                title="Graph neural networks",
                year=2019,
                citation_count=900,
                venue="NeurIPS",
            ),
            paper(
                "openalex",
                2,
                title="Dense passage retrieval for question answering",
                year=2023,
                citation_count=40,
                venue="EMNLP",
            ),
            paper(
                "openalex",
                3,
                title="Retrieval for question answering",
                year=2024,
                citation_count=None,
                venue="ACL Findings",
            ),
        )

    def search(self, **body: object) -> dict[str, object]:
        response = build(Fake("openalex", papers=self.papers())).post(
            SEARCH, json={"query": "dense passage retrieval", **body}
        )
        assert response.status_code == HTTPStatus.OK, response.text
        return dict(response.json())

    @staticmethod
    def ids(body: dict[str, object]) -> list[str]:
        return [r["add"]["source_id"] for r in body["results"]]  # type: ignore[index, union-attr]

    def test_results_are_ordered_by_score_not_by_the_sources_rank(self) -> None:
        body = self.search()

        assert self.ids(body)[0] == "openalex-2"
        scores = [r["score"] for r in body["results"]]  # type: ignore[union-attr]
        assert scores == sorted(scores, reverse=True)

    def test_the_response_says_what_ranked_it(self) -> None:
        ranking = self.search()["ranking"]

        assert ranking["version"] == RANKER_VERSION  # type: ignore[index]
        # No encoder is wired in this app, so relevance is lexical, and says so.
        assert ranking["relevance"] == "lexical"  # type: ignore[index]
        assert ranking["weights"] == {  # type: ignore[index]
            "relevance": SHIPPED_WEIGHTS.relevance,
            "citation_impact": SHIPPED_WEIGHTS.citation,
            "recency": SHIPPED_WEIGHTS.recency,
        }

    def test_a_filter_hides_works_without_changing_the_parts_of_those_left(self) -> None:
        everything = {r["add"]["source_id"]: r for r in self.search()["results"]}  # type: ignore[union-attr]

        body = self.search(year_from=2023)

        assert self.ids(body) == ["openalex-2", "openalex-3"]
        assert body["excluded"] == {"year": 1, "citations": 0, "venue": 0, "unknown_citations": 0}
        for r in body["results"]:  # type: ignore[union-attr]
            assert r == everything[r["add"]["source_id"]]

    def test_a_minimum_citation_count_excludes_unknown_counts_and_says_how_many(self) -> None:
        body = self.search(min_citations=10)

        assert sorted(self.ids(body)) == ["openalex-1", "openalex-2"]
        assert body["excluded"]["citations"] == 1  # type: ignore[index]
        assert body["excluded"]["unknown_citations"] == 1  # type: ignore[index]

    def test_a_venue_filter_matches_part_of_the_name_in_any_case(self) -> None:
        assert self.ids(self.search(venue="findings")) == ["openalex-3"]

    def test_sorting_by_citations_puts_an_unknown_count_last(self) -> None:
        body = self.search(sort="citations")

        assert self.ids(body) == ["openalex-1", "openalex-2", "openalex-3"]
        assert body["ordering"] == "citations"

    @pytest.mark.parametrize(
        "body",
        [
            {"year_from": 2024, "year_to": 2020},
            {"min_citations": -1},
            {"sort": "popularity"},
            {"venue": ""},
        ],
    )
    def test_a_filter_that_cannot_mean_anything_is_a_422(self, body: dict[str, object]) -> None:
        response = build(Fake("openalex")).post(SEARCH, json={"query": "x", **body})

        assert response.status_code == HTTPStatus.UNPROCESSABLE_ENTITY


class TestNoSourceAnswering:
    def test_is_a_503_problem_never_an_empty_list(self) -> None:
        client = build(
            Fake("openalex", raises=down(SourceFailure.RATE_LIMITED, retry_after=7.0)),
            Fake("arxiv", raises=down(SourceFailure.CIRCUIT_OPEN, retry_after=15.0)),
        )

        response = client.post(SEARCH, json={"query": "x"})

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert response.headers["content-type"] == PROBLEM_CONTENT_TYPE
        body = response.json()
        assert body["type"] == "/problems/discovery-unavailable"
        assert "results" not in body, "a failure must not look like an answer with no papers"
        assert [(s["source"], s["status"]) for s in body["source_status"]] == [
            ("openalex", "rate_limited"),
            ("arxiv", "circuit_open"),
        ]
        # Every source said when; the soonest is when this search can next succeed.
        assert response.headers["Retry-After"] == "7"

    def test_says_nothing_about_when_unless_every_source_did(self) -> None:
        client = build(
            Fake("openalex", raises=down(SourceFailure.RATE_LIMITED, retry_after=7.0)),
            Fake("arxiv", raises=down(SourceFailure.ERROR)),
        )

        response = client.post(SEARCH, json={"query": "x"})

        assert response.status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert "Retry-After" not in response.headers


class TestWhatTheClientIsNotTold:
    @pytest.mark.parametrize("partial", [True, False])
    def test_why_a_source_failed_stays_in_the_log(self, partial: bool) -> None:
        leak = "upstream says: db-7.internal refused user scholiast"
        sources: list[object] = [Fake("arxiv", raises=down(SourceFailure.ERROR, detail=leak))]
        if partial:
            sources.append(Fake("openalex"))

        with capture_logs() as logs:
            response = build(*sources).post(SEARCH, json={"query": "x"})

        assert leak not in response.text
        assert any(leak in str(entry.get("detail")) for entry in logs), "and it is not lost"

    @respx.mock
    def test_the_openalex_key_reaches_neither_a_log_nor_the_client(self) -> None:
        respx.get("https://api.openalex.org/works").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        with capture_logs() as logs:
            response = build(OpenAlexSource(KEY), Fake("arxiv")).post(SEARCH, json={"query": "x"})

        assert response.status_code == HTTPStatus.OK
        assert KEY not in response.text
        assert KEY not in repr(logs)

    def test_a_crashing_source_is_logged_with_its_traceback(self) -> None:
        with capture_logs() as logs:
            response = build(Fake("buggy", raises=KeyError("results")), Fake("ok")).post(
                SEARCH, json={"query": "x"}
            )

        assert response.status_code == HTTPStatus.OK
        crashed = [e for e in logs if e["event"] == "discovery.source_crashed"]
        assert crashed and crashed[0]["source"] == "buggy"
        assert crashed[0]["exc_info"][0] is KeyError


class TestTheRequest:
    def test_a_query_of_punctuation_is_a_400_and_no_source_is_asked(self) -> None:
        source = Fake("openalex")
        response = build(source).post(SEARCH, json={"query": " ?!? "})

        assert response.status_code == HTTPStatus.BAD_REQUEST
        assert response.json()["type"] == "/problems/invalid-request"
        assert source.calls == 0

    @pytest.mark.parametrize(
        "payload",
        [{"query": "   "}, {"query": "x" * 501}, {"query": "x", "limit": 5}, {}],
        ids=["blank", "too long", "unknown field", "missing"],
    )
    def test_a_malformed_body_is_a_422(self, payload: dict[str, object]) -> None:
        assert build(Fake("a")).post(SEARCH, json=payload).status_code == (
            HTTPStatus.UNPROCESSABLE_ENTITY
        )

    def test_login_is_required_even_without_the_limiter(self) -> None:
        """Authentication must not depend on the limiter happening to need a user.

        Today the limiter resolves the principal whether or not limiting is on, so
        switching limiting off proves nothing: a mutation run showed that version of
        this test passing with the route's own login dependency deleted. Replacing
        the limiter outright, as moving limits to a gateway would, is what isolates
        the claim.
        """
        client = build(Fake("a"), authenticated=False)
        client.app.dependency_overrides[deps.DiscoveryRateLimit.dependency] = lambda: None  # type: ignore[attr-defined]

        assert client.post(SEARCH, json={"query": "x"}).status_code == HTTPStatus.UNAUTHORIZED

    def test_the_configured_discovery_allowance_is_the_one_applied(self) -> None:
        client = build(Fake("a"), settings=Settings(rate_limit_discovery_per_minute=2))

        statuses = [client.post(SEARCH, json={"query": "x"}).status_code for _ in range(3)]

        assert statuses == [HTTPStatus.OK, HTTPStatus.OK, HTTPStatus.TOO_MANY_REQUESTS]


class TestMetrics:
    def test_each_sources_outcome_is_counted(self) -> None:
        def count(source: str, status: str) -> float:
            value = REGISTRY.get_sample_value(
                "scholiast_discovery_source_outcomes_total", {"source": source, "status": status}
            )
            return value or 0.0

        before = (count("m-ok", "ok"), count("m-down", "rate_limited"))
        build(Fake("m-ok"), Fake("m-down", raises=down(SourceFailure.RATE_LIMITED))).post(
            SEARCH, json={"query": "x"}
        )

        assert (count("m-ok", "ok"), count("m-down", "rate_limited")) == (
            before[0] + 1,
            before[1] + 1,
        )
