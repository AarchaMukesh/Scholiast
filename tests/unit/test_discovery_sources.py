"""OpenAlex and arXiv adapters, hermetic (D117, D118, D119).

`respx` intercepts every call, so this file makes zero network calls (NFR-T3). The
fixtures under tests/fixtures/discovery/ are cut from the spike's frozen pools, real
responses (provenance.json says which), except the arXiv error feed, which the spike
never received and which is written to arXiv's documented format.

What is worth testing is what fails silently: a failure read as "no papers", a
wildcard that turns every typed question into an error, a key in a URL, a citation
count invented as zero, and a PDF link rebuilt from a bare id.
"""

from __future__ import annotations

import json
import ssl
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import respx

from scholiast.application.ports.discovery import (
    AcademicSource,
    SourceFailure,
    SourceUnavailable,
)
from scholiast.domain.discovery import LinkKind
from scholiast.infrastructure.discovery.arxiv import (
    STOPWORDS,
    ArxivSource,
    arxiv_terms,
    parse_feed,
)
from scholiast.infrastructure.discovery.openalex import (
    OpenAlexSource,
    openalex_search_text,
    parse_works,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "discovery"
PROTOCOL = Path(__file__).resolve().parents[2] / "spike" / "m4-protocol.json"
OPENALEX = "https://openalex.test"
ARXIV = "https://arxiv.test/api/query"
KEY = "test-key-0123456789"


def fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def openalex() -> OpenAlexSource:
    return OpenAlexSource(KEY, base_url=OPENALEX)


def arxiv() -> ArxivSource:
    return ArxivSource(base_url=ARXIV)


class TestTheContract:
    def test_both_adapters_satisfy_the_port(self) -> None:
        assert isinstance(openalex(), AcademicSource)
        assert isinstance(arxiv(), AcademicSource)

    @pytest.mark.parametrize("query", ["??", " * ", ""])
    def test_a_query_with_nothing_searchable_is_the_callers_error_not_openalex_failing(
        self, query: str
    ) -> None:
        # Reported as a source failure, this would mark OpenAlex down in per-source
        # status (FR-D6) for what was a bad request.
        with pytest.raises(ValueError):
            openalex().request(query, limit=10)

    @pytest.mark.parametrize("query", ["the of and", "?", "a b c"])
    def test_a_query_of_only_stopwords_is_the_callers_error_not_arxiv_failing(
        self, query: str
    ) -> None:
        with pytest.raises(ValueError):
            arxiv().request(query, limit=10)

    def test_openalex_refuses_to_start_without_a_key(self) -> None:
        # Keyless OpenAlex is a test allowance only (D116); starting without one
        # would fail every search at runtime instead of once at wiring.
        with pytest.raises(ValueError):
            OpenAlexSource("  ")


class TestOpenAlexRequest:
    def test_a_question_mark_never_reaches_openalex(self) -> None:
        # D117: OpenAlex refuses '?' and '*' in stemmed search with HTTP 400, so
        # every question a user types as a question would fail.
        request = openalex().request("How do diffusion models generate *images?", limit=50)
        search = parse_qs(urlsplit(str(request.url)).query)["search"][0]
        assert "?" not in search and "*" not in search
        assert search == "how do diffusion models generate images"
        assert openalex_search_text("a?b*c") == "a b c"

    def test_the_key_travels_in_the_authorization_header_and_never_in_the_url(self) -> None:
        # A URL is written to access logs, proxies and tracebacks; a header is not.
        request = openalex().request("retrieval augmented generation", limit=50)
        assert KEY not in str(request.url)
        assert request.headers["Authorization"] == f"Bearer {KEY}"

    def test_the_request_has_the_shape_the_pools_were_recorded_with(self) -> None:
        # D119: phase 3 changes this with measurements, not before.
        params = parse_qs(urlsplit(str(openalex().request("x y", limit=50).url)).query)
        assert params["per_page"] == ["50"]
        assert (
            "relevance_score" in params["select"][0]
            and "abstract_inverted_index" in params["select"][0]
        )


class TestOpenAlexParsing:
    def test_a_recorded_page_becomes_candidates_with_what_openalex_reported(self) -> None:
        page = parse_works(json.loads(fixture("openalex_search.json")))
        assert page.source == "openalex" and page.skipped == 0
        assert page.total_hits is not None and page.total_hits > len(page.candidates)
        assert [c.rank for c in page.candidates] == [1, 2, 3]
        assert {c.adapter_version for c in page.candidates} == {"openalex-2"}

    def test_an_arxiv_id_is_found_through_a_location_when_the_doi_is_a_publishers(self) -> None:
        # Phase 7 links this work to arXiv's record of it only through this id.
        page = parse_works(json.loads(fixture("openalex_search.json")))
        work = next(c for c in page.candidates if c.source_id == "W3015883388")
        assert work.doi is not None and not work.doi.startswith("10.48550/")
        assert work.arxiv_id is not None

    def test_an_arxiv_datacite_doi_yields_the_arxiv_id(self) -> None:
        page = parse_works(json.loads(fixture("openalex_search.json")))
        work = next(c for c in page.candidates if c.source_id == "W4389984066")
        assert work.doi is not None and work.doi.startswith("10.48550/arxiv.")
        assert work.arxiv_id == work.doi.removeprefix("10.48550/arxiv.")

    def test_an_absent_abstract_is_none_and_an_oa_url_is_a_landing_page_not_a_pdf(self) -> None:
        # The spike found every oa_url-only link it fetched was an HTML page.
        page = parse_works(json.loads(fixture("openalex_search.json")))
        work = next(c for c in page.candidates if c.source_id == "W4410705984")
        assert work.abstract is None
        assert work.oa_links and all(link.kind is LinkKind.LANDING for link in work.oa_links)

    def test_the_inverted_index_abstract_is_rebuilt_in_word_order(self) -> None:
        page = parse_works(
            {
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W9",
                        "display_name": "T",
                        "abstract_inverted_index": {"attention": [1], "is": [2], "all": [3, 0]},
                    }
                ],
            }
        )
        assert page.candidates[0].abstract == "all attention is all"

    def test_a_record_with_no_title_is_counted_as_skipped_not_dropped_silently(self) -> None:
        page = parse_works(
            {
                "results": [
                    {"id": "https://openalex.org/W1", "display_name": " "},
                    {"id": "", "display_name": "T"},
                ]
            }
        )
        assert page.candidates == () and page.skipped == 2

    def test_a_missing_citation_count_stays_none(self) -> None:
        page = parse_works({"results": [{"id": "https://openalex.org/W1", "display_name": "T"}]})
        assert page.candidates[0].citation_count is None


class TestOpenAlexFailures:
    @respx.mock
    async def test_openalexs_own_400_is_a_source_error_with_its_message_not_an_empty_page(
        self,
    ) -> None:
        respx.get(f"{OPENALEX}/works").mock(
            return_value=httpx.Response(400, text=fixture("openalex_400_wildcard.json"))
        )
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert caught.value.kind is SourceFailure.ERROR
        assert "Wildcards" in caught.value.detail

    @respx.mock
    async def test_a_429_is_rate_limited_and_carries_retry_after(self) -> None:
        respx.get(f"{OPENALEX}/works").mock(
            return_value=httpx.Response(429, headers={"retry-after": "30"})
        )
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert caught.value.kind is SourceFailure.RATE_LIMITED
        assert caught.value.retry_after == pytest.approx(30.0)

    @respx.mock
    async def test_a_server_error_is_an_error(self) -> None:
        respx.get(f"{OPENALEX}/works").mock(return_value=httpx.Response(502, text="bad gateway"))
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert caught.value.kind is SourceFailure.ERROR

    @respx.mock
    async def test_a_timeout_is_reported_as_a_timeout(self) -> None:
        respx.get(f"{OPENALEX}/works").mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert caught.value.kind is SourceFailure.TIMEOUT

    @respx.mock
    async def test_a_200_that_is_not_json_is_an_error_not_an_empty_page(self) -> None:
        respx.get(f"{OPENALEX}/works").mock(
            return_value=httpx.Response(200, text="<html>maintenance</html>")
        )
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert caught.value.kind is SourceFailure.ERROR

    @respx.mock
    async def test_a_failure_message_never_contains_the_key(self) -> None:
        respx.get(f"{OPENALEX}/works").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(SourceUnavailable) as caught:
            await openalex().search("anything", limit=5)
        assert KEY not in str(caught.value)

    @respx.mock
    async def test_a_successful_search_sends_one_request_and_parses_it(self) -> None:
        route = respx.get(f"{OPENALEX}/works").mock(
            return_value=httpx.Response(200, text=fixture("openalex_search.json"))
        )
        page = await openalex().search("retrieval augmented generation long documents", limit=50)
        assert route.call_count == 1 and len(page.candidates) == 3
        assert route.calls.last.request.headers["Authorization"] == f"Bearer {KEY}"


class TestArxivRequest:
    def test_only_words_reach_arxivs_query_syntax(self) -> None:
        # A user typing a field prefix, a bracket or an operator must not be able
        # to change what arXiv is asked.
        terms = arxiv_terms('ti:attention (Is) ANDNOT "all" you need?')
        assert terms == ["ti", "attention", "andnot", "all", "you", "need"]
        query = parse_qs(urlsplit(str(arxiv().request("ti:xx (yy) zz", limit=5).url)).query)[
            "search_query"
        ][0]
        assert query == "all:ti AND all:xx AND all:yy AND all:zz"

    def test_the_stopwords_are_the_ones_the_spike_measured_with(self) -> None:
        # D120: the construction shipped must be the construction measured.
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        assert frozenset(protocol["query_construction"]["arxiv_stopwords"]) == STOPWORDS

    def test_the_join_is_and_or_or_and_nothing_else(self) -> None:
        with pytest.raises(ValueError):
            ArxivSource(join="NOT")
        query = parse_qs(
            urlsplit(str(ArxivSource(join="OR").request("dense retrieval", limit=5).url)).query
        )
        assert query["search_query"] == ["all:dense OR all:retrieval"]


class TestArxivParsing:
    def test_a_recorded_feed_becomes_candidates(self) -> None:
        page = parse_feed(fixture("arxiv_search.xml"))
        assert len(page.candidates) == 3 and page.skipped == 0
        assert page.total_hits is not None and page.total_hits >= 3
        assert all(c.abstract and c.authors and c.year for c in page.candidates)

    def test_arxiv_never_reports_a_citation_count_so_none_is_recorded(self) -> None:
        # Zero would rank every arXiv candidate as uncited.
        assert {c.citation_count for c in parse_feed(fixture("arxiv_search.xml")).candidates} == {
            None
        }

    def test_the_pdf_link_is_the_versioned_one_arxiv_gave_never_rebuilt_from_the_id(self) -> None:
        # The spike rebuilt https://arxiv.org/pdf/2507.20128 from the bare id and got
        # a 404: arXiv marks this entry a superseded duplicate.
        page = parse_feed(fixture("arxiv_superseded_duplicate.xml"))
        (candidate,) = page.candidates
        assert candidate.arxiv_id == "2507.20128"
        assert candidate.source_id == "2507.20128v2"
        assert [link.url for link in candidate.oa_links] == ["https://arxiv.org/pdf/2507.20128v2"]

    def test_an_error_feed_raises_rather_than_reading_as_no_papers(self) -> None:
        # arXiv reports a bad query as a 200 whose single entry is the error.
        with pytest.raises(SourceUnavailable) as caught:
            parse_feed(fixture("arxiv_error.xml"))
        assert caught.value.kind is SourceFailure.ERROR
        assert "incorrect id format" in caught.value.detail

    def test_an_entity_expansion_bomb_is_refused_not_expanded(self) -> None:
        # The XML comes from outside. Expat 2.6 refuses the amplification; if it ever
        # did not, this would hang or exhaust memory instead of failing.
        lol = "lol" * 10
        dtd = "".join(f'<!ENTITY l{i} "{("&l" + str(i - 1) + ";") * 10}">' for i in range(1, 10))
        bomb = f'<?xml version="1.0"?><!DOCTYPE feed [<!ENTITY l0 "{lol}">{dtd}]><feed>&l9;</feed>'
        with pytest.raises(SourceUnavailable) as caught:
            parse_feed(bomb)
        assert caught.value.kind is SourceFailure.ERROR

    def test_an_unparseable_feed_is_an_error(self) -> None:
        with pytest.raises(SourceUnavailable):
            parse_feed("<feed><entry>")


class TestArxivFailures:
    @respx.mock
    async def test_a_503_is_arxivs_throttle_and_is_rate_limited_with_retry_after(self) -> None:
        respx.get(ARXIV).mock(return_value=httpx.Response(503, headers={"retry-after": "15"}))
        with pytest.raises(SourceUnavailable) as caught:
            await arxiv().search("dense retrieval", limit=5)
        assert caught.value.kind is SourceFailure.RATE_LIMITED
        assert caught.value.retry_after == pytest.approx(15.0)

    @respx.mock
    async def test_a_timeout_is_reported_as_a_timeout(self) -> None:
        # The spike's A1 run met 20 s arXiv timeouts on 2026-10-02.
        respx.get(ARXIV).mock(side_effect=httpx.ReadTimeout("slow"))
        with pytest.raises(SourceUnavailable) as caught:
            await arxiv().search("dense retrieval", limit=5)
        assert caught.value.kind is SourceFailure.TIMEOUT

    @respx.mock
    async def test_a_client_error_is_an_error(self) -> None:
        respx.get(ARXIV).mock(return_value=httpx.Response(400, text="bad"))
        with pytest.raises(SourceUnavailable) as caught:
            await arxiv().search("dense retrieval", limit=5)
        assert caught.value.kind is SourceFailure.ERROR

    @respx.mock
    async def test_a_successful_search_parses_the_feed(self) -> None:
        route = respx.get(ARXIV).mock(
            return_value=httpx.Response(200, text=fixture("arxiv_search.xml"))
        )
        page = await arxiv().search("dense retrieval open domain question answering", limit=50)
        assert route.call_count == 1 and len(page.candidates) == 3


class TestASearchDoesNotRebuildTheTlsContext:
    """D122: a TLS context costs 183 ms to build, synchronously, on this machine.

    Phase 2's adapters built one per search, inside a fresh httpx client, so two
    sources blocked the event loop for about 370 ms before either request left:
    18% of the 2.0 s deadline, and outside what addendum A2 measured. Found because
    the fan-out tests through the real adapters took 0.38 s each.
    """

    @respx.mock
    async def test_searches_after_the_first_build_no_tls_context(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        respx.get(f"{OPENALEX}/works").mock(
            return_value=httpx.Response(200, text=fixture("openalex_search.json"))
        )
        respx.get(ARXIV).mock(return_value=httpx.Response(200, text=fixture("arxiv_search.xml")))
        # The shared context may be built once, by whoever needs it first.
        await openalex().search("dense retrieval", limit=25)
        await arxiv().search("dense retrieval", limit=50)

        built: list[object] = []
        real = ssl.create_default_context

        def counting(*args: object, **kwargs: object) -> ssl.SSLContext:
            built.append(args)
            return real(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(ssl, "create_default_context", counting)
        for _ in range(2):
            await openalex().search("dense retrieval", limit=25)
            await arxiv().search("dense retrieval", limit=50)

        assert built == [], f"{len(built)} TLS contexts built for 4 searches"


class TestAPoisonedLinkCostsTheLinkNotTheRecord:
    """D122: a non-http(s) URL in a source's record is dropped before it reaches a browser.

    Dropped, not refused with the record: a `javascript:` link raising inside the
    parser would fail the whole page, and one poisoned record in OpenAlex would then
    take OpenAlex out of every search that matched it.
    """

    def test_openalex_drops_the_bad_link_and_keeps_the_rest(self) -> None:
        payload = json.loads(fixture("openalex_search.json"))
        work = payload["results"][1]
        work["best_oa_location"]["pdf_url"] = "javascript:alert(document.cookie)"
        work["open_access"]["oa_url"] = "data:text/html,<script>alert(1)</script>"

        page = parse_works(payload)

        kept = page.candidates[1]
        assert len(page.candidates) == 3
        assert [link.url for link in kept.oa_links] == [
            "https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf",
            "https://arxiv.org/pdf/2004.04906",
        ]

    def test_arxiv_drops_the_bad_link_and_keeps_the_rest(self) -> None:
        feed = fixture("arxiv_search.xml").replace(
            'href="https://arxiv.org/pdf/2004.04906v3"', 'href="javascript:alert(1)"'
        )

        page = parse_feed(feed)

        assert len(page.candidates) == 3
        assert page.candidates[0].arxiv_id == "2004.04906"
        assert page.candidates[0].oa_links == ()
        assert all(link.url.startswith("https://") for c in page.candidates for link in c.oa_links)
