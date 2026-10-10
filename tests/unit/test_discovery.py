"""Discovery's domain rules (D119): the normalisation three phases share, and the candidate.

Every test names a merge, a split or a silent default that would otherwise corrupt
deduplication or ranking downstream.
"""

from __future__ import annotations

import pytest

from scholiast.domain.discovery import (
    LinkKind,
    OpenAccessLink,
    PaperCandidate,
    arxiv_id_from_url,
    interleave,
    is_searchable,
    is_web_url,
    normalise_arxiv_id,
    normalise_doi,
    normalise_query,
    normalise_title,
    paper_identity,
)
from scholiast.domain.errors import InvariantViolation


def _candidate(**overrides: object) -> PaperCandidate:
    fields: dict[str, object] = {
        "source": "openalex",
        "source_id": "W1",
        "rank": 1,
        "title": "LoRA: Low-Rank Adaptation of Large Language Models",
        "adapter_version": "openalex-1",
    }
    fields.update(overrides)
    return PaperCandidate(**fields)  # type: ignore[arg-type]


class TestNormalisation:
    def test_one_doi_spelled_four_ways_is_one_identity(self) -> None:
        spellings = [
            "https://doi.org/10.18653/V1/2020.EMNLP-MAIN.550",
            "http://dx.doi.org/10.18653/v1/2020.emnlp-main.550",
            "DOI:10.18653/v1/2020.emnlp-main.550",
            " 10.18653/v1/2020.emnlp-main.550 ",
        ]
        assert {normalise_doi(s) for s in spellings} == {"10.18653/v1/2020.emnlp-main.550"}

    @pytest.mark.parametrize(
        "value", ["", None, "N/A", "arXiv:2106.09685", "https://example.org/x", "10.1234"]
    )
    def test_a_value_that_is_not_a_doi_never_becomes_an_identity(self, value: str | None) -> None:
        # A junk string in a source's DOI field would otherwise merge every work
        # that carries the same junk (phase 7).
        assert normalise_doi(value) is None

    @pytest.mark.parametrize(
        "spelling",
        [
            "2106.09685",
            "2106.09685v2",
            "arXiv:2106.09685v1",
            "10.48550/arXiv.2106.09685",
            "https://arxiv.org/abs/2106.09685v3",
            "http://arxiv.org/pdf/2106.09685v2.pdf",
            "https://export.arxiv.org/abs/2106.09685",
        ],
    )
    def test_every_spelling_and_version_of_an_arxiv_paper_is_one_identity(
        self, spelling: str
    ) -> None:
        assert normalise_arxiv_id(spelling) == "2106.09685"

    def test_old_style_arxiv_ids_keep_their_archive(self) -> None:
        assert normalise_arxiv_id("hep-th/9901001v3") == "hep-th/9901001"

    def test_an_old_style_id_has_one_spelling_whatever_its_case(self) -> None:
        # D132: identity is unique per owner on this value, and a unique index over
        # two spellings of one id enforces nothing.
        assert normalise_arxiv_id("HEP-TH/9901001") == "hep-th/9901001"

    def test_an_arxiv_shaped_path_on_another_host_is_not_an_arxiv_id(self) -> None:
        assert arxiv_id_from_url("https://example.org/abs/2106.09685") is None
        assert normalise_arxiv_id("https://evil.example/pdf/2106.09685") is None

    def test_queries_that_differ_only_in_case_width_or_spacing_normalise_alike(self) -> None:
        # Phase 4 caches under this form: two spellings of one query must share it.
        assert normalise_query("  Low-Rank\tADAPTATION ") == normalise_query("low-rank adaptation")
        assert normalise_query("\uff2c\uff4f\uff32\uff21") == "lora"  # full-width "LoRA" (NFKC)
        assert normalise_query("Straße") == "strasse"  # casefold, not lower

    def test_normalisation_does_not_stem_or_drop_words(self) -> None:
        # Stemming here would be a relevance decision hidden inside a cache key.
        assert normalise_query("The Models of Attention") == "the models of attention"

    def test_titles_that_differ_only_in_punctuation_and_case_compare_equal(self) -> None:
        assert normalise_title("SimCSE: Simple Contrastive Learning") == normalise_title(
            "simcse — simple contrastive learning"
        )
        assert normalise_title(None) == ""


class TestCandidate:
    def test_an_unreported_citation_count_is_none_never_zero(self) -> None:
        # Zero would rank the paper as uncited; arXiv simply does not report counts.
        assert _candidate().citation_count is None

    def test_a_negative_citation_count_is_refused(self) -> None:
        with pytest.raises(InvariantViolation):
            _candidate(citation_count=-1)

    def test_a_doi_that_was_not_normalised_is_refused(self) -> None:
        # Two adapters storing one DOI two ways would make phase 7 miss the merge.
        with pytest.raises(InvariantViolation):
            _candidate(doi="https://doi.org/10.1/X")

    def test_a_versioned_arxiv_id_is_refused(self) -> None:
        with pytest.raises(InvariantViolation):
            _candidate(arxiv_id="2106.09685v2")

    @pytest.mark.parametrize(
        "overrides",
        [
            {"title": "   "},
            {"source": ""},
            {"source_id": " "},
            {"rank": 0},
            {"adapter_version": ""},
        ],
    )
    def test_a_candidate_missing_what_identifies_it_is_refused(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(InvariantViolation):
            _candidate(**overrides)

    def test_an_open_access_link_needs_a_url(self) -> None:
        with pytest.raises(InvariantViolation):
            OpenAccessLink(" ", LinkKind.PDF)


class TestOnlyAWebLinkMayReachABrowser:
    """D122: a link from a third-party record is shown to a browser from phase 3 on."""

    @pytest.mark.parametrize(
        "url",
        [
            "javascript:alert(document.cookie)",
            "JAVASCRIPT:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
            "ftp://example.org/paper.pdf",
            "//example.org/paper.pdf",
            "https://",
            "https://[::1/paper.pdf",
            "   ",
        ],
    )
    def test_anything_but_an_absolute_http_url_is_refused(self, url: str) -> None:
        assert not is_web_url(url)
        with pytest.raises(InvariantViolation):
            OpenAccessLink(url, LinkKind.PDF)

    @pytest.mark.parametrize(
        "url",
        ["https://arxiv.org/pdf/2106.09685v2", "http://aclanthology.org/2020.emnlp-main.550.pdf"],
    )
    def test_ordinary_links_are_kept(self, url: str) -> None:
        assert is_web_url(url)
        assert OpenAccessLink(url, LinkKind.PDF).url == url


class TestSearchability:
    @pytest.mark.parametrize("query", ["?", " ?!? ", "...", "", "   ", "* -- *"])
    def test_a_query_with_no_letter_or_digit_has_nothing_to_search(self, query: str) -> None:
        assert not is_searchable(query)

    @pytest.mark.parametrize(
        "query", ["LoRA", "how do diffusion models work?", "Ψ-net", "2106.09685"]
    )
    def test_any_letter_or_digit_in_any_script_is_searchable(self, query: str) -> None:
        assert is_searchable(query)


class TestInterleaving:
    """D122's order until phase 8: each source's first, then each one's second."""

    def test_sources_take_turns_and_a_shorter_page_simply_runs_out(self) -> None:
        a = [_candidate(source="a", source_id=f"a{i}", rank=i) for i in (1, 2, 3)]
        b = [_candidate(source="b", source_id=f"b{i}", rank=i) for i in (1,)]

        assert [c.source_id for c in interleave([a, b])] == ["a1", "b1", "a2", "a3"]

    def test_duplicates_across_sources_are_kept_for_phase_7(self) -> None:
        # Merging here would be dedup before dedup exists, with no measurement.
        same = _candidate(source="a", source_id="x", doi="10.1/same")
        other = _candidate(source="b", source_id="y", doi="10.1/same")

        assert len(interleave([[same], [other]])) == 2

    def test_no_pages_is_no_candidates(self) -> None:
        assert interleave([]) == ()
        assert interleave([[], []]) == ()


class TestOneWorkHasOneIdentity:
    """D132: the (DOI, arXiv id) a paper is found by, one spelling each."""

    def test_arxivs_own_doi_is_folded_into_the_arxiv_id(self) -> None:
        # OpenAlex keeps the DataCite DOI as the DOI; arXiv reports none. Kept apart,
        # the same work would carry two identities and could be held twice.
        assert paper_identity("https://doi.org/10.48550/arXiv.2312.10997", None) == (
            None,
            "2312.10997",
        )

    def test_a_publisher_doi_stays_beside_the_arxiv_id(self) -> None:
        assert paper_identity("10.18653/V1/2020.EMNLP-MAIN.550", "2004.04906v3") == (
            "10.18653/v1/2020.emnlp-main.550",
            "2004.04906",
        )

    def test_a_reported_arxiv_id_wins_over_the_one_inside_a_datacite_doi(self) -> None:
        assert paper_identity("10.48550/arxiv.2312.10997", "2312.10997v2") == (None, "2312.10997")

    def test_nothing_identifying_is_nothing(self) -> None:
        assert paper_identity(None, None) == (None, None)
        assert paper_identity("not a doi", "not an id") == (None, None)
