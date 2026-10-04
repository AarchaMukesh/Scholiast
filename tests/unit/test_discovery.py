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
    normalise_arxiv_id,
    normalise_doi,
    normalise_query,
    normalise_title,
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
