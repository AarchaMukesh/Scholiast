"""The ranking score and its explanation (FR-D4, FR-D5, FR-D8, D151-D153).

What fails silently here is an explanation that does not add up to the score, an
unknown read as zero, a filter that changes what it was meant only to hide, and a
sort that puts unknowns first. Each test builds works by hand; the measured weights
are held to their fit in `test_eval_ranking.py`.
"""

from __future__ import annotations

import math

import pytest

from scholiast.domain.dedup import DiscoveredWork, merge
from scholiast.domain.discovery import PaperCandidate
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.ranking import (
    RankingWeights,
    SearchFilters,
    SortOrder,
    apply_filters,
    citation_impact,
    lexical_relevance,
    rank,
    recency,
    relevance_text,
    scale,
)

YEAR = 2026
EVEN = RankingWeights(relevance=0.4, citation=0.3, recency=0.3)


def work(n: int, **fields: object) -> DiscoveredWork:
    base: dict[str, object] = {
        "source": "openalex",
        "source_id": f"W{n}",
        "rank": n,
        "title": f"Paper {n}",
        "year": 2020,
        "adapter_version": "t",
    }
    base.update(fields)
    return merge([PaperCandidate(**base)])  # type: ignore[arg-type]


class TestTheParts:
    def test_citation_impact_divides_by_age_so_old_papers_do_not_win_by_waiting(self) -> None:
        # 100 citations in one year beats 300 over nine.
        young = citation_impact(100, 2025, YEAR)
        old = citation_impact(300, 2017, YEAR)

        assert young is not None and old is not None
        assert young > old
        assert young == pytest.approx(math.log1p(100 / 2))

    def test_recency_halves_every_three_years(self) -> None:
        assert recency(YEAR, YEAR) == 1.0
        assert recency(YEAR - 3, YEAR) == pytest.approx(0.5)
        assert recency(YEAR - 6, YEAR) == pytest.approx(0.25)

    def test_a_future_year_is_not_more_recent_than_this_one(self) -> None:
        assert recency(YEAR + 2, YEAR) == 1.0

    def test_unknown_inputs_are_unknown_not_zero(self) -> None:
        assert citation_impact(None, 2020, YEAR) is None
        assert citation_impact(10, None, YEAR) is None
        assert recency(None, YEAR) is None


class TestScaling:
    def test_spans_zero_to_one(self) -> None:
        assert scale([2.0, 4.0, 3.0]) == [0.0, 1.0, 0.5]

    def test_equal_relevance_is_full_relevance_not_none(self) -> None:
        assert scale([0.7, 0.7]) == [1.0, 1.0]

    def test_nothing_matched_stays_zero(self) -> None:
        assert scale([0.0, 0.0]) == [0.0, 0.0]


class TestTheScore:
    def test_is_exactly_the_weighted_sum_of_the_parts_shown(self) -> None:
        works = [
            work(1, citation_count=50, year=2024),
            work(2, citation_count=5, year=2018),
            work(3, citation_count=None, year=None),
        ]

        for r in rank(works, [3.0, 1.0, 2.0], EVEN, ranking_year=YEAR):
            p = r.parts
            assert r.score == pytest.approx(
                EVEN.relevance * p.relevance + EVEN.citation * p.citation + EVEN.recency * p.recency
            )

    def test_an_unknown_citation_count_takes_the_searchs_median_and_says_so(self) -> None:
        works = [
            work(1, citation_count=0, year=2024),
            work(2, citation_count=1000, year=2024),
            work(3, citation_count=40, year=2024),
            work(4, citation_count=None, year=2024),
        ]

        ranked = {
            r.work.add.source_id: r.parts
            for r in rank(works, [1, 1, 1, 1], EVEN, ranking_year=YEAR)
        }

        assert ranked["W4"].citation_imputed
        assert ranked["W4"].citation == pytest.approx(ranked["W3"].citation)
        assert not ranked["W3"].citation_imputed
        assert 0.0 < ranked["W4"].citation < 1.0, "unknown was read as uncited or as the best"

    def test_an_unknown_year_takes_the_median_recency_and_says_so(self) -> None:
        works = [work(1, year=2026), work(2, year=2020), work(3, year=None)]

        parts = {
            r.work.add.source_id: r.parts for r in rank(works, [1, 1, 1], EVEN, ranking_year=YEAR)
        }

        assert parts["W3"].recency_imputed
        assert parts["W3"].recency == pytest.approx((1.0 + 0.25) / 2)

    def test_higher_scores_come_first_and_ties_keep_the_sources_order(self) -> None:
        works = [work(1), work(2), work(3)]

        ranked = rank(works, [1.0, 5.0, 1.0], RankingWeights(1.0, 0.0, 0.0), ranking_year=YEAR)

        assert [r.work.add.source_id for r in ranked] == ["W2", "W1", "W3"]
        assert [r.position for r in ranked] == [1, 0, 2]

    def test_one_relevance_score_per_work_or_nothing(self) -> None:
        with pytest.raises(InvariantViolation):
            rank([work(1), work(2)], [1.0], EVEN, ranking_year=YEAR)

    @pytest.mark.parametrize(
        ("relevance", "citation", "recency"), [(0.5, 0.5, 0.5), (1.2, -0.2, 0.0)]
    )
    def test_weights_that_do_not_sum_to_one_or_go_negative_are_refused(
        self, relevance: float, citation: float, recency: float
    ) -> None:
        with pytest.raises(InvariantViolation):
            RankingWeights(relevance, citation, recency)


class TestLexicalRelevance:
    def test_a_term_every_result_shares_counts_for_less_than_one_that_separates_them(self) -> None:
        texts = [
            "diffusion models for images",
            "diffusion models for audio",
            "diffusion models for molecules",
        ]

        scores = lexical_relevance("diffusion audio", texts)

        assert scores[1] == max(scores)
        assert scores[0] == scores[2] > 0

    def test_a_rare_term_outweighs_repeating_a_common_one(self) -> None:
        # Without IDF, saying "diffusion" three times would beat saying "audio"
        # once, though every result says "diffusion" and only one says "audio".
        texts = [
            "diffusion diffusion diffusion models",
            "audio models",
            "diffusion images",
            "diffusion molecules",
        ]

        scores = lexical_relevance("diffusion audio", texts)

        assert scores[1] > scores[0]

    def test_an_identifier_survives_tokenisation(self) -> None:
        scores = lexical_relevance("LLaMA-2", ["we fine-tune LLaMA-2", "we fine-tune GPT-2"])

        assert scores[0] > scores[1]

    def test_relevance_reads_the_title_and_the_abstract(self) -> None:
        assert relevance_text(work(1, abstract="An abstract.")) == "Paper 1. An abstract."
        assert relevance_text(work(2)) == "Paper 2"


class TestFilters:
    def ranked(self) -> tuple:
        works = [
            work(1, year=2019, citation_count=500, venue="NeurIPS"),
            work(2, year=2024, citation_count=None, venue=None),
            work(3, year=2022, citation_count=12, venue="ACL Findings"),
            work(4, year=None, citation_count=80, venue="EMNLP"),
        ]
        return rank(works, [4.0, 3.0, 2.0, 1.0], EVEN, ranking_year=YEAR)

    def test_a_filter_hides_works_and_never_changes_a_remaining_works_parts(self) -> None:
        ranked = self.ranked()
        before = {r.work.add.source_id: (r.score, r.parts) for r in ranked}

        kept = apply_filters(ranked, SearchFilters(year_from=2020))

        for r in kept.works:
            assert (r.score, r.parts) == before[r.work.add.source_id]

    def test_a_year_range_excludes_unknown_years_and_counts_them(self) -> None:
        kept = apply_filters(self.ranked(), SearchFilters(year_from=2020, year_to=2023))

        assert [r.work.add.source_id for r in kept.works] == ["W3"]
        assert kept.excluded == {"year": 3}

    def test_unknown_citations_cannot_satisfy_a_minimum_and_are_counted_apart(self) -> None:
        kept = apply_filters(self.ranked(), SearchFilters(min_citations=50))

        assert [r.work.add.source_id for r in kept.works] == ["W1", "W4"]
        assert kept.excluded == {"citations": 2}
        assert kept.excluded_unknown_citations == 1

    def test_a_venue_matches_case_insensitively_inside_the_name(self) -> None:
        kept = apply_filters(self.ranked(), SearchFilters(venue="findings"))

        assert [r.work.add.source_id for r in kept.works] == ["W3"]

    def test_sorting_by_citations_or_year_puts_unknowns_last(self) -> None:
        ranked = self.ranked()

        by_citations = apply_filters(ranked, SearchFilters(sort=SortOrder.CITATIONS))
        by_year = apply_filters(ranked, SearchFilters(sort=SortOrder.YEAR))

        assert [r.work.add.source_id for r in by_citations.works] == ["W1", "W4", "W3", "W2"]
        assert [r.work.add.source_id for r in by_year.works] == ["W2", "W3", "W1", "W4"]

    def test_by_default_the_score_order_is_kept(self) -> None:
        kept = apply_filters(self.ranked(), SearchFilters())

        assert [r.work.add.source_id for r in kept.works] == [
            r.work.add.source_id for r in self.ranked()
        ]

    def test_an_empty_year_range_or_a_negative_minimum_is_refused(self) -> None:
        with pytest.raises(InvariantViolation):
            SearchFilters(year_from=2024, year_to=2020)
        with pytest.raises(InvariantViolation):
            SearchFilters(min_citations=-1)
