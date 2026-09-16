"""Metrics checked against values derived by hand, not against the pipeline.

The temptation with retrieval metrics is to assert that `ndcg_at_k` returns what
`ndcg_at_k` returned last time. That is a change-detector, not a test: it passes
just as happily when the formula is wrong from the start. Every expected value
below is written as the formula itself — `(1 + 3/log2(3)) / (3 + 1/log2(3))` rather
than `0.797` — so the test encodes the definition independently of the code, and a
transposed discount or a linear gain shows up as a failure rather than a new
baseline.
"""

from __future__ import annotations

import math

import pytest

from scholiast.eval.metrics import (
    BOOTSTRAP_SEED,
    bootstrap_interval,
    dcg_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_counts_relevant_chunks_inside_the_cutoff(self) -> None:
        # Two relevant chunks exist; one is inside the top 2.
        assert recall_at_k(["a", "b", "c"], {"a": 2, "z": 2}, 2) == 0.5

    def test_the_denominator_is_all_relevant_chunks_not_the_cutoff(self) -> None:
        # The classic error is dividing by k, which makes recall fall as k rises
        # even when nothing was missed. Four relevant, all found, k=10 -> 1.0.
        relevant = {"a": 2, "b": 2, "c": 2, "d": 2}
        assert recall_at_k(["a", "b", "c", "d"], relevant, 10) == 1.0

    def test_chunks_past_the_cutoff_do_not_count(self) -> None:
        assert recall_at_k(["x", "y", "a"], {"a": 2}, 2) == 0.0

    def test_an_unjudged_query_raises_rather_than_scoring_zero(self) -> None:
        # Returning 0.0 or 1.0 would silently shift the mean over a query set.
        with pytest.raises(ValueError, match="undefined"):
            recall_at_k(["a"], {}, 10)


class TestReciprocalRank:
    def test_is_the_inverse_of_the_first_relevant_position(self) -> None:
        assert reciprocal_rank(["x", "y", "a"], {"a": 2}) == pytest.approx(1 / 3)

    def test_is_one_when_the_top_hit_is_relevant(self) -> None:
        assert reciprocal_rank(["a", "x"], {"a": 2}) == 1.0

    def test_is_zero_when_nothing_relevant_was_retrieved(self) -> None:
        # Unlike recall's empty-qrels case, a miss is a real outcome, not bad input.
        assert reciprocal_rank(["x", "y"], {"a": 2}) == 0.0

    def test_only_the_first_hit_matters(self) -> None:
        assert reciprocal_rank(["x", "a", "b"], {"a": 2, "b": 2}) == pytest.approx(0.5)


class TestDcgAndNdcg:
    def test_dcg_uses_the_exponential_gain_and_log2_discount(self) -> None:
        # Hand-derived: grade 2 at rank 1 contributes (2^2-1)/log2(2) = 3,
        # grade 1 at rank 2 contributes (2^1-1)/log2(3).
        expected = 3.0 / math.log2(2) + 1.0 / math.log2(3)
        assert dcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(expected)

    def test_a_perfect_ranking_scores_one(self) -> None:
        assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 2) == pytest.approx(1.0)

    def test_swapping_two_grades_costs_exactly_the_derived_amount(self) -> None:
        # The whole point of a rank-sensitive metric: same chunks, worse order.
        expected = (1.0 + 3.0 / math.log2(3)) / (3.0 + 1.0 / math.log2(3))
        assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) == pytest.approx(expected)
        assert expected < 1.0

    def test_the_ideal_ranking_is_by_grade_not_by_what_we_retrieved(self) -> None:
        # Normalising against our own ranking sorted by grade would divide the
        # ordering error away and make every ranking score 1.0.
        assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 2) < 1.0

    def test_ndcg_normalises_against_the_ideal_truncated_at_k(self) -> None:
        # Retrieving one of two equally relevant chunks at k=1 is nDCG@1 = 1.0,
        # because one slot filled with the best available item IS the ideal
        # ranking for one slot. That is the standard definition, and the first
        # draft of this test asserted the opposite — it expected the unretrieved
        # chunk to cost something, which is recall's job, not nDCG's.
        #
        # Pinned deliberately: normalising against the UNtruncated ideal is a
        # plausible-looking "fix" that would silently deflate every nDCG in every
        # report and make the ranking metric partly a coverage metric.
        assert ndcg_at_k(["a"], {"a": 2, "b": 2}, 1) == pytest.approx(1.0)

    def test_recall_is_what_sees_the_chunk_ndcg_at_k_cannot(self) -> None:
        # The division of labour, asserted on one case so neither metric quietly
        # grows into the other's job.
        ranking, relevant = ["a"], {"a": 2, "b": 2}
        assert ndcg_at_k(ranking, relevant, 1) == pytest.approx(1.0)
        assert recall_at_k(ranking, relevant, 1) == 0.5

    def test_ordering_within_the_cutoff_still_matters_at_equal_grades(self) -> None:
        # nDCG@k being blind past k must not make it blind inside k.
        good = ndcg_at_k(["a", "x", "b"], {"a": 2, "b": 2}, 3)
        bad = ndcg_at_k(["x", "a", "b"], {"a": 2, "b": 2}, 3)
        assert bad < good <= 1.0

    def test_never_exceeds_one_under_float_drift(self) -> None:
        # With equal grades, ideal and actual DCG are the same sum in a different
        # order, and float addition is not associative — an uncapped implementation
        # can return 1.0000000000000002 and print "better than ideal".
        relevant = {f"c{i}": 2 for i in range(40)}
        ranking = [f"c{i}" for i in range(40)]
        assert ndcg_at_k(ranking, relevant, 40) <= 1.0

    def test_an_unjudged_query_raises(self) -> None:
        with pytest.raises(ValueError, match="undefined"):
            ndcg_at_k(["a"], {}, 10)


class TestBootstrap:
    def test_the_interval_brackets_the_point_estimate(self) -> None:
        values = [0.0, 0.25, 0.5, 0.75, 1.0] * 8
        interval = bootstrap_interval(values)
        assert interval.low <= interval.point <= interval.high
        assert interval.point == pytest.approx(0.5)
        assert interval.n == 40

    def test_two_runs_with_the_same_seed_agree_exactly(self) -> None:
        # FR-E4: an unseeded bootstrap would move the interval on every invocation
        # for reasons unrelated to retrieval, and the report would never reproduce.
        values = [i / 30 for i in range(30)]
        first = bootstrap_interval(values)
        second = bootstrap_interval(values)
        assert (first.low, first.point, first.high) == (second.low, second.point, second.high)

    def test_a_different_seed_moves_the_interval_but_not_the_point(self) -> None:
        # Guards the claim that seeding is what makes it reproducible: if the seed
        # did nothing, the test above would be proving nothing.
        values = [i / 30 for i in range(30)]
        default = bootstrap_interval(values)
        other = bootstrap_interval(values, seed=BOOTSTRAP_SEED + 1)
        assert other.point == pytest.approx(default.point)
        assert (other.low, other.high) != (default.low, default.high)

    def test_a_larger_sample_gives_a_tighter_interval(self) -> None:
        # The property D69 rests on when it calls FULL a robustness check.
        pattern = [0.0, 1.0]
        small = bootstrap_interval(pattern * 25)
        large = bootstrap_interval(pattern * 125)
        assert large.half_width < small.half_width

    def test_a_single_observation_reports_no_spread_rather_than_crashing(self) -> None:
        interval = bootstrap_interval([0.4])
        assert (interval.low, interval.point, interval.high) == (0.4, 0.4, 0.4)

    def test_an_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty sample"):
            bootstrap_interval([])
