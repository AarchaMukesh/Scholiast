"""Fitting the weighted-fusion weight (D76, doc 6 §6.6).

The property worth most here is not arithmetic — it is **where the weight is
allowed to come from**. A weight fitted on the evaluation queries would put a tuned
configuration against RRF's untuned constant and report the tuning as a measured
win, so `training_queries` is the seam that keeps the fitting set and the reported
set apart. That is the test that would catch the mistake nobody would notice in a
number.
"""

from __future__ import annotations

import pytest

from scholiast.application.ports.parsing import ParsedDocument
from scholiast.domain.paper import Chunk, ChunkRole, Provenance, SectionKind
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.eval.dret import DRet, Grade, Judgment, Query, Split
from scholiast.eval.fusion import (
    DEFAULT_WEIGHT_GRID,
    FusionFit,
    fit_dense_weight,
    score_weight,
    training_queries,
)
from scholiast.eval.harness import ArmHits
from scholiast.eval.qasper import CorpusPaper


def _chunk(chunk_id: str, paper_id: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=f"text of {chunk_id}",
        role=ChunkRole.CHILD,
        ordinal=0,
        provenance=Provenance(
            paper_id=paper_id,
            section_path="Results",
            section_kind=SectionKind.RESULTS,
            page=1,
            char_start=0,
            char_end=10,
        ),
        chunker_version="structure-aware/1.2.0",
        token_count=3,
        parent_id=f"{paper_id}:p0",
    )


def _paper(paper_id: str) -> CorpusPaper:
    return CorpusPaper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        document=ParsedDocument(sections=()),
        paragraph_index={},
        section_headings=frozenset(),
    )


def _dataset(*, judged_train: bool = True, with_train: bool = True) -> DRet:
    """One train paper and one eval paper, split the way D-RET splits: by paper."""
    queries = [Query("e1", "an evaluation question", "eval-paper", Split.EVAL, in_core=True)]
    judgments = [Judgment("e1", "eval-chunk", Grade.DIRECTLY_ANSWERS)]
    if with_train:
        queries.insert(0, Query("t1", "a training question", "train-paper", Split.TRAIN))
        if judged_train:
            judgments.append(Judgment("t1", "train-chunk", Grade.DIRECTLY_ANSWERS))
    return DRet(
        queries=queries,
        judgments=judgments,
        chunks={
            "train-chunk": _chunk("train-chunk", "train-paper"),
            "eval-chunk": _chunk("eval-chunk", "eval-paper"),
        },
        papers={"train-paper": _paper("train-paper"), "eval-paper": _paper("eval-paper")},
    )


def _hits(*, dense: list[str], lexical: list[str], relevant: list[str]) -> ArmHits:
    """One query's two arms, with descending synthetic scores on each side."""

    def ranked(ids: list[str], arm: RetrievalArm) -> tuple[ScoredChunk, ...]:
        return tuple(
            ScoredChunk(chunk_id=cid, paper_id="p", score=1.0 - i / 10, arm=arm, rank=i + 1)
            for i, cid in enumerate(ids)
        )

    return ArmHits(
        query_id="q",
        qrels={cid: int(Grade.DIRECTLY_ANSWERS) for cid in relevant},
        dense=ranked(dense, RetrievalArm.DENSE),
        lexical=ranked(lexical, RetrievalArm.LEXICAL),
    )


class TestTheFittingSetExcludesTheEvaluationQueries:
    def test_only_train_split_queries_are_offered_for_fitting(self) -> None:
        # The whole point of D76: CORE and FULL are the report, so a weight chosen
        # on them would measure its own tuning.
        queries = training_queries(_dataset())
        assert [q.id for q in queries] == ["t1"]
        assert all(q.split is Split.TRAIN for q in queries)

    def test_a_train_query_without_judgments_is_not_offered(self) -> None:
        # Nothing to fit against, and `score_weight` would raise on it later —
        # further from the cause than a filter here.
        with pytest.raises(ValueError, match="no train-split queries"):
            training_queries(_dataset(judged_train=False))

    def test_a_dataset_with_no_train_split_at_all_is_refused(self) -> None:
        # Distinct from the case above: here the split is missing rather than
        # unjudged. Falling back to the evaluation queries would be the one
        # unacceptable recovery, so there is no fallback at all — and the message
        # says why, because a future reader may be tempted to add one.
        with pytest.raises(ValueError, match="report its own tuning"):
            training_queries(_dataset(with_train=False))


class TestScoringOneWeight:
    def test_a_full_dense_weight_scores_the_dense_ranking(self) -> None:
        hits = [_hits(dense=["a", "gold"], lexical=["gold", "a"], relevant=["gold"])]
        score = score_weight(hits, 1.0, depth=10)
        assert score.mrr == pytest.approx(0.5)
        assert score.recall == pytest.approx(1.0)

    def test_a_full_lexical_weight_scores_the_lexical_ranking(self) -> None:
        hits = [_hits(dense=["a", "gold"], lexical=["gold", "a"], relevant=["gold"])]
        assert score_weight(hits, 0.0, depth=10).mrr == pytest.approx(1.0)

    def test_recall_is_capped_by_the_depth_it_was_scored_at(self) -> None:
        # Depth is part of the fit: a weight tuned on a 50-long list is not
        # necessarily the weight for a 10-long one.
        hits = [_hits(dense=["a", "b", "gold"], lexical=["a", "b", "gold"], relevant=["gold"])]
        assert score_weight(hits, 1.0, depth=2).recall == pytest.approx(0.0)
        assert score_weight(hits, 1.0, depth=3).recall == pytest.approx(1.0)

    def test_an_empty_fitting_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="empty fitting set"):
            score_weight([], 0.5, depth=10)


class TestFittingTheWeight:
    def test_it_chooses_the_weight_that_maximises_the_objective(self) -> None:
        # Lexical is right, dense is wrong, so the fit must lean lexical — i.e.
        # away from the dense arm, which is the direction a weight on dense means.
        hits = [_hits(dense=["wrong", "gold"], lexical=["gold", "wrong"], relevant=["gold"])]
        fit = fit_dense_weight(hits, depth=10, grid=(0.0, 0.5, 1.0))
        assert fit.weight == 0.0

    def test_the_curve_reports_every_weight_that_was_tried(self) -> None:
        # Without the curve a flat optimum and a sharp one look identical, and the
        # report would imply the choice was better determined than it was.
        hits = [_hits(dense=["gold"], lexical=["gold"], relevant=["gold"])]
        fit = fit_dense_weight(hits, depth=10)
        assert [score.weight for score in fit.curve] == list(DEFAULT_WEIGHT_GRID)

    def test_ties_keep_the_first_weight_on_the_grid(self) -> None:
        # Both arms identical, so every weight scores the same. FR-E4: the choice
        # must not depend on iteration order.
        hits = [_hits(dense=["gold", "a"], lexical=["gold", "a"], relevant=["gold"])]
        fit = fit_dense_weight(hits, depth=10, grid=(0.2, 0.4, 0.6))
        assert fit.weight == 0.2
        assert fit_dense_weight(hits, depth=10, grid=(0.2, 0.4, 0.6)).weight == fit.weight

    def test_it_reports_a_second_objective_that_disagrees(self) -> None:
        # Constructed so the two objectives genuinely pull apart: the first query's
        # dense arm puts the one relevant chunk at rank 1 (MRR likes dense), while
        # the second query's lexical arm surfaces BOTH relevant chunks inside the
        # depth (Recall likes lexical). D76 reports the disagreement rather than
        # quietly choosing whichever objective flatters the result.
        mrr_favours_dense = _hits(dense=["g", "a"], lexical=["a", "g"], relevant=["g"])
        recall_favours_lexical = _hits(
            dense=["g1", "a", "b"], lexical=["g1", "g2", "z"], relevant=["g1", "g2"]
        )
        fit = fit_dense_weight(
            [mrr_favours_dense, recall_favours_lexical], depth=2, grid=(0.0, 1.0)
        )
        assert fit.weight == 0.0, "the fitting objective is recall, which prefers lexical here"
        assert fit.mrr_weight == 1.0, "MRR prefers dense here, and that must be visible"
        assert not fit.objectives_agree

    def test_it_records_how_many_queries_it_was_fitted_on(self) -> None:
        hits = [_hits(dense=["gold"], lexical=["gold"], relevant=["gold"])] * 3
        assert fit_dense_weight(hits, depth=10).n_queries == 3

    def test_the_objective_is_named_rather_than_implied(self) -> None:
        hits = [_hits(dense=["gold"], lexical=["gold"], relevant=["gold"])]
        assert fit_dense_weight(hits, depth=10).objective == "recall@10"

    def test_an_empty_grid_is_refused(self) -> None:
        hits = [_hits(dense=["gold"], lexical=["gold"], relevant=["gold"])]
        with pytest.raises(ValueError, match="grid is empty"):
            fit_dense_weight(hits, depth=10, grid=())

    def test_agreement_is_computed_from_the_two_optima_not_assumed(self) -> None:
        agreeing = FusionFit(
            weight=0.3, curve=(), objective="recall@10", mrr_weight=0.3, n_queries=10
        )
        disagreeing = FusionFit(
            weight=0.3, curve=(), objective="recall@10", mrr_weight=0.7, n_queries=10
        )
        assert agreeing.objectives_agree
        assert not disagreeing.objectives_agree
