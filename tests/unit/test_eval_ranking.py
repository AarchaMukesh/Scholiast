"""FR-D4's measurement holds still: the fit, the held-out score, the dense scores (D151-D153).

What fails silently here is a weight edited by hand while the log says it was
fitted, a fit that peeked at the held-out searches, a ranking change that moves the
reported nDCG without anyone re-recording it, and dense scores from one encoder
fitting the weights while production ranks with another. Offline: the dense scores
are committed (D152), so nothing here loads a model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.ml import Precision
from scholiast.config.settings import Settings
from scholiast.domain import ranking
from scholiast.domain.ranking import RankingWeights, RelevanceSignal
from scholiast.eval import ranking as measure
from scholiast.infrastructure.ml.embedder import LocalEmbedder


@cache
def _searches() -> tuple[measure.Search, ...]:
    return measure.load_searches()


@cache
def _dense() -> dict[str, Any]:
    return measure.load_dense()


def _split(name: str) -> list[measure.Search]:
    return [s for s in _searches() if s.split == name]


class TestTheSearches:
    def test_are_a3s_twenty_queries_split_by_topic(self) -> None:
        assert (len(_split("train")), len(_split("held_out"))) == (10, 10)
        for s in _searches():
            odd = int(s.topic.removeprefix("t")) % 2 == 1
            assert s.split == ("train" if odd else "held_out"), s.query_id

    def test_every_work_has_a_grade_and_a_dense_score(self) -> None:
        for s in _searches():
            assert set(s.judged) <= set(s.grades), s.query_id
            scores = _dense()["scores"][s.query_id]
            for w in s.works:
                assert measure.record_key(w.add) in scores, s.query_id


class TestTheDenseScores:
    def test_come_from_the_encoder_production_runs(self) -> None:
        settings = Settings()
        production = LocalEmbedder(
            settings.embedding_model,
            device=settings.embedding_device,
            precision=Precision(settings.embedding_precision),
        ).version

        assert _dense()["embedder_version"] == production, (
            "the weights were fitted on another encoder's scores; rerun "
            "`eval --ranking --recompute-dense`"
        )

    def test_another_encoders_scores_are_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "dense.json"
        path.write_text(json.dumps({**_dense(), "embedder_version": "other/1"}), encoding="utf-8")

        with pytest.raises(ValueError, match="recompute-dense"):
            measure.load_dense(path, expect_version="BAAI/bge-base-en-v1.5/fp16/1.0.0")


class TestTheFit:
    def test_the_shipped_ranker_is_the_one_the_training_searches_fit(self) -> None:
        signal, weights, _ = measure.fit(_split("train"), _dense())

        assert (signal, weights) == (ranking.SHIPPED_SIGNAL, ranking.SHIPPED_WEIGHTS), (
            "the shipped ranker is not D151's fit; re-run `eval --ranking` and record "
            "the change rather than editing a weight by hand"
        )

    def test_covers_d151s_grid_for_both_signals(self) -> None:
        _, _, grid = measure.fit(_split("train"), _dense())

        assert len(grid) == 2 * 66
        assert {r.signal for r in grid} == {"lexical", "dense"}

    def test_never_reads_a_held_out_search(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        real_fit = measure.fit

        def spy(
            searches: Sequence[measure.Search], dense: Any
        ) -> tuple[RelevanceSignal, RankingWeights, tuple[measure.GridRow, ...]]:
            seen.extend(s.split for s in searches)
            return real_fit(searches, dense)

        monkeypatch.setattr(measure, "fit", spy)
        measure.run_ranking()

        assert seen and set(seen) == {"train"}


class TestTheHeldOutScore:
    def test_is_the_one_d153_records(self) -> None:
        arms = measure.held_out_arms(
            _split("held_out"), ranking.SHIPPED_SIGNAL, ranking.SHIPPED_WEIGHTS, _dense()
        )

        means = {name: round(arm.mean.point, 4) for name, arm in arms.items()}
        assert means == {
            "today's order": 0.873,
            "lexical only": 0.9052,
            "dense only": 0.9348,
            "fitted": 0.9264,
        }

    def test_the_gain_over_todays_order_is_the_one_d153_records(self) -> None:
        report = measure.run_ranking()

        delta = report.deltas["fitted - today's order"]
        assert (round(delta.point, 4), round(delta.low, 4), round(delta.high, 4)) == (
            0.0534,
            0.0033,
            0.1001,
        )


class TestTheMetric:
    def test_a_paper_shown_twice_counts_once(self) -> None:
        # Two works of one search that belong to one judged work are one paper seen
        # twice, not two relevant papers: counting both would reward an unmerged
        # duplicate.
        # Grades below the cap: counted twice, the partial paper would gain.
        grades = {"a": 1, "b": 2}

        twice = measure._ndcg(["a", "a", "b"], grades)
        once = measure._ndcg(["a", "b"], grades)

        assert twice == pytest.approx(once)
        assert once < 1.0
