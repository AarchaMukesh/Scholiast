"""D177's abstention scoring, held to the rule and the clarifications fixed before it.

What could go wrong quietly here is never the retrieval (production's calls, made
unchanged) but the arithmetic and plumbing around it: a question asked in the
wrong scope, a withheld question asked where nobody checked, five draws outvoting
one question, a threshold one step too low, a tie broken the wrong way, an
interval that resamples instances instead of papers, or a run that writes to the
collection it measures. Each class below pins one of those.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.ml import TextRole
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk
from scholiast.eval import abstention as ab
from scholiast.eval import answerability, cli
from scholiast.eval.answerability import Category, withheld_scope
from scholiast.eval.dret import DRet, build_dret
from scholiast.eval.gate import scope_for
from scholiast.eval.snapshot import load_snapshot
from tests.unit.test_eval_gate import _corpus

ANS = Category.ANSWERABLE
QASPER = Category.UNANSWERABLE_QASPER
WITHHELD = Category.UNANSWERABLE_WITHHELD


def row(
    category: Category,
    question_id: str,
    *,
    paper: str | None = None,
    split: str = "train",
    size: int = 10,
    draw: int = 0,
    dense: float | None = 0.5,
    lexical: float | None = 0.5,
    rerank: float | None = 0.5,
) -> ab.Scored:
    return ab.Scored(
        key=f"{category.value}:{question_id}:{size}:{draw}",
        question_id=question_id,
        paper_id=paper or f"paper-{question_id}",
        split=split,
        category=category,
        size=size,
        draw=draw,
        scores={"dense": dense, "lexical": lexical, "rerank": rerank},
    )


def singles(category: Category, scores: Sequence[float | None], **how: Any) -> list[ab.Scored]:
    """One instance per question, each its own paper: weight 1 apiece."""
    return [row(category, f"{category.value}-{i}", dense=s, **how) for i, s in enumerate(scores)]


# --- what is asked where, on the real sets ---------------------------------------


@pytest.fixture(scope="module")
def dataset() -> DRet:
    return build_dret(load_snapshot())


@pytest.fixture(scope="module")
def real_sets() -> dict[str, Any]:
    return answerability.load_sets()


@pytest.fixture(scope="module")
def asked(real_sets: dict[str, Any], dataset: DRet) -> list[ab.Instance]:
    return ab.instances(real_sets, dataset)


class TestEachQuestionIsAskedWhereD177Says:
    def test_the_counts_are_fifteen_scopes_per_question_and_two_per_withheld(
        self, asked: list[ab.Instance]
    ) -> None:
        # 867 answerable and 56 QASPER-unanswerable at 3 sizes x 5 draws; 100
        # withheld at sizes 5 and 10 only.
        assert len(asked) == (867 + 56) * 15 + 100 * 2
        assert len({i.key for i in asked}) == len(asked)

    def test_a_withheld_question_is_never_asked_at_twenty_five(
        self, asked: list[ab.Instance]
    ) -> None:
        # D177 (2026-10-28): 15 of those papers were never checked by hand.
        sizes = {i.size for i in asked if i.category is WITHHELD}
        assert sizes == {5, 10}

    def test_answerable_and_qasper_questions_use_d168s_scopes(
        self, asked: list[ab.Instance], dataset: DRet
    ) -> None:
        papers = sorted(dataset.papers)
        for i in asked:
            if i.category is not WITHHELD:
                assert i.paper_id in i.scope
                expected = scope_for(i.question_id, i.paper_id, papers, size=i.size, draw=i.draw)
                assert i.scope == expected

    def test_a_withheld_question_is_asked_only_inside_its_checked_scope(
        self, asked: list[ab.Instance], real_sets: dict[str, Any]
    ) -> None:
        checked = {
            q["question_id"]: set(q["scope"])
            for q in real_sets["questions"]
            if q["category"] == WITHHELD.value
        }
        for i in asked:
            if i.category is WITHHELD:
                assert i.paper_id not in i.scope
                assert len(i.scope) == i.size
                assert set(i.scope) <= checked[i.question_id]

    def test_a_withheld_scope_the_corpus_no_longer_draws_is_refused(
        self, real_sets: dict[str, Any], dataset: DRet
    ) -> None:
        edited = json.loads(json.dumps(real_sets))
        entry = next(q for q in edited["questions"] if q["category"] == WITHHELD.value)
        entry["scope"] = sorted(dataset.papers)[:10]
        with pytest.raises(ValueError, match="not the scope the hand-check judged"):
            ab.instances(edited, dataset)


# --- weights ----------------------------------------------------------------------


class TestEveryQuestionCountsOnce:
    def test_five_draws_of_one_question_weigh_what_one_withheld_question_weighs(self) -> None:
        rows = [row(QASPER, "q", draw=d) for d in range(5)] + [row(WITHHELD, "w")]
        weights = ab.weighted(rows)
        assert sum(w for r, w in weights if r.category is QASPER) == sum(
            w for r, w in weights if r.category is WITHHELD
        )

    def test_the_pooled_false_answer_rate_counts_questions_not_instances(self) -> None:
        # One QASPER question answered in all five draws, four withheld questions
        # refused: one question of five was answered falsely. Per instance it
        # would be five of nine.
        rows = [row(QASPER, "q", draw=d, dense=0.9) for d in range(5)]
        rows += [row(WITHHELD, f"w{i}", dense=0.1) for i in range(4)]
        rows += [row(ANS, "a", dense=0.9)]
        assert ab.rates(ab.weighted(rows), "dense", 0.5).false_answer == Fraction(1, 5)

    def test_a_score_exactly_at_tau_is_answered(self) -> None:
        # D177: answered means "at or above tau". Eval applies a train score as
        # tau, so an eval instance landing on it exactly is not hypothetical.
        rows = [row(ANS, "a", dense=0.5), row(QASPER, "u", dense=0.5)]
        measured = ab.rates(ab.weighted(rows), "dense", 0.5)
        assert measured.answer_recall == 1 and measured.false_answer == 1

    def test_each_category_has_its_own_rate_beside_the_pooled_one(self) -> None:
        rows = [row(QASPER, "q", dense=0.9), row(WITHHELD, "w", dense=0.1), row(ANS, "a")]
        measured = ab.rates(ab.weighted(rows), "dense", 0.5)
        assert measured.by_category == {QASPER: Fraction(1), WITHHELD: Fraction(0)}


# --- the threshold ------------------------------------------------------------------


def _train_for_threshold() -> list[ab.Scored]:
    answerable = singles(ANS, [0.99, 0.9, 0.88, 0.86, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    unanswerable = singles(QASPER, [0.95, 0.85, 0.5, 0.4, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
    return answerable + unanswerable


class TestTauIsTheLowestScoreInsideTheLimit:
    def test_it_is_the_lowest_observed_score_at_most_one_in_ten_false(self) -> None:
        # At 0.86 one unanswerable of ten (0.95) is answered: exactly the limit.
        # At 0.85 a second one is, which breaks it.
        fit = ab.fit_threshold(ab.weighted(_train_for_threshold()), "dense")
        assert fit.tau == 0.86
        assert fit.answer_recall == Fraction(4, 10)
        assert fit.false_answer == Fraction(1, 10)

    def test_a_rate_exactly_at_the_limit_is_allowed(self) -> None:
        rows = singles(ANS, [0.9, 0.8]) + singles(QASPER, [0.95] + [0.1] * 9)
        assert ab.fit_threshold(ab.weighted(rows), "dense").tau == 0.8

    def test_one_draw_past_the_limit_is_not(self) -> None:
        # 0.9 answers 1 of 9 unanswerable questions: 0.111 > 0.10.
        rows = singles(ANS, [0.9, 0.8]) + singles(QASPER, [0.95] + [0.1] * 8)
        fit = ab.fit_threshold(ab.weighted(rows), "dense")
        assert fit.tau == math.inf and fit.answer_recall == 0

    def test_a_missing_score_is_never_answered(self) -> None:
        rows = singles(ANS, [0.9, None, 0.7]) + singles(QASPER, [None] * 5)
        fit = ab.fit_threshold(ab.weighted(rows), "dense")
        assert fit.tau == 0.7
        assert fit.answer_recall == Fraction(2, 3)
        assert fit.false_answer == 0

    def test_a_missing_score_ranks_below_a_negative_one(self) -> None:
        # Cross-encoder scores are logits and go negative: "nothing found" must
        # still rank below them, not at zero.
        rows = singles(ANS, [-2.0, None]) + singles(QASPER, [-3.0] * 10)
        fit = ab.fit_threshold(ab.weighted(rows), "dense")
        assert fit.tau == -2.0
        assert fit.answer_recall == Fraction(1, 2)

    def test_it_reads_only_the_signal_it_is_asked_for(self) -> None:
        rows = [
            row(ANS, "a", dense=0.9, lexical=0.1),
            row(QASPER, "u", dense=0.1, lexical=0.9),
        ]
        assert ab.fit_threshold(ab.weighted(rows), "dense").answer_recall == 1
        assert ab.fit_threshold(ab.weighted(rows), "lexical").answer_recall == 0

    def test_a_split_without_both_classes_is_refused(self) -> None:
        with pytest.raises(ValueError, match="answerable and unanswerable"):
            ab.fit_threshold(ab.weighted(singles(ANS, [0.5])), "dense")


def _fits(**recalls: float) -> dict[str, ab.Fit]:
    return {
        signal: ab.Fit(signal, 0.5, Fraction(recall).limit_denominator(10_000), Fraction(0))
        for signal, recall in recalls.items()
    }


class TestTheSignalIsChosenByD177sRule:
    def test_the_highest_recall_wins(self) -> None:
        assert ab.choose(_fits(dense=0.40, lexical=0.60, rerank=0.55)) == "lexical"

    def test_the_dense_cosine_wins_within_a_hundredth(self) -> None:
        assert ab.choose(_fits(dense=0.595, lexical=0.60, rerank=0.30)) == "dense"

    def test_exactly_a_hundredth_behind_is_still_within(self) -> None:
        assert ab.choose(_fits(dense=0.59, lexical=0.60, rerank=0.30)) == "dense"

    def test_more_than_a_hundredth_behind_loses(self) -> None:
        assert ab.choose(_fits(dense=0.589, lexical=0.60, rerank=0.30)) == "lexical"

    def test_an_exact_tie_between_the_others_goes_to_d177s_order(self) -> None:
        assert ab.choose(_fits(dense=0.1, lexical=0.6, rerank=0.6)) == "lexical"


def _held_out(false_answer: Fraction | None) -> ab.Rates:
    return ab.Rates(Fraction(1, 2), false_answer, {})


class TestTheOutcomesFixedInAdvance:
    def test_no_signal_at_half_recall_adopts_no_tau(self) -> None:
        verdict = ab.decide(_fits(dense=0.49, lexical=0.3, rerank=0.2), "dense", _held_out(None))
        assert verdict["outcome"] == "no tau adopted"

    def test_exactly_half_recall_reaches_the_floor(self) -> None:
        verdict = ab.decide(
            _fits(dense=0.5, lexical=0.3, rerank=0.2), "dense", _held_out(Fraction(1, 10))
        )
        assert verdict["outcome"] == "adopt"

    def test_a_tie_break_onto_a_signal_below_the_floor_goes_to_the_owner(self) -> None:
        fits = _fits(dense=0.495, lexical=0.505, rerank=0.2)
        assert ab.choose(fits) == "dense"
        verdict = ab.decide(fits, "dense", _held_out(Fraction(0)))
        assert verdict["outcome"] == "owner decides"
        assert "lexical" in verdict["reason"]

    def test_an_eval_false_answer_rate_of_exactly_a_fifth_is_adopted(self) -> None:
        verdict = ab.decide(_fits(dense=0.6), "dense", _held_out(Fraction(1, 5)))
        assert verdict["outcome"] == "adopt"

    def test_above_a_fifth_nothing_is_wired_in(self) -> None:
        verdict = ab.decide(_fits(dense=0.6), "dense", _held_out(Fraction(21, 100)))
        assert verdict["outcome"] == "owner decides"

    def test_an_undefined_eval_rate_is_not_a_pass(self) -> None:
        verdict = ab.decide(_fits(dense=0.6), "dense", _held_out(None))
        assert verdict["outcome"] == "owner decides"


# --- the report's other numbers ------------------------------------------------------


class TestTheAucIsAWeightedMannWhitney:
    def test_perfect_separation_is_one(self) -> None:
        rows = ab.weighted(singles(ANS, [0.9, 0.8]) + singles(QASPER, [0.2, 0.1]))
        assert ab.auc(rows, "dense", (QASPER,)) == 1.0

    def test_reversed_separation_is_zero(self) -> None:
        rows = ab.weighted(singles(ANS, [0.1, 0.2]) + singles(QASPER, [0.8, 0.9]))
        assert ab.auc(rows, "dense", (QASPER,)) == 0.0

    def test_a_tie_counts_half(self) -> None:
        rows = ab.weighted(singles(ANS, [0.5]) + singles(QASPER, [0.5]))
        assert ab.auc(rows, "dense", (QASPER,)) == 0.5

    def test_two_missing_scores_tie(self) -> None:
        rows = ab.weighted(singles(ANS, [None]) + singles(QASPER, [None]))
        assert ab.auc(rows, "dense", (QASPER,)) == 0.5

    def test_a_question_counts_once_whatever_its_draws(self) -> None:
        # Five draws of a QASPER question that beats the answerable one, and one
        # withheld question that does not: half the questions, not five sixths.
        rows = [row(QASPER, "q", draw=d, dense=0.9) for d in range(5)]
        rows += [row(WITHHELD, "w", dense=0.1), row(ANS, "a", dense=0.5)]
        assert ab.auc(ab.weighted(rows), "dense", ab.UNANSWERABLE) == 0.5

    def test_an_empty_side_has_no_auc(self) -> None:
        assert ab.auc(ab.weighted(singles(ANS, [0.5])), "dense", (WITHHELD,)) is None


class TestTheCurve:
    def test_every_observed_score_is_a_point_and_recall_only_grows_downward(self) -> None:
        points = ab.curve(ab.weighted(_train_for_threshold()), "dense")
        assert points["tau"][0] == 0.99
        assert points["tau"] == sorted(points["tau"], reverse=True)
        recalls = [r for r in points["answer_recall"] if r is not None]
        assert recalls == sorted(recalls)
        assert points["answer_recall"][-1] == 1.0
        assert points["false_answer"][-1] == 1.0


class TestIntervalsResamplePapers:
    def test_two_papers_that_disagree_span_everything_between(self) -> None:
        # Paper A's five draws are all answered, paper B's all refused. Resampling
        # papers can draw A twice or B twice; resampling the ten instances would
        # almost never stray far from one half.
        rows = [row(ANS, "a", paper="A", draw=d, dense=0.9) for d in range(5)]
        rows += [row(ANS, "b", paper="B", draw=d, dense=0.1) for d in range(5)]
        rows += [row(QASPER, "u", paper="A", dense=0.1)]
        out = ab.paper_intervals(ab.weighted(rows), "dense", 0.5)
        assert out["answer_recall"]["point"] == 0.5
        assert out["answer_recall"]["ci95"] == [0.0, 1.0]
        assert out["papers"]["resampled"] == 2

    def test_a_resample_missing_a_category_is_counted_not_imputed(self) -> None:
        rows = [
            row(ANS, "a", paper="A"),
            row(QASPER, "u", paper="B"),
            row(WITHHELD, "w", paper="C"),
        ]
        out = ab.paper_intervals(ab.weighted(rows), "dense", 0.5)
        assert out["false_answer/unanswerable-withheld"]["undefined_resamples"] > 0
        assert out["answer_recall"]["undefined_resamples"] > 0

    def test_two_runs_give_the_same_interval(self) -> None:
        rows = ab.weighted(_train_for_threshold())
        assert ab.paper_intervals(rows, "dense", 0.5) == ab.paper_intervals(rows, "dense", 0.5)


class TestTheAnalysis:
    def test_eval_applies_the_train_tau_unchanged(self) -> None:
        train = _train_for_threshold()
        held = [
            row(ANS, "e1", split="eval", dense=0.87),
            row(ANS, "e2", split="eval", dense=0.85),
            row(QASPER, "e3", split="eval", dense=0.9),
            row(WITHHELD, "e4", split="eval", dense=0.1),
        ]
        out = ab.analyse(train + held)
        assert out["chosen"]["tau"] == 0.86
        assert out["eval"]["answer_recall"]["point"] == 0.5
        assert out["eval"]["false_answer"]["point"] == 0.5
        assert out["verdict"]["outcome"] == "no tau adopted"

    def test_only_the_ten_paper_scopes_decide(self) -> None:
        train = _train_for_threshold()
        # At size 5 every unanswerable question outscores every answerable one.
        noise = singles(ANS, [0.0] * 5, size=5) + singles(QASPER, [1.0] * 5, size=5)
        held = [row(ANS, "e", split="eval"), row(QASPER, "u", split="eval", dense=0.0)]
        assert ab.analyse(train + held)["chosen"] == ab.analyse(train + held + noise)["chosen"]
        sizes = ab.analyse(train + held + noise)["sizes"]
        assert sizes["N=10"]["deciding"] and not sizes["N=5"]["deciding"]
        assert not sizes["N=25"]["withheld_reported"]


# --- scoring, on stand-ins ------------------------------------------------------------


@dataclass
class ScoreIndex:
    """Scores each chunk from a table per arm, keeping only the scope's papers."""

    dataset: DRet
    dense: dict[str, float]
    lexical: dict[str, float]
    calls: list[str] = field(default_factory=list)
    scopes: list[tuple[str, ...]] = field(default_factory=list)
    exists: bool = True
    points: int | None = None

    def _hits(
        self, table: dict[str, float], paper_ids: Sequence[str], limit: int, arm: RetrievalArm
    ) -> list[ScoredChunk]:
        allowed = set(paper_ids)
        ranked = sorted(
            (c for c in table if self.dataset.chunks[c].provenance.paper_id in allowed),
            key=lambda c: -table[c],
        )[:limit]
        return [
            ScoredChunk(
                chunk_id=c,
                paper_id=self.dataset.chunks[c].provenance.paper_id,
                score=table[c],
                arm=arm,
                rank=rank,
            )
            for rank, c in enumerate(ranked, start=1)
        ]

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append("search")
        self.scopes.append(tuple(paper_ids))
        return self._hits(self.dense, paper_ids, limit, RetrievalArm.DENSE)

    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.calls.append("search_lexical")
        self.scopes.append(tuple(paper_ids))
        return self._hits(self.lexical, paper_ids, limit, RetrievalArm.LEXICAL)

    async def existing_count(self) -> int | None:
        self.calls.append("existing_count")
        if not self.exists:
            return None
        return len(self.dataset.chunks) if self.points is None else self.points

    async def ensure_ready(self) -> None:
        self.calls.append("ensure_ready")


class CountingEmbedder:
    version = "BAAI/bge-base-en-v1.5/fp16/1.0.0"

    def __init__(self) -> None:
        self.texts: list[list[str]] = []

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        assert role is TextRole.QUERY
        self.texts.append(list(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


class LengthReranker:
    """The first passage scores highest: its score is the shortlist's length."""

    version = "stub-reranker/fp16/1.0.0"

    def __init__(self) -> None:
        self.shortlists: list[list[str]] = []

    async def rerank(
        self, query: str, passages: Sequence[str], *, top_k: int | None = None
    ) -> list[tuple[int, float]]:
        self.shortlists.append(list(passages))
        return [(i, float(len(passages) - i)) for i in range(len(passages))]


def _world(papers: int = 30) -> tuple[DRet, ScoreIndex]:
    dataset = _corpus(papers, [("q0", "p00", False)])
    chunks = sorted(dataset.chunks)
    index = ScoreIndex(
        dataset,
        dense={c: 0.30 + i / 100 for i, c in enumerate(chunks)},
        lexical={c: 9.0 - i / 10 for i, c in enumerate(chunks)},
    )
    return dataset, index


def _instance(scope: tuple[str, ...], question: str = "what is it?", key: str = "q") -> ab.Instance:
    return ab.Instance(key, scope[0], "train", ANS, len(scope), 0, question, scope)


class TestEachSignalIsReadFromItsScope:
    async def test_the_tops_come_from_the_scope_only(self) -> None:
        dataset, index = _world()
        embedder, reranker = CountingEmbedder(), LengthReranker()
        scope = ("p03", "p05")
        scored = await ab.score_one(
            _instance(scope),
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=embedder,
            reranker=reranker,
            vectors={},
        )
        # Out-of-scope chunks score higher on dense (p29) and lexical (p00).
        assert scored.scores["dense"] == pytest.approx(0.35)
        assert scored.scores["lexical"] == pytest.approx(8.7)
        assert index.scopes == [scope, scope]

    async def test_the_cross_encoder_reranks_the_fused_shortlist_with_its_header(self) -> None:
        dataset, index = _world()
        # Each arm finds a different half of the scope, so only their fusion
        # holds all ten.
        index.dense = {c: s for c, s in index.dense.items() if c < "c-p05"}
        index.lexical = {c: s for c, s in index.lexical.items() if "c-p05" <= c < "c-p10"}
        reranker = LengthReranker()
        scope = tuple(f"p{i:02d}" for i in range(10))
        scored = await ab.score_one(
            _instance(scope),
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=CountingEmbedder(),
            reranker=reranker,
            vectors={},
        )
        (shortlist,) = reranker.shortlists
        assert len(shortlist) == 10
        assert all(passage.startswith("[Paper: Paper p") for passage in shortlist)
        assert scored.scores["rerank"] == 10.0

    async def test_an_empty_scope_has_no_scores_and_asks_no_model(self) -> None:
        dataset, index = _world()
        index.dense, index.lexical = {}, {}
        reranker = LengthReranker()
        scored = await ab.score_one(
            _instance(("p01",)),
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=CountingEmbedder(),
            reranker=reranker,
            vectors={},
        )
        assert scored.scores == {"dense": None, "lexical": None, "rerank": None}
        assert reranker.shortlists == []

    async def test_a_question_is_embedded_once_and_alone(self) -> None:
        dataset, index = _world()
        embedder = CountingEmbedder()
        todo = [_instance((f"p{i:02d}",), key=f"q{i}") for i in range(4)]
        todo.append(_instance(("p09",), question="another?", key="q9"))
        await ab.score_all(
            todo,
            done={},
            sink=lambda _row: None,
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=embedder,
            reranker=LengthReranker(),
        )
        assert embedder.texts == [["what is it?"], ["another?"]]


class TestAnInterruptedRunResumes:
    async def test_rows_already_written_are_reused_and_new_ones_sunk_in_order(self) -> None:
        dataset, index = _world()
        todo = [_instance((f"p{i:02d}",), key=f"q{i}") for i in range(3)]
        earlier = ab.Scored.of(todo[1], {"dense": 0.1, "lexical": 0.2, "rerank": 0.3})
        sunk: list[ab.Scored] = []
        rows = await ab.score_all(
            todo,
            done={earlier.key: earlier},
            sink=sunk.append,
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=CountingEmbedder(),
            reranker=LengthReranker(),
        )
        assert [r.key for r in rows] == [i.key for i in todo]
        assert rows[1] is earlier
        assert [r.key for r in sunk] == [todo[0].key, todo[2].key]
        assert index.calls.count("search") == 2

    def test_the_progress_file_holds_each_row_as_soon_as_it_is_written(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "progress.jsonl"
        first = row(ANS, "a")
        with ab.progress_writer(path) as sink:
            sink(first)
            # Read while still open: a standby here must not lose the row.
            assert ab.read_progress(path) == {first.key: first}
            sink(row(QASPER, "u"))
        assert len(ab.read_progress(path)) == 2

    def test_a_torn_last_line_is_forgiven(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.jsonl"
        good = row(ANS, "a")
        path.write_text(json.dumps(good.to_json()) + "\n" + '{"key": "ans', encoding="utf-8")
        assert ab.read_progress(path) == {good.key: good}

    def test_a_torn_line_in_the_middle_is_not(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.jsonl"
        good = json.dumps(row(ANS, "a").to_json())
        path.write_text(f'{{"key": "ans\n{good}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            ab.read_progress(path)

    def test_no_file_means_nothing_done(self, tmp_path: Path) -> None:
        assert ab.read_progress(tmp_path / "absent.jsonl") == {}


class TestTheReproducibilityCheck:
    def test_the_sample_is_twenty_fixed_ten_paper_instances(self, asked: list[ab.Instance]) -> None:
        sample = ab.reproduce_sample(asked)
        assert len(sample) == 20 and {i.size for i in sample} == {10}
        assert sample == ab.reproduce_sample(list(reversed(asked)))

    def test_identical_scores_differ_by_nothing(self) -> None:
        rows = [row(ANS, "a", rerank=None)]
        assert ab.largest_differences(rows, rows) == {"dense": 0.0, "lexical": 0.0, "rerank": 0.0}

    def test_the_largest_difference_is_reported(self) -> None:
        first = [row(ANS, "a", dense=0.5), row(ANS, "b", dense=0.5)]
        second = [row(ANS, "a", dense=0.25), row(ANS, "b", dense=0.75 + 1e-9)]
        assert ab.largest_differences(first, second)["dense"] == pytest.approx(0.25 + 1e-9)

    def test_found_once_and_missing_once_is_not_a_number(self) -> None:
        differences = ab.largest_differences([row(ANS, "a")], [row(ANS, "a", lexical=None)])
        assert differences["lexical"] is None


# --- the run, end to end on stand-ins ------------------------------------------------


def _sets_for(dataset: DRet) -> dict[str, Any]:
    papers = sorted(dataset.papers)

    def entry(qid: str, paper: str, split: str, category: Category) -> dict[str, Any]:
        out = {
            "question_id": qid,
            "paper_id": paper,
            "split": split,
            "category": category.value,
            "question": f"question {qid}",
        }
        if category is WITHHELD:
            out |= {"withheld_paper": paper, "scope": list(withheld_scope(qid, paper, papers))}
        return out

    questions = [entry(f"t{i}", f"p{i:02d}", "train", ANS) for i in range(4)]
    questions += [entry("tu", "p05", "train", QASPER), entry("t0", "p00", "train", WITHHELD)]
    questions += [entry(f"e{i}", f"p{10 + i:02d}", "eval", ANS) for i in range(2)]
    questions += [entry("eu", "p13", "eval", QASPER), entry("e0", "p10", "eval", WITHHELD)]
    return {"set_version": answerability.SET_VERSION, "sheet_sha256": "s", "questions": questions}


@pytest.fixture
def run_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    dataset, index = _world()
    embedder, reranker = CountingEmbedder(), LengthReranker()
    sets_path = tmp_path / "sets.json"
    sets_path.write_text("{}", encoding="utf-8")
    archive = {
        "blocks": {
            cli.GATE_DENSE_BLOCK: {"collection": "eval", "embedder_version": embedder.version},
            cli.GATE_HYBRID_BLOCK: {
                "collection": "eval_hybrid",
                "embedder_version": embedder.version,
            },
        }
    }

    def build(_dataset: DRet, _spec: Any, *, sparse: bool = False) -> tuple[Any, Any, str]:
        return embedder, index, "eval_hybrid" if sparse else "eval"

    monkeypatch.setattr(cli, "_load_reference", lambda _path: archive)
    monkeypatch.setattr(cli, "load_sets", lambda: _sets_for(dataset))
    monkeypatch.setattr(answerability, "SETS", sets_path)
    monkeypatch.setattr(cli, "_build_dependencies", build)
    monkeypatch.setattr(cli, "_build_reranker", lambda: reranker)
    monkeypatch.setattr(cli, "load_snapshot", lambda: [])
    monkeypatch.setattr(cli, "build_dret", lambda _rows: dataset)
    monkeypatch.setattr(cli, "verify_split", lambda _d: None)
    monkeypatch.setattr(cli, "verify_core", lambda _d: None)
    return {"index": index, "archive": archive, "dir": tmp_path, "embedder": embedder}


class TestTheRun:
    async def test_it_scores_every_instance_and_reads_the_store_only(
        self, run_world: dict[str, Any]
    ) -> None:
        report = await cli.run_abstention(progress_dir=run_world["dir"])
        index: ScoreIndex = run_world["index"]
        assert "ensure_ready" not in index.calls
        assert index.calls.count("existing_count") == 1
        # 6 answerable + 2 QASPER questions x 15 scopes, 2 withheld x 2, then 20 re-scored.
        assert report.config["instances"] == 8 * 15 + 2 * 2
        assert index.calls.count("search") == report.config["instances"] + 20
        assert report.blocks["reproducibility"]["largest_abs_difference"] == {
            "dense": 0.0,
            "lexical": 0.0,
            "rerank": 0.0,
        }
        assert report.config["llm_calls"] == 0
        json.loads(report.to_json())

    async def test_a_rerun_resumes_from_its_progress_file(self, run_world: dict[str, Any]) -> None:
        first = await cli.run_abstention(progress_dir=run_world["dir"])
        run_world["index"].calls.clear()
        second = await cli.run_abstention(progress_dir=run_world["dir"])
        assert second.config["resumed_rows"] == second.config["instances"]
        assert run_world["index"].calls.count("search") == 20
        assert second.blocks["instances"] == first.blocks["instances"]

    async def test_a_missing_collection_is_refused_without_being_created(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["index"].exists = False
        with pytest.raises(ValueError, match="never creates one"):
            await cli.run_abstention(progress_dir=run_world["dir"])
        assert run_world["index"].calls == ["existing_count"]

    async def test_a_partial_collection_is_refused_not_built(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["index"].points = 7
        with pytest.raises(ValueError, match="holds 7 of 30 chunks"):
            await cli.run_abstention(progress_dir=run_world["dir"])
        assert run_world["index"].calls == ["existing_count"]

    async def test_a_collection_m2_did_not_measure_is_refused_before_any_qdrant_call(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["archive"]["blocks"][cli.GATE_HYBRID_BLOCK]["collection"] = "other"
        with pytest.raises(ValueError, match="collections M2's arms read"):
            await cli.run_abstention(progress_dir=run_world["dir"])
        assert run_world["index"].calls == []

    async def test_a_progress_file_from_other_models_is_not_reused(
        self, run_world: dict[str, Any]
    ) -> None:
        await cli.run_abstention(progress_dir=run_world["dir"])
        run_world["embedder"].version = "BAAI/bge-base-en-v1.5/fp32/1.0.0"
        for block in run_world["archive"]["blocks"].values():
            block["embedder_version"] = run_world["embedder"].version
        report = await cli.run_abstention(progress_dir=run_world["dir"])
        assert report.config["resumed_rows"] == 0


class TestTheCommand:
    def test_dense_beside_abstention_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--abstention", "--dense"])

    def test_it_archives_as_its_own_kind(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reports = run_world["dir"] / "reports"
        original = cli.run_abstention

        async def here() -> cli.EvalReport:
            return await original(progress_dir=run_world["dir"])

        monkeypatch.setattr(cli, "run_abstention", here)
        assert cli.main(["--abstention", "--report-dir", str(reports)]) == 0
        (archived,) = reports.glob("abstention-*.json")
        assert json.loads(archived.read_text(encoding="utf-8"))["blocks"]["verdict"]["outcome"]
