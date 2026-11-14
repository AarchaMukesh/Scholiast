"""D179's measurements, decided on hand-built answers (M5 phase 2, step B2).

Every number phase 2 reports is a count of claims, pairs or answers, so the ways
it goes wrong are miscounts: a refused answer contributing claims, a fabricated
label paired as if valid, an uncited claim dropped from the denominator, partial
answers missed as false answers, an interval that resamples questions instead of
papers. Each is pinned here without a model, a GPU or Qdrant.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

import pytest

from scholiast.application.ports.ml import Entailment
from scholiast.domain.claims import Claim, ClaimKind
from scholiast.eval.abstention import Instance
from scholiast.eval.answerability import Category
from scholiast.eval.answers import Answer, Passage, Reading
from scholiast.eval.grounding import (
    REGENERATED,
    Pair,
    compare_regenerated,
    count,
    nli_pairs,
    paper_intervals,
    rates,
    read_pairs,
    regeneration_sample,
    totals,
)
from scholiast.eval.refusal import Refusal

ANSWERABLE = Category.ANSWERABLE
QASPER = Category.UNANSWERABLE_QASPER
WITHHELD = Category.UNANSWERABLE_WITHHELD

DATASET: Any = types.SimpleNamespace(
    chunks={f"k{i}": types.SimpleNamespace(text=f"passage text {i}") for i in range(1, 9)}
)


def answer(
    key: str = "a",
    *,
    paper: str = "p1",
    category: Category = ANSWERABLE,
    labels: Sequence[str] = ("C1", "C2"),
    citations: Sequence[tuple[str, str | None, str]] = (),
    judged: Sequence[bool] = (),
    judged_total: int | None = None,
    text: str = "answer text",
) -> Answer:
    """`judged` marks which shown passages are evidence; `judged_total` is how much
    evidence the corpus holds (default: the shown ones), so evidence can exist unshown."""
    passages = tuple(
        Passage(label, f"k{label[1:]}", paper, 0.5, label_judged)
        for label, label_judged in zip(
            labels, list(judged) + [False] * (len(labels) - len(judged)), strict=True
        )
    )
    return Answer(
        key=key,
        question_id=f"q-{key}",
        paper_id=paper,
        split="eval",
        category=category,
        question="?",
        scope=(paper,),
        passages=passages,
        messages_sha256=None,
        prompt_chars=0,
        text=text,
        shown=text,
        citations=tuple(citations),
        provider="ollama",
        model="m",
        prompt_tokens=None,
        completion_tokens=None,
        retrieval_ms=0,
        generation_ms=0,
        structural_refusal=False,
        judged=sum(judged) if judged_total is None else judged_total,
    )


def claim(
    index: int, *labels: str, kind: ClaimKind = ClaimKind.CLAIM, text: str | None = None
) -> Claim:
    return Claim(
        index=index,
        source=f"source {index} " + " ".join(f"[{label}]" for label in labels),
        text=f"claim {index}" if text is None else text,
        labels=labels,
        kind=kind,
        refuses=kind is ClaimKind.REFUSAL,
    )


def reading(refusal: Refusal, *claims: Claim) -> Reading:
    return Reading(claims=claims, refusal=refusal)


def nli(key: str, index: int, label: str, *, supported: bool, truncated: bool = False) -> Any:
    entail = Entailment(
        entailment=0.9 if supported else 0.05,
        neutral=0.05 if supported else 0.9,
        contradiction=0.05,
        premise_truncated=truncated,
    )
    return {(key, index, label): entail}


class TestWhichClaimsArePaired:
    def test_a_refused_answer_makes_no_pairs(self) -> None:
        # A refusal states the evidence does not answer; it makes no claim to check.
        assert nli_pairs(answer(), reading(Refusal.REFUSED, claim(0, "C1")), DATASET) == []

    def test_a_partial_answer_is_paired(self) -> None:
        pairs = nli_pairs(answer(), reading(Refusal.PARTIAL, claim(0, "C1")), DATASET)
        assert [p.label for p in pairs] == ["C1"]

    def test_a_fabricated_label_makes_no_pair(self) -> None:
        # C9 was never shown: there is no passage to check the claim against.
        pairs = nli_pairs(answer(), reading(Refusal.ANSWERED, claim(0, "C1", "C9")), DATASET)
        assert [p.label for p in pairs] == ["C1"]

    def test_framing_and_refusal_sentences_are_not_paired(self) -> None:
        r = reading(
            Refusal.PARTIAL,
            claim(0, "C1", kind=ClaimKind.FRAMING),
            claim(1, "C1", kind=ClaimKind.REFUSAL),
            claim(2, "C2"),
        )
        assert [(p.claim_index, p.label) for p in nli_pairs(answer(), r, DATASET)] == [(2, "C2")]

    def test_the_premise_is_the_chunk_text_and_the_hypothesis_the_nli_text(self) -> None:
        # D179's clarification: D-RET text under the heading, and Rule 2's NLI text,
        # not the sentence as written with its markers.
        (pair,) = nli_pairs(answer(), reading(Refusal.ANSWERED, claim(0, "C2")), DATASET)
        assert pair.premise == "passage text 2"
        assert pair.hypothesis == "claim 0"
        assert pair.chunk_id == "k2"

    def test_a_claim_with_no_text_left_for_nli_is_not_paired(self) -> None:
        r = reading(Refusal.ANSWERED, claim(0, "C1", text="  "))
        assert nli_pairs(answer(), r, DATASET) == []

    async def test_readings_are_keyed_back_to_their_pairs(self) -> None:
        class Checker:
            version = "fake"

            async def entails(self, premises: Sequence[str], claims: Sequence[str]) -> list[Any]:
                return [
                    Entailment(float(p.endswith("2")), 0.0, 0.0, False)
                    for p, _ in zip(premises, claims, strict=True)
                ]

        pairs = [
            Pair("a", 0, "C1", "k1", "passage text 1", "x"),
            Pair("a", 0, "C2", "k2", "passage text 2", "x"),
        ]
        got = await read_pairs(pairs, Checker())
        assert not got[("a", 0, "C1")].supported
        assert got[("a", 0, "C2")].entailment == 1.0


class TestTheClaimCounts:
    def test_an_uncited_claim_is_in_the_denominator_and_unsupported(self) -> None:
        # D179: an uncited claim counts as unsupported. Dropping it would inflate
        # groundedness exactly when coverage is poor.
        a = answer()
        r = reading(Refusal.ANSWERED, claim(0, "C1"), claim(1))
        c = count(a, r, nli("a", 0, "C1", supported=True))
        assert (c.claims, c.covered_claims, c.supported_claims) == (2, 1, 1)
        got = rates([c])
        assert got["answerable/groundedness"] == pytest.approx(0.5)
        assert got["answerable/citation_coverage"] == pytest.approx(0.5)

    def test_a_claim_is_grounded_if_any_cited_passage_entails_it(self) -> None:
        a = answer()
        r = reading(Refusal.ANSWERED, claim(0, "C1", "C2"))
        found = nli("a", 0, "C1", supported=False) | nli("a", 0, "C2", supported=True)
        c = count(a, r, found)
        assert (c.supported_claims, c.pairs, c.supported_pairs) == (1, 2, 1)
        got = rates([c])
        assert got["answerable/groundedness"] == 1
        assert got["answerable/citation_support_precision"] == pytest.approx(0.5)

    def test_a_claim_citing_only_a_fabricated_label_is_uncovered(self) -> None:
        c = count(answer(), reading(Refusal.ANSWERED, claim(0, "C9")), {})
        assert (c.claims, c.covered_claims, c.pairs) == (1, 0, 0)

    def test_a_refused_answer_contributes_no_claims(self) -> None:
        c = count(answer(), reading(Refusal.REFUSED, claim(0, "C1")), {})
        assert c.claims == 0

    def test_citation_validity_reads_layer_twos_statuses(self) -> None:
        a = answer(
            citations=[
                ("C1", "k1", "unverified"),
                ("C9", None, "fabricated"),
                ("C2", "k2", "unverified"),
            ]
        )
        c = count(a, reading(Refusal.ANSWERED, claim(0, "C1")), {})
        assert (c.labels_emitted, c.labels_resolving) == (3, 2)
        assert rates([c])["answerable/citation_validity"] == pytest.approx(2 / 3)

    def test_hallucination_counts_answers_with_any_unsupported_claim(self) -> None:
        good = count(
            answer("g"),
            reading(Refusal.ANSWERED, claim(0, "C1")),
            nli("g", 0, "C1", supported=True),
        )
        bad = count(
            answer("b"),
            reading(Refusal.ANSWERED, claim(0, "C1"), claim(1, "C2")),
            nli("b", 0, "C1", supported=True) | nli("b", 1, "C2", supported=False),
        )
        refused = count(answer("r"), reading(Refusal.REFUSED), {})
        # The refused answer makes no claims, so it is not in the denominator.
        assert rates([good, bad, refused])["answerable/hallucination"] == pytest.approx(0.5)

    def test_truncated_pairs_are_counted(self) -> None:
        c = count(
            answer(),
            reading(Refusal.ANSWERED, claim(0, "C1")),
            nli("a", 0, "C1", supported=True, truncated=True),
        )
        assert totals([c])["truncated_pairs"] == 1


class TestTheRefusalRates:
    def test_a_partial_answer_to_an_unanswerable_question_is_a_false_answer(self) -> None:
        # D179: false-answer rate counts answered OR partial.
        rows = [
            count(answer("1", category=QASPER), reading(Refusal.PARTIAL), {}),
            count(answer("2", category=QASPER), reading(Refusal.REFUSED), {}),
            count(answer("3", category=WITHHELD), reading(Refusal.ANSWERED), {}),
            count(answer("4", category=WITHHELD), reading(Refusal.ANSWERED), {}),
        ]
        got = rates(rows)
        assert got["false_answer"] == pytest.approx(0.75)
        assert got["false_answer/unanswerable-qasper"] == pytest.approx(0.5)
        assert got["false_answer/unanswerable-withheld"] == 1

    def test_answerable_refusals_are_split_by_whether_the_evidence_was_shown(self) -> None:
        # Refusing when the question's evidence was never shown is correct behaviour.
        rows = [
            count(answer("1", judged=[True]), reading(Refusal.REFUSED), {}),
            count(answer("2", judged=[True]), reading(Refusal.ANSWERED), {}),
            count(answer("3", judged=[False], judged_total=2), reading(Refusal.REFUSED), {}),
            # No evidence in the corpus at all: its own group, never "not shown".
            count(answer("4"), reading(Refusal.ANSWERED), {}),
        ]
        got = rates(rows)
        assert got["refused/evidence_shown"] == pytest.approx(0.5)
        assert got["refused/evidence_not_shown"] == 1
        assert got["refused/evidence_no_judged_evidence"] == 0
        assert got["refused"] == pytest.approx(0.5)

    def test_unanswerable_false_answers_have_their_own_claim_metrics(self) -> None:
        # Reported, not judged, and never mixed into the answerable numbers.
        un = count(
            answer("u", category=WITHHELD),
            reading(Refusal.ANSWERED, claim(0, "C1")),
            nli("u", 0, "C1", supported=True),
        )
        an = count(
            answer("a"),
            reading(Refusal.ANSWERED, claim(0, "C1")),
            nli("a", 0, "C1", supported=False),
        )
        got = rates([un, an])
        assert got["unanswerable_false/groundedness"] == 1
        assert got["answerable/groundedness"] == 0

    def test_an_empty_population_is_undefined_not_zero(self) -> None:
        assert rates([])["false_answer"] is None


class TestTheIntervalsResamplePapers:
    def test_one_papers_questions_move_together(self) -> None:
        # Two answers about one paper: every resample draws both or neither, so the
        # interval collapses to the point. Resampling answers would not.
        rows = [
            count(answer("1", category=QASPER), reading(Refusal.ANSWERED), {}),
            count(answer("2", category=QASPER), reading(Refusal.REFUSED), {}),
        ]
        got = paper_intervals(rows, iterations=200)
        assert got["false_answer"]["point"] == pytest.approx(0.5)
        assert got["false_answer"]["ci95"] == [0.5, 0.5]
        assert got["papers"]["resampled"] == 1

    def test_the_interval_is_seeded(self) -> None:
        rows = [
            count(answer(str(i), paper=f"p{i}", category=QASPER), reading(r), {})
            for i, r in enumerate([Refusal.ANSWERED, Refusal.REFUSED] * 5)
        ]
        assert paper_intervals(rows, iterations=200) == paper_intervals(rows, iterations=200)
        assert paper_intervals(rows, iterations=200)["papers"]["seed"] == 20260915

    def test_a_resample_without_a_category_is_counted_undefined(self) -> None:
        rows = [count(answer("a", paper="p1"), reading(Refusal.REFUSED), {})] + [
            count(answer(str(i), paper=f"q{i}", category=WITHHELD), reading(Refusal.REFUSED), {})
            for i in range(5)
        ]
        got = paper_intervals(rows, iterations=200)
        assert got["refused"]["undefined_resamples"] > 0
        assert got["false_answer"]["undefined_resamples"] < 200


def instance(question_id: str) -> Instance:
    return Instance(
        question_id=question_id,
        paper_id="p",
        split="eval",
        category=ANSWERABLE,
        size=10,
        draw=0,
        question="?",
        scope=("p",),
    )


class TestTheRegeneration:
    def test_twenty_are_chosen_by_hash_whatever_the_order(self) -> None:
        todo = [instance(f"q{i}") for i in range(60)]
        first = regeneration_sample(todo)
        assert len(first) == REGENERATED
        assert {i.key for i in regeneration_sample(list(reversed(todo)))} == {i.key for i in first}

    def test_changes_are_counted_by_text_and_by_rule_one(self) -> None:
        a, b = answer("x", text="one"), answer("x", text="two")
        same = answer("y", text="same")
        got = compare_regenerated(
            [(a, reading(Refusal.ANSWERED)), (same, reading(Refusal.ANSWERED))],
            [(b, reading(Refusal.REFUSED)), (same, reading(Refusal.ANSWERED))],
        )
        assert (got["text_changed"], got["refusal_label_changed"]) == (1, 1)
        assert got["changes"] == [{"key": "x", "first": "answered", "again": "refused"}]

    def test_a_regeneration_compared_against_the_wrong_answer_is_refused(self) -> None:
        with pytest.raises(ValueError, match="regenerated"):
            compare_regenerated(
                [(answer("x"), reading(Refusal.ANSWERED))],
                [(answer("y"), reading(Refusal.ANSWERED))],
            )
