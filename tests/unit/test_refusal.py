"""D179's Rule 1 (`refusal/1.0.0`): refused, partial or answered, read off Rule 2's sentences.

The false-answer rate on unanswerable questions counts answered and partial, so a
partial read as refused would understate it and a refusal read as answered would
overstate it. Bar 1 is judged on the owner's labels; this rule is quoted beside
them only if it agrees with them (kappa at least 0.80, D179).
"""

from __future__ import annotations

from scholiast.application.use_cases.answer_question import NO_EVIDENCE_ANSWER
from scholiast.domain.claims import split_claims
from scholiast.eval.refusal import REFUSAL_VERSION, Refusal, label_answer


def _label(answer: str, *, structural: bool = False) -> Refusal:
    return label_answer(split_claims(answer), structural=structural)


class TestRuleOne:
    def test_the_version_is_the_one_d179_freezes(self) -> None:
        assert REFUSAL_VERSION == "refusal/1.0.0"

    def test_d60s_structural_refusal_is_refused_whatever_its_text(self) -> None:
        assert _label("The model is a CNN [C1].", structural=True) is Refusal.REFUSED

    def test_d60s_own_words_read_as_a_refusal_too(self) -> None:
        assert _label(NO_EVIDENCE_ANSWER) is Refusal.REFUSED

    def test_an_answer_that_only_says_the_evidence_does_not_answer_is_refused(self) -> None:
        answer = (
            "I couldn't find any information about the OSG dataset in the provided passages. "
            "The passages do not mention how it was built."
        )
        assert _label(answer) is Refusal.REFUSED

    def test_framing_beside_a_refusal_does_not_make_it_partial(self) -> None:
        answer = "Based on the evidence provided.\nThere is no mention of accents in the passages."
        assert _label(answer) is Refusal.REFUSED

    def test_a_refusal_followed_by_a_claim_is_partial(self) -> None:
        answer = (
            "The size is not explicitly stated in the provided passages. "
            "However, [C5] mentions that the train set has 7455 sentences."
        )
        assert _label(answer) is Refusal.PARTIAL

    def test_a_refusal_inside_a_claim_is_partial(self) -> None:
        answer = (
            "According to [C1], the dataset is a keyphrase dataset, but the passage does not "
            "provide further information on what it contains."
        )
        assert _label(answer) is Refusal.PARTIAL

    def test_an_answer_that_never_refuses_is_answered(self) -> None:
        assert _label("According to [C2], the dimension is 300. It is fixed.") is Refusal.ANSWERED

    def test_a_negative_answer_is_an_answer_not_a_refusal(self) -> None:
        assert _label("No, they do not report results only on English data [C3].") is (
            Refusal.ANSWERED
        )

    def test_an_answer_with_no_sentences_is_answered(self) -> None:
        """No refusal is stated, so D179's definition leaves it answered; it is not D60's."""
        assert _label("[C1]") is Refusal.ANSWERED
