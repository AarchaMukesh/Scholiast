"""D179's Rule 2 (`claims/1.0.0`): what one claim is, and which passages it cites.

Every groundedness, support and coverage rate in M5 phase 2 is counted over these
sentences, so each test names a way a miscount would move a reported number. The
answers here are written to the shapes the 167 train answers took (D179 builds the
rule on those and nothing else); none is an eval answer.
"""

from __future__ import annotations

import pytest

from scholiast.domain.claims import (
    CLAIMS_VERSION,
    OTHER_WORK_SUFFIX,
    Claim,
    ClaimKind,
    split_claims,
    states_refusal,
)
from scholiast.domain.retrieval import EvidenceSet, RetrievalArm, ScoredChunk, audit_citations


def _claims(answer: str, *headings: str) -> tuple[Claim, ...]:
    return split_claims(answer, headings=headings)


def _labels(answer: str, *headings: str) -> list[tuple[str, tuple[str, ...]]]:
    return [(c.source, c.labels) for c in _claims(answer, *headings)]


class TestTheRuleIsVersioned:
    def test_the_version_is_the_one_d179_freezes(self) -> None:
        assert CLAIMS_VERSION == "claims/1.0.0"


# --- a claim is a sentence ------------------------------------------------------------


class TestAClaimIsASentence:
    def test_each_sentence_is_one_claim(self) -> None:
        claims = _claims("The model is a CNN. It has 3 layers! Is it deep? Yes.")
        assert [c.source for c in claims] == [
            "The model is a CNN.",
            "It has 3 layers!",
            "Is it deep?",
            "Yes.",
        ]
        assert [c.index for c in claims] == [0, 1, 2, 3]

    def test_a_line_break_ends_a_sentence_without_a_full_stop(self) -> None:
        assert [c.source for c in _claims("Datasets used\nSST-2 and IMDb")] == [
            "Datasets used",
            "SST-2 and IMDb",
        ]

    def test_a_list_marker_is_not_a_claim_of_its_own(self) -> None:
        sources = [c.source for c in _claims("1. CTB5\n2. CTB6\n- UD1.4\n* MSRA\n\t+ OntoNotes")]
        assert sources == ["CTB5", "CTB6", "UD1.4", "MSRA", "OntoNotes"]

    @pytest.mark.parametrize(
        "sentence",
        [
            "Waseem et al. BIBREF5 collected 16k tweets.",
            "It is trained on news (e.g. Reuters) and Eq. 3 defines the loss.",
            "The corpus was built by J. Smith in 2014.",
            "They compare CNN vs. RNN models.",
        ],
    )
    def test_an_abbreviation_or_initial_does_not_end_a_sentence(self, sentence: str) -> None:
        assert [c.source for c in _claims(sentence)] == [sentence]

    def test_a_single_capital_before_a_full_stop_reads_as_an_initial(self) -> None:
        """The cost of not splitting "J. Smith": "option B. Then" stays one sentence."""
        assert len(_claims("They chose option B. Then they trained it.")) == 1

    def test_a_quotation_stays_in_the_sentence_that_introduces_it(self) -> None:
        answer = '[C6] states: "RC is central. Many datasets exist." This suggests a focus on RC.'
        assert [c.source for c in _claims(answer)] == [
            '[C6] states: "RC is central. Many datasets exist."',
            "This suggests a focus on RC.",
        ]

    def test_a_quotation_in_curly_quotes_stays_whole(self) -> None:
        answer = "It reads “RC is central. Data is scarce.” Then transfer follows."
        assert [c.source for c in _claims(answer)] == [
            "It reads “RC is central. Data is scarce.”",
            "Then transfer follows.",
        ]

    def test_windows_line_endings_split_like_any_other(self) -> None:
        assert _labels("A holds [C1].\r\nB holds [C2].") == [
            ("A holds [C1].", ("C1",)),
            ("B holds [C2].", ("C2",)),
        ]

    def test_an_empty_answer_has_no_claims(self) -> None:
        assert _claims("") == ()
        assert _claims("   \n\n  ") == ()


# --- which passage a claim cites ------------------------------------------------------


class TestALabelBelongsToItsSentence:
    def test_a_label_inside_a_sentence_is_that_sentences(self) -> None:
        assert _labels("A is B [C1]. C is D [C2].") == [
            ("A is B [C1].", ("C1",)),
            ("C is D [C2].", ("C2",)),
        ]

    def test_a_label_opening_a_sentence_is_that_sentences(self) -> None:
        assert _labels("Bert is used. [C2] The model uses X.") == [
            ("Bert is used.", ()),
            ("[C2] The model uses X.", ("C2",)),
        ]

    def test_a_label_standing_alone_after_a_sentence_attaches_to_it(self) -> None:
        """D179's own words: the model's habit of `claim. [C1]` cites the claim before."""
        assert _labels("The baseline is the LSTM-DQN baseline. [C1]") == [
            ("The baseline is the LSTM-DQN baseline.", ("C1",)),
        ]

    def test_a_label_line_ending_a_paragraph_attaches_to_the_sentence_before(self) -> None:
        assert _labels("A first claim.\nThe last claim.\n[C4]") == [
            ("A first claim.", ()),
            ("The last claim.", ("C4",)),
        ]

    def test_a_label_line_ending_a_paragraph_does_not_reach_the_next_paragraph(self) -> None:
        assert _labels("A claim.\n[C4]\n\nThe next paragraph.") == [
            ("A claim.", ("C4",)),
            ("The next paragraph.", ()),
        ]

    def test_grouped_and_repeated_labels_are_kept_once_in_order(self) -> None:
        assert _labels("X holds [C2, C1] and Y holds [C2][C3].") == [
            ("X holds [C2, C1] and Y holds [C2][C3].", ("C2", "C1", "C3")),
        ]

    def test_only_bracketed_labels_are_citations(self) -> None:
        """Layer 2 audits `[Cn]` alone; a bare C4 is prose, as it is to the validity count."""
        assert _labels("In C4, removing attention costs 2 points (C3).") == [
            ("In C4, removing attention costs 2 points (C3).", ()),
        ]

    def test_rule_2_sees_exactly_the_labels_layer_2_audits(self) -> None:
        answer = (
            "A holds [C1]. B holds [C2, C3].\n[C9] heads this.\nC holds. [C4][C1]\n"
            "In C5 and (C6), nothing is cited. D holds [C 7] and [c8]."
        )
        hits = [
            ScoredChunk(f"chunk-{n}", "p", 1.0 / n, RetrievalArm.DENSE, n) for n in range(1, 10)
        ]
        audited = {
            c.label for c in audit_citations(answer, EvidenceSet.assemble(hits, limit=9)).citations
        }
        cited = {label for claim in _claims(answer) for label in claim.labels}
        assert cited == audited == {"C1", "C2", "C3", "C4", "C9"}

    def test_an_answer_of_labels_alone_has_no_claims(self) -> None:
        assert _claims("[C1]") == ()
        assert _claims("[C1], [C2] and [C3].") == ()


class TestAnEchoedHeadingLabelsThePassageTextAfterIt:
    HEADING = "Bridging Pre-Training and Fine-Tuning · Our method ::: TCEN Architecture"

    def test_the_heading_is_removed_and_its_label_reaches_the_text_below(self) -> None:
        answer = f"[C3] {self.HEADING}\nThe attention module is pre-trained on ASR data."
        assert _labels(answer, self.HEADING) == [
            ("The attention module is pre-trained on ASR data.", ("C3",)),
        ]

    def test_a_heading_alone_in_its_paragraph_labels_the_next_paragraph_only(self) -> None:
        answer = (
            f"[C1] {self.HEADING} \n\nThey evaluate on German. They report 80.9 F1.\n\n"
            "Overall, the results are strong."
        )
        assert _labels(answer, self.HEADING) == [
            ("They evaluate on German.", ("C1",)),
            ("They report 80.9 F1.", ("C1",)),
            ("Overall, the results are strong.", ()),
        ]

    def test_each_heading_labels_only_its_own_block(self) -> None:
        other = "Paper B · Abstract"
        answer = (
            f"[C1] {self.HEADING}\nFirst passage text. More of it.\n\n"
            f"[C2] {other}\nSecond passage text.\n\nA summary of both."
        )
        assert _labels(answer, self.HEADING, other) == [
            ("First passage text.", ("C1",)),
            ("More of it.", ("C1",)),
            ("Second passage text.", ("C2",)),
            ("A summary of both.", ()),
        ]

    def test_a_heading_stops_where_the_next_heading_starts(self) -> None:
        other = "Paper B · Abstract"
        answer = f"[C1] {self.HEADING}\nFirst text.\n[C2] {other}\nSecond text."
        assert _labels(answer, self.HEADING, other) == [
            ("First text.", ("C1",)),
            ("Second text.", ("C2",)),
        ]

    def test_a_lone_heading_stops_at_the_next_paragraphs_own_heading(self) -> None:
        other = "Paper B · Abstract"
        answer = f"[C1] {self.HEADING}\n\nFirst text.\n[C2] {other}\nSecond text."
        assert _labels(answer, self.HEADING, other) == [
            ("First text.", ("C1",)),
            ("Second text.", ("C2",)),
        ]

    def test_a_heading_with_the_other_work_marker_is_removed_too(self) -> None:
        answer = f"[C2] {self.HEADING}{OTHER_WORK_SUFFIX}\nBIBREF3 proposed it."
        assert _labels(answer, self.HEADING) == [("BIBREF3 proposed it.", ("C2",))]

    def test_a_heading_inside_a_sentence_leaves_the_sentence(self) -> None:
        answer = f"The experiments, specifically [C1] {self.HEADING}, do not focus on one domain."
        (claim,) = _claims(answer, self.HEADING)
        assert claim.labels == ("C1",)
        assert self.HEADING not in claim.source
        assert claim.text == "The experiments, specifically, do not focus on one domain."

    def test_several_headings_on_one_line_label_what_follows(self) -> None:
        other = "Paper B · Abstract"
        answer = f"[C1] {self.HEADING}, [C7] {other}.\n\nThe models are CNNs."
        assert _labels(answer, self.HEADING, other) == [("The models are CNNs.", ("C1", "C7"))]

    def test_a_heading_with_nothing_after_it_attaches_to_the_sentence_before(self) -> None:
        answer = f"The answer is X.\n\n[C5] {self.HEADING}"
        assert _labels(answer, self.HEADING) == [("The answer is X.", ("C5",))]


class TestAColonIntroducesAList:
    def test_the_introduction_is_framing_and_its_labels_reach_its_items(self) -> None:
        answer = "According to [C6], the metrics are:\n1. BLEU\n2. ROUGE-L\nThey also report F1."
        claims = _claims(answer)
        assert [(c.source, c.labels, c.kind) for c in claims] == [
            ("According to [C6], the metrics are:", ("C6",), ClaimKind.FRAMING),
            ("BLEU", ("C6",), ClaimKind.CLAIM),
            ("ROUGE-L", ("C6",), ClaimKind.CLAIM),
            ("They also report F1.", (), ClaimKind.CLAIM),
        ]

    def test_a_fragment_item_is_read_as_the_end_of_its_introduction(self) -> None:
        claims = _claims("The datasets used are:\n\n* SST-2\n* IMDb (movie reviews)")
        assert [c.text for c in claims[1:]] == [
            "The datasets used are: SST-2",
            "The datasets used are: IMDb (movie reviews)",
        ]

    def test_an_item_that_is_a_sentence_stands_on_its_own(self) -> None:
        claims = _claims("We can compare the models:\n- The CRC model is more accurate.")
        assert claims[1].text == "The CRC model is more accurate."

    def test_a_blank_line_before_the_items_does_not_stop_the_labels(self) -> None:
        answer = "In [C3], the tasks are:\n\n* Task 1: sentiment\n* Task 2: rating"
        assert [c.labels for c in _claims(answer)] == [("C3",), ("C3",), ("C3",)]

    def test_the_first_line_that_is_not_an_item_ends_the_list(self) -> None:
        answer = "[C2] lists:\n- A\nB is unrelated."
        assert [c.labels for c in _claims(answer)] == [("C2",), ("C2",), ()]

    def test_a_colon_inside_a_line_introduces_nothing(self) -> None:
        (claim, item) = _claims("The criteria are: (a) fluency, (b) adequacy.\n- Extra")
        assert claim.kind is ClaimKind.CLAIM
        assert item.text == "Extra"


# --- which sentences are claims -------------------------------------------------------


class TestARefusalIsNotAClaim:
    @pytest.mark.parametrize(
        "sentence",
        [
            "There is no information in the provided evidence passages about the corpus size.",
            "I couldn't find any evidence that directly mentions the average title length.",
            "I cannot find any information about who judged the outputs.",
            "I can't answer your question based on the provided evidence.",
            "I cannot determine the most accurately predicted class from the evidence.",
            "The size of the corpus is not explicitly stated in the provided passages.",
            "The paper does not explicitly state the size of the descriptions.",
            "The passage you provided does not mention the goal of the study.",
            "The authors do not specify whether the representation is supervised.",
            "There is no direct mention of accents in the provided evidence passages.",
            "No statement in the provided evidence suggests the method transfers.",
            "None of the passages provided indicate any approach for Vietnamese.",
            "There is no passage that explains why the setting is appropriate.",
            "To fully answer the question, more information is needed.",
            "Without further information, it is difficult to determine the architecture.",
            "Therefore, it is unclear whether they incorporate WordNet.",
            "The question of which model works better is not directly answerable.",
            "There is no clear answer to what the meta knowledge is.",
            "[C5] No specific results for the benchmark are mentioned in the provided text.",
            "It does not provide further insight into why the proxy is good.",
            "The evidence provided does not support the claim that they use RNNs.",
            "The exact size is not specified.",
            "The quality of the dataset is not explicitly reported as high.",
            "Its definition is not available in the provided passages.",
            "I could not find anything in this workspace's papers that speaks to that question.",
            "The exact number of annotators cannot be determined from the passages.",
            "It is difficult to determine which architecture the authors adopt.",
        ],
    )
    def test_a_statement_that_the_evidence_does_not_answer_is_a_refusal(
        self, sentence: str
    ) -> None:
        (claim,) = _claims(sentence)
        assert (claim.kind, claim.refuses, claim.trivial) == (ClaimKind.REFUSAL, True, True)

    def test_a_curly_apostrophe_refuses_like_a_straight_one(self) -> None:
        assert states_refusal(f"I couldn{chr(0x2019)}t find any information about it.")

    @pytest.mark.parametrize(
        "sentence",
        [
            "There is no mention of Estonian, though.",
            "There is no mention of the corpus size in the passages, but alas.",
            "There is no mention of the corpus size in the passages, but sadly so.",
        ],
    )
    def test_a_short_trailing_clause_cannot_carry_a_claim(self, sentence: str) -> None:
        (claim,) = _claims(sentence)
        assert claim.kind is ClaimKind.REFUSAL

    def test_a_three_word_clause_can(self) -> None:
        (claim,) = _claims("There is no mention of the corpus size, but English is covered.")
        assert (claim.kind, claim.refuses) == (ClaimKind.CLAIM, True)

    @pytest.mark.parametrize(
        "sentence",
        [
            "According to [C1], the dataset is a keyphrase dataset, but the passage does not "
            "provide further information on what it contains.",
            "While the passage does not explicitly state the elements, it does mention "
            "musical expression as a consideration.",
            "The size is described as small, but the exact size is not specified.",
            "[C4] groups the models by objective; it does not specify an architecture.",
            "The answer is not explicitly stated, although several methods are listed.",
        ],
    )
    def test_a_sentence_that_refuses_and_claims_is_a_claim(self, sentence: str) -> None:
        (claim,) = _claims(sentence)
        assert (claim.kind, claim.refuses, claim.trivial) == (ClaimKind.CLAIM, True, False)

    @pytest.mark.parametrize(
        "sentence",
        [
            "Implying that the reward is not provided at each step.",
            "No, they do not report results only on English data.",
            "[C7] does not focus on multiple choice QA.",
            "The generator does not receive the reward at each step.",
            "They found no improvement over the baseline.",
            "No, they have not explicitly made any attempt to correct the gold standards.",
            "The training data is not included in the test set.",
            "The model does not compare favourably with BERT.",
        ],
    )
    def test_a_fact_stated_in_the_negative_is_not_a_refusal(self, sentence: str) -> None:
        (claim,) = _claims(sentence)
        assert (claim.kind, claim.refuses) == (ClaimKind.CLAIM, False)


class TestFramingIsNotAClaim:
    @pytest.mark.parametrize(
        "sentence",
        ["Based on the evidence provided.", "According to the passages:", "Based on the text,"],
    )
    def test_framing_alone_is_trivial(self, sentence: str) -> None:
        (claim,) = _claims(sentence)
        assert (claim.kind, claim.trivial) == (ClaimKind.FRAMING, True)

    def test_framing_before_content_leaves_a_claim(self) -> None:
        (claim,) = _claims("Based on the evidence, the model uses a CNN encoder.")
        assert claim.kind is ClaimKind.CLAIM


# --- what NLI reads -------------------------------------------------------------------


class TestTheNliTextIsTheClaimAlone:
    @pytest.mark.parametrize(
        ("source", "text"),
        [
            ("According to [C2], the dimension is 300.", "The dimension is 300."),
            ("[C5] mentions that the corpus has 10k words.", "The corpus has 10k words."),
            ("The passage [C4] states that X holds.", "X holds."),
            ("Passage [C6] explicitly states that Y holds.", "Y holds."),
            ("It uses RNNs, as mentioned in [C1], [C3], and [C7].", "It uses RNNs."),
            ("However, [C1] does mention that Y holds.", "However, Y holds."),
            ("[C1] The model is a CNN.", "The model is a CNN."),
            ("In [C2], they report results on German.", "They report results on German."),
            ("The F1 is 0.8 [C1, C2].", "The F1 is 0.8."),
        ],
    )
    def test_markers_and_attributions_are_removed(self, source: str, text: str) -> None:
        (claim,) = _claims(source)
        assert claim.text == text

    def test_the_source_keeps_what_the_model_wrote(self) -> None:
        (claim,) = _claims("According to [C2], the dimension is 300.")
        assert claim.source == "According to [C2], the dimension is 300."
