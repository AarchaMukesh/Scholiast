"""Conversation and turn invariants (§5.4.3, FR-Q6, FR-Q9).

Small module, two invariants, and both exist to stop a specific confusion in the
stored record rather than to police types.
"""

from __future__ import annotations

import pytest

from scholiast.domain.conversation import Conversation, StageTimings, Turn, TurnRole
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.retrieval import Citation, CitationStatus, RetrievalArm, ScoredChunk

EVIDENCE = (ScoredChunk(chunk_id="c-1", paper_id="p-1", score=0.8, arm=RetrievalArm.DENSE),)
CITATION = Citation(
    label="C1", chunk_id="c-1", claim_text="A claim.", status=CitationStatus.UNVERIFIED
)


class TestAQuestionCarriesNoEvidence:
    def test_a_user_turn_with_evidence_is_rejected(self) -> None:
        # Which turn the evidence belongs to has to be unambiguous: an evidence
        # panel that could be attached to either half of an exchange is not an
        # audit trail.
        with pytest.raises(InvariantViolation):
            Turn(role=TurnRole.USER, content="What is attention?", evidence=EVIDENCE)

    def test_a_user_turn_with_citations_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation):
            Turn(role=TurnRole.USER, content="What is attention?", citations=(CITATION,))

    def test_an_assistant_turn_may_carry_both(self) -> None:
        # Positive control: an invariant that rejected every turn would pass both
        # tests above and make the table unwritable.
        turn = Turn(
            role=TurnRole.ASSISTANT,
            content="Attention scales quadratically [C1].",
            evidence=EVIDENCE,
            citations=(CITATION,),
        )
        assert turn.evidence == EVIDENCE
        assert turn.citations == (CITATION,)

    def test_a_blank_question_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation):
            Turn(role=TurnRole.USER, content="   ")

    def test_a_blank_answer_is_allowed(self) -> None:
        # §5.11's last rung returns evidence with no synthesis, so an empty
        # assistant turn is a real state — rejecting it would make the
        # degradation ladder unrecordable.
        assert Turn(role=TurnRole.ASSISTANT, content="", degraded=("generation_unavailable",))


class TestDegradationIsVisibleOnTheTurn:
    def test_a_clean_turn_reports_none(self) -> None:
        assert not Turn(role=TurnRole.ASSISTANT, content="Answer.").was_degraded

    def test_a_flagged_turn_reports_it(self) -> None:
        turn = Turn(role=TurnRole.ASSISTANT, content="Answer.", degraded=("provider_failover",))
        assert turn.was_degraded


class TestStageTimings:
    def test_the_total_sums_the_stages_it_should(self) -> None:
        # `first_token_ms` is a *point* inside generation, not a stage beside it
        # — adding it to the total would double-count the same milliseconds.
        timings = StageTimings(
            retrieval_ms=40, assembly_ms=10, first_token_ms=300, generation_ms=900
        )
        assert timings.total_ms == 950

    def test_the_dict_carries_every_stage_for_the_message_row(self) -> None:
        # AP-8's breakdown is only as good as what reaches `messages.latency_ms`.
        keys = StageTimings().as_dict().keys()
        assert keys == {
            "retrieval_ms",
            "assembly_ms",
            "first_token_ms",
            "generation_ms",
            "total_ms",
        }


class TestConversation:
    def test_a_new_conversation_is_empty(self) -> None:
        assert Conversation(id="c", workspace_id="w").is_empty

    def test_a_conversation_with_a_turn_is_not(self) -> None:
        conversation = Conversation(
            id="c", workspace_id="w", turns=[Turn(role=TurnRole.USER, content="Hi")]
        )
        assert not conversation.is_empty
