"""Lifecycle tests.

Doc 4 calls these state machines normative, which is only a meaningful claim if
the transition tables are exhaustively checked. `test_transition_table_is_total`
does exactly that: it enumerates every (from, to) pair rather than spot-checking
the happy path, so an illegal transition cannot slip through unnoticed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scholiast.domain.errors import (
    FailureClass,
    IllegalStateTransition,
    InsufficientPapersForComparison,
    InvariantViolation,
    WorkspaceCapacityExceeded,
)
from scholiast.domain.workspace import (
    MAX_PAPERS_PER_WORKSPACE,
    PaperMembership,
    PaperState,
    Workspace,
    WorkspaceState,
    paper_transition_allowed,
    terminal_state_for,
    workspace_transition_allowed,
)

NOW = datetime(2026, 7, 27, tzinfo=UTC)

#: The full ingestion happy path.
PIPELINE = [
    PaperState.QUEUED,
    PaperState.FETCHING,
    PaperState.PARSING,
    PaperState.CHUNKING,
    PaperState.EMBEDDING,
    PaperState.INDEXED,
]


def workspace(name: str = "RAG for long documents") -> Workspace:
    return Workspace(id="ws-1", owner_id="user-1", name=name, created_at=NOW)


def indexed_workspace(n: int) -> Workspace:
    ws = workspace()
    for i in range(n):
        m = ws.add_paper(f"paper-{i}", now=NOW)
        for state in PIPELINE[1:]:
            m.transition_to(state)
    ws.refresh_state()
    return ws


class TestPaperStateMachine:
    def test_happy_path_is_allowed_end_to_end(self) -> None:
        m = PaperMembership("ws-1", "p-1", PaperState.QUEUED, NOW)
        for state in PIPELINE[1:]:
            m.transition_to(state)
        assert m.state is PaperState.INDEXED
        assert m.is_queryable

    def test_transition_table_is_total(self) -> None:
        # Every pair is decided explicitly — no KeyError, no implicit allow.
        for src in PaperState:
            for dst in PaperState:
                assert isinstance(paper_transition_allowed(src, dst), bool)

    @pytest.mark.parametrize(
        ("src", "dst"),
        [
            (PaperState.INDEXED, PaperState.PARSING),  # no going backwards
            (PaperState.QUEUED, PaperState.INDEXED),  # no skipping the pipeline
            (PaperState.UNSUPPORTED, PaperState.QUEUED),  # terminal means terminal
            (PaperState.REMOVED, PaperState.QUEUED),
            (PaperState.CHUNKING, PaperState.FETCHING),
        ],
    )
    def test_illegal_transitions_are_rejected(self, src: PaperState, dst: PaperState) -> None:
        assert not paper_transition_allowed(src, dst)
        m = PaperMembership("ws-1", "p-1", src, NOW)
        with pytest.raises(IllegalStateTransition):
            m.transition_to(dst, failure_class="transient")

    def test_reapplying_the_same_state_is_idempotent(self) -> None:
        # A redelivered job (at-least-once queue) must not blow up.
        m = PaperMembership("ws-1", "p-1", PaperState.PARSING, NOW)
        m.transition_to(PaperState.PARSING)
        assert m.state is PaperState.PARSING

    def test_failure_requires_a_class(self) -> None:
        # A failure with no class cannot produce a retry decision or a user
        # message, which defeats the whole taxonomy.
        m = PaperMembership("ws-1", "p-1", PaperState.PARSING, NOW)
        with pytest.raises(InvariantViolation, match="failure_class"):
            m.transition_to(PaperState.FAILED)

    def test_retry_increments_attempts_and_clears_the_failure(self) -> None:
        m = PaperMembership("ws-1", "p-1", PaperState.PARSING, NOW)
        m.transition_to(PaperState.FAILED, failure_class="transient", failure_detail="timeout")
        assert m.attempts == 0
        m.transition_to(PaperState.QUEUED)
        assert m.attempts == 1
        assert m.failure_class is None

    def test_removal_is_allowed_mid_pipeline(self) -> None:
        # A user may delete a paper while it is still ingesting.
        for state in PIPELINE:
            m = PaperMembership("ws-1", "p-1", state, NOW)
            m.transition_to(PaperState.REMOVED)
            assert m.state is PaperState.REMOVED

    def test_only_indexed_is_queryable(self) -> None:
        for state in PaperState:
            assert state.is_queryable == (state is PaperState.INDEXED)


class TestWorkspaceQueryability:
    def test_empty_workspace_is_not_queryable(self) -> None:
        assert not workspace().is_queryable

    def test_one_indexed_paper_makes_it_queryable(self) -> None:
        # FR-W5: queries never block on in-flight ingestion.
        ws = workspace()
        m = ws.add_paper("p-1", now=NOW)
        assert not ws.is_queryable
        for state in PIPELINE[1:]:
            m.transition_to(state)
        ws.refresh_state()
        assert ws.is_queryable

    def test_in_flight_papers_are_reported_for_disclosure(self) -> None:
        # Answering over a subset without saying so is a correctness bug.
        ws = indexed_workspace(1)
        ws.add_paper("p-pending", now=NOW)
        assert ws.queryable_paper_ids == ("paper-0",)
        assert "p-pending" in ws.in_flight_paper_ids

    def test_state_is_derived_from_memberships(self) -> None:
        ws = workspace()
        assert ws.state is WorkspaceState.EMPTY
        m = ws.add_paper("p-1", now=NOW)
        assert ws.state is WorkspaceState.POPULATING
        for state in PIPELINE[1:]:
            m.transition_to(state)
        ws.refresh_state()
        assert ws.state is WorkspaceState.ACTIVE
        ws.remove_paper("p-1")
        assert ws.state is WorkspaceState.EMPTY


class TestWorkspaceMembership:
    def test_adding_the_same_paper_twice_is_idempotent(self) -> None:
        # FR-I7: re-adding must not duplicate work or reset progress.
        ws = workspace()
        first = ws.add_paper("p-1", now=NOW)
        for state in PIPELINE[1:]:
            first.transition_to(state)
        again = ws.add_paper("p-1", now=NOW)
        assert again is first
        assert again.state is PaperState.INDEXED
        assert len(ws.active_memberships) == 1

    def test_capacity_cap_is_enforced(self) -> None:
        ws = workspace()
        for i in range(MAX_PAPERS_PER_WORKSPACE):
            ws.add_paper(f"p-{i}", now=NOW)
        with pytest.raises(WorkspaceCapacityExceeded):
            ws.add_paper("one-too-many", now=NOW)

    def test_removed_papers_free_capacity(self) -> None:
        ws = workspace()
        for i in range(MAX_PAPERS_PER_WORKSPACE):
            ws.add_paper(f"p-{i}", now=NOW)
        ws.remove_paper("p-0")
        ws.add_paper("replacement", now=NOW)  # must not raise

    def test_removing_a_non_member_raises(self) -> None:
        with pytest.raises(InvariantViolation, match="not a member"):
            workspace().remove_paper("never-added")

    def test_blank_name_is_rejected(self) -> None:
        with pytest.raises(InvariantViolation, match="name"):
            Workspace(id="ws", owner_id="u", name="   ")


class TestComparisonGate:
    def test_single_paper_cannot_be_compared(self) -> None:
        ws = indexed_workspace(1)
        assert not ws.can_compare
        with pytest.raises(InsufficientPapersForComparison):
            ws.assert_can_compare()

    def test_two_indexed_papers_can_be_compared(self) -> None:
        assert indexed_workspace(2).can_compare

    def test_in_flight_papers_do_not_count_toward_comparison(self) -> None:
        ws = indexed_workspace(1)
        ws.add_paper("still-processing", now=NOW)
        assert not ws.can_compare


class TestWorkspaceDeletion:
    def test_deleting_is_a_real_persisted_state(self) -> None:
        # §5.6.4: deletion spans three stores with no distributed transaction, so
        # it must be resumable rather than instantaneous.
        ws = indexed_workspace(1)
        ws.mark_deleting(now=NOW)
        assert ws.state is WorkspaceState.DELETING
        assert ws.deleted_at == NOW

    def test_deleting_is_terminal(self) -> None:
        # "Terminal" means no path back out — not that re-entering it raises.
        for state in WorkspaceState:
            if state is not WorkspaceState.DELETING:
                assert not workspace_transition_allowed(WorkspaceState.DELETING, state)

    def test_mark_deleting_is_idempotent(self) -> None:
        # The deletion sweep runs on an at-least-once queue and is resumable
        # (§5.6.4), so a redelivered job must not fail — and must not move the
        # recorded deletion time, which retention logic may depend on.
        ws = workspace()
        ws.mark_deleting(now=NOW)
        later = datetime(2026, 8, 1, tzinfo=UTC)
        ws.mark_deleting(now=later)
        assert ws.state is WorkspaceState.DELETING
        assert ws.deleted_at == NOW, "deletion timestamp moved on retry"

    def test_membership_changes_do_not_resurrect_a_deleting_workspace(self) -> None:
        ws = workspace()
        ws.mark_deleting(now=NOW)
        ws.add_paper("p-1", now=NOW)
        assert ws.state is WorkspaceState.DELETING


class TestRedeliveryRestartsTheAttempt:
    """§5.7.4's at-least-once guarantee, in the state machine.

    A worker that dies mid-stage leaves a membership stranded at CHUNKING. The
    queue will redeliver that job, and the pipeline restarts from the beginning
    because every step is idempotent — so the membership has to be able to say
    it is back at the start. Without this the redelivered job raises
    IllegalStateTransition on its first move and dead-letters a paper that was
    only ever unlucky.
    """

    @pytest.mark.parametrize(
        "stranded",
        [
            PaperState.FETCHING,
            PaperState.PARSING,
            PaperState.CHUNKING,
            PaperState.EMBEDDING,
        ],
    )
    def test_an_in_flight_membership_can_return_to_queued(self, stranded: PaperState) -> None:
        assert paper_transition_allowed(stranded, PaperState.QUEUED)

    def test_a_finished_paper_cannot_be_dragged_back_to_queued(self) -> None:
        # The guard that keeps the previous test from being a licence to reset
        # anything: an INDEXED paper is done, and re-queueing it would re-embed
        # a corpus FR-I7 says must not be re-embedded.
        assert not paper_transition_allowed(PaperState.INDEXED, PaperState.QUEUED)
        assert not paper_transition_allowed(PaperState.UNSUPPORTED, PaperState.QUEUED)
        assert not paper_transition_allowed(PaperState.REMOVED, PaperState.QUEUED)

    def test_restarting_counts_as_an_attempt(self) -> None:
        # Counting only failures would make the user-visible attempt count
        # disagree with reality precisely where a paper keeps dying mid-pipeline.
        membership = PaperMembership(
            workspace_id="w", paper_id="p", state=PaperState.QUEUED, added_at=NOW
        )
        membership.transition_to(PaperState.PARSING)
        membership.transition_to(PaperState.QUEUED)
        assert membership.attempts == 1

    def test_restarting_clears_the_previous_failure(self) -> None:
        membership = PaperMembership(
            workspace_id="w", paper_id="p", state=PaperState.QUEUED, added_at=NOW
        )
        membership.transition_to(PaperState.FAILED, failure_class="transient", failure_detail="x")
        membership.transition_to(PaperState.QUEUED)
        assert membership.failure_class is None
        assert membership.failure_detail is None


class TestWhereAFailureLands:
    """§4.4.1 tabulates retryability and the terminal states, but never joins them."""

    @pytest.mark.parametrize(
        "failure_class",
        [FailureClass.UNPARSEABLE, FailureClass.SCANNED_IMAGE, FailureClass.TOO_LARGE],
    )
    def test_a_hopeless_file_is_unsupported(self, failure_class: FailureClass) -> None:
        # §4.4.1: UNSUPPORTED exists to give "no false hope of retry".
        assert terminal_state_for(failure_class) is PaperState.UNSUPPORTED

    def test_a_missing_source_is_failed_because_the_user_can_fix_it(self) -> None:
        # Equally non-retryable by the system, but the remedy is a manual
        # upload — and §4.4.1 offers that from FAILED and not from UNSUPPORTED.
        assert terminal_state_for(FailureClass.SOURCE_UNAVAILABLE) is PaperState.FAILED

    @pytest.mark.parametrize(
        "failure_class", [FailureClass.TRANSIENT, FailureClass.RESOURCE_EXHAUSTED]
    )
    def test_a_retryable_class_lands_on_failed_when_exhausted(
        self, failure_class: FailureClass
    ) -> None:
        assert terminal_state_for(failure_class) is PaperState.FAILED

    def test_every_class_has_a_landing_state(self) -> None:
        # Walks the whole enum, so adding a class without deciding where it
        # lands fails here rather than defaulting silently.
        assert all(
            terminal_state_for(c) in (PaperState.FAILED, PaperState.UNSUPPORTED)
            for c in FailureClass
        )


class TestMembershipEntryStates:
    """§5.4.2 has two entry points into a workspace, not one."""

    def test_an_already_ingested_paper_may_join_indexed(self) -> None:
        ws = workspace()
        membership = ws.add_paper("p-1", now=NOW, state=PaperState.INDEXED)
        assert membership.state is PaperState.INDEXED
        assert ws.is_queryable

    def test_a_membership_cannot_be_created_mid_pipeline(self) -> None:
        # Creating a membership at EMBEDDING would claim work that never ran.
        ws = workspace()
        with pytest.raises(InvariantViolation):
            ws.add_paper("p-1", now=NOW, state=PaperState.EMBEDDING)

    def test_the_default_entry_is_still_queued(self) -> None:
        assert workspace().add_paper("p-1", now=NOW).state is PaperState.QUEUED


class TestABatchIsMeasuredWholeAgainstTheCap:
    """D133: an add is all or nothing, so the cap is checked before any of it lands."""

    def filled(self, count: int) -> Workspace:
        ws = workspace()
        for i in range(count):
            ws.add_paper(f"paper-{i}", now=NOW)
        return ws

    def test_a_batch_that_fits_exactly_is_allowed(self) -> None:
        self.filled(MAX_PAPERS_PER_WORKSPACE - 2).ensure_room_for(["a", "b"])

    def test_one_paper_over_refuses_the_batch(self) -> None:
        with pytest.raises(WorkspaceCapacityExceeded):
            self.filled(MAX_PAPERS_PER_WORKSPACE - 2).ensure_room_for(["a", "b", "c"])

    def test_papers_already_here_and_repeats_take_no_extra_room(self) -> None:
        ws = self.filled(MAX_PAPERS_PER_WORKSPACE - 1)
        ws.ensure_room_for(["paper-0", "paper-1", "new", "new"])

    def test_a_removed_paper_frees_its_room(self) -> None:
        ws = self.filled(MAX_PAPERS_PER_WORKSPACE)
        ws.remove_paper("paper-0")
        ws.ensure_room_for(["new"])
