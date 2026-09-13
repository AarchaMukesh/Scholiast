"""Workspace and paper lifecycles.

Doc 4 §4.3 and §4.4.1 define these state machines *normatively* — meaning the
implementation matches them or the document changes first. Encoding the
transition tables as data rather than as scattered `if` statements is what makes
that claim checkable: the tables below are exhaustively tested, so an illegal
transition is impossible rather than merely unlikely.

The subtle rule worth reading twice is `Workspace.is_queryable`: a workspace is
queryable as soon as *one* paper is indexed, and queries must then disclose which
papers were excluded (FR-W5). Silently answering over a subset is a correctness
bug, not a UX detail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique

from scholiast.domain.errors import (
    FailureClass,
    IllegalStateTransition,
    InsufficientPapersForComparison,
    InvariantViolation,
    WorkspaceCapacityExceeded,
)

#: NFR-S1. A deliberate simplification: capping per-workspace size keeps
#: retrieval scoped small enough that per-workspace vector filtering stays
#: viable without index partitioning.
MAX_PAPERS_PER_WORKSPACE = 100

#: §4.8 — comparison is structurally meaningless below two papers.
MIN_PAPERS_FOR_COMPARISON = 2


@unique
class PaperState(StrEnum):
    """Per-membership ingestion lifecycle (doc 4 §4.4.1).

    State lives on the *membership*, not the paper: the same open-access paper may
    be INDEXED globally while a given workspace has only just queued it.
    """

    CANDIDATE = "candidate"  # discovery result; not yet a member
    QUEUED = "queued"
    FETCHING = "fetching"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXED = "indexed"  # queryable
    FAILED = "failed"  # retryable or exhausted; see IngestionFailure
    UNSUPPORTED = "unsupported"  # terminal; retrying cannot help
    REMOVED = "removed"

    @property
    def is_terminal(self) -> bool:
        return self in (PaperState.REMOVED, PaperState.UNSUPPORTED)

    @property
    def is_in_flight(self) -> bool:
        return self in _IN_FLIGHT

    @property
    def is_queryable(self) -> bool:
        return self is PaperState.INDEXED


_IN_FLIGHT: frozenset[PaperState] = frozenset(
    {
        PaperState.QUEUED,
        PaperState.FETCHING,
        PaperState.PARSING,
        PaperState.CHUNKING,
        PaperState.EMBEDDING,
    }
)

#: Legal forward transitions. Every pipeline stage may also fail, and removal is
#: permitted from anywhere non-terminal (a user may delete mid-ingestion).
_PAPER_TRANSITIONS: dict[PaperState, frozenset[PaperState]] = {
    PaperState.CANDIDATE: frozenset({PaperState.QUEUED}),
    PaperState.QUEUED: frozenset({PaperState.FETCHING, PaperState.PARSING}),
    PaperState.FETCHING: frozenset({PaperState.PARSING}),
    PaperState.PARSING: frozenset({PaperState.CHUNKING}),
    PaperState.CHUNKING: frozenset({PaperState.EMBEDDING}),
    PaperState.EMBEDDING: frozenset({PaperState.INDEXED}),
    PaperState.INDEXED: frozenset(),
    PaperState.FAILED: frozenset({PaperState.QUEUED}),  # manual or automatic retry
    PaperState.UNSUPPORTED: frozenset(),
    PaperState.REMOVED: frozenset(),
}

_FAILABLE: frozenset[PaperState] = _IN_FLIGHT | {PaperState.CANDIDATE}

#: The only two states a membership may be *created* in. See `add_paper`.
_ENTRY_STATES: frozenset[PaperState] = frozenset({PaperState.QUEUED, PaperState.INDEXED})


#: Classes for which retrying cannot help *and* the user has no remedy either —
#: the file is what it is. §4.4.1 calls this UNSUPPORTED: "no false hope of
#: retry". SOURCE_UNAVAILABLE is deliberately absent: it is equally
#: non-retryable, but the user *can* fix it by uploading the PDF, and §4.4.1
#: offers a manual retry from FAILED and none from UNSUPPORTED.
_UNSUPPORTED_CLASSES: frozenset[FailureClass] = frozenset(
    {FailureClass.UNPARSEABLE, FailureClass.SCANNED_IMAGE, FailureClass.TOO_LARGE}
)


def terminal_state_for(failure_class: FailureClass) -> PaperState:
    """Where a membership lands when this class is final (§4.4.1).

    §4.4.1 tabulates retryability per class and tabulates the two terminal states
    separately, but never joins them — so the mapping had been left to whichever
    call site was writing the transition. Stating it once means the difference
    that matters to a user, "try again yourself" versus "this will never work",
    is decided in one place instead of being re-derived per caller.
    """
    return PaperState.UNSUPPORTED if failure_class in _UNSUPPORTED_CLASSES else PaperState.FAILED


def paper_transition_allowed(current: PaperState, requested: PaperState) -> bool:
    """Single source of truth for legal paper-state moves."""
    if requested is current:
        return True  # idempotent re-application, e.g. a redelivered job
    if requested is PaperState.REMOVED:
        return not current.is_terminal or current is PaperState.UNSUPPORTED
    if requested in (PaperState.FAILED, PaperState.UNSUPPORTED):
        return current in _FAILABLE
    if requested is PaperState.QUEUED:
        # Back to the start of the pipeline. Two callers need this and the
        # §4.4.1 diagram only draws one of them: a manual or automatic retry
        # from FAILED, and — the undrawn case — a job redelivered after its
        # worker died mid-stage. An at-least-once queue guarantees that second
        # case will happen (§5.7.4), and the attempt genuinely restarts from
        # the beginning, so a membership stuck at CHUNKING must be able to say
        # so rather than forcing the pipeline to fake forward progress.
        return current in _FAILABLE or current is PaperState.FAILED
    return requested in _PAPER_TRANSITIONS[current]


@unique
class WorkspaceState(StrEnum):
    """Workspace lifecycle (doc 4 §4.3)."""

    EMPTY = "empty"
    POPULATING = "populating"  # members exist, none indexed yet
    ACTIVE = "active"  # >=1 indexed => queryable
    DELETING = "deleting"  # persisted, resumable cascade (§5.6.4)

    @property
    def is_terminal(self) -> bool:
        return self is WorkspaceState.DELETING


_WORKSPACE_TRANSITIONS: dict[WorkspaceState, frozenset[WorkspaceState]] = {
    WorkspaceState.EMPTY: frozenset({WorkspaceState.POPULATING, WorkspaceState.DELETING}),
    WorkspaceState.POPULATING: frozenset(
        {
            WorkspaceState.ACTIVE,
            WorkspaceState.EMPTY,
            WorkspaceState.DELETING,
        }
    ),
    WorkspaceState.ACTIVE: frozenset(
        {
            WorkspaceState.POPULATING,
            WorkspaceState.EMPTY,
            WorkspaceState.DELETING,
        }
    ),
    WorkspaceState.DELETING: frozenset(),
}


def workspace_transition_allowed(current: WorkspaceState, requested: WorkspaceState) -> bool:
    if requested is current:
        return True
    return requested in _WORKSPACE_TRANSITIONS[current]


@dataclass(slots=True)
class PaperMembership:
    """A paper's presence in one workspace, with its ingestion state."""

    workspace_id: str
    paper_id: str
    state: PaperState
    added_at: datetime
    failure_class: str | None = None
    failure_detail: str | None = None
    attempts: int = 0

    def transition_to(
        self,
        requested: PaperState,
        *,
        failure_class: str | None = None,
        failure_detail: str | None = None,
    ) -> None:
        if not paper_transition_allowed(self.state, requested):
            raise IllegalStateTransition(f"PaperMembership({self.paper_id})", self.state, requested)
        if requested in (PaperState.FAILED, PaperState.UNSUPPORTED) and not failure_class:
            # A failure without a class cannot produce a retry decision or a user
            # message, which defeats the entire taxonomy.
            raise InvariantViolation(f"transition to {requested} requires a failure_class")
        if requested is PaperState.QUEUED and self.state is not PaperState.CANDIDATE:
            # Any return to QUEUED is a fresh attempt — a retry after failure, or
            # a restart after a redelivery. Counting only the first would make
            # the user-visible attempt count disagree with reality precisely in
            # the case where a paper keeps dying mid-pipeline.
            self.attempts += 1
            self.failure_class = None
            self.failure_detail = None
        elif requested in (PaperState.FAILED, PaperState.UNSUPPORTED):
            self.failure_class = failure_class
            self.failure_detail = failure_detail
        self.state = requested

    @property
    def is_queryable(self) -> bool:
        return self.state.is_queryable


@dataclass(slots=True)
class Workspace:
    """The unit of curation and the boundary of every retrieval scope.

    There is no global search over content: retrieval is always workspace-scoped,
    which is what makes citations meaningful and grounding tractable.
    """

    id: str
    owner_id: str
    name: str
    state: WorkspaceState = WorkspaceState.EMPTY
    memberships: list[PaperMembership] = field(default_factory=list)
    created_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise InvariantViolation("workspace name cannot be blank")

    # --- membership -------------------------------------------------------

    @property
    def active_memberships(self) -> list[PaperMembership]:
        return [m for m in self.memberships if m.state is not PaperState.REMOVED]

    @property
    def queryable_paper_ids(self) -> tuple[str, ...]:
        """Exactly the corpus a query may retrieve from."""
        return tuple(m.paper_id for m in self.memberships if m.is_queryable)

    @property
    def in_flight_paper_ids(self) -> tuple[str, ...]:
        """Papers still processing — disclosed as excluded on every answer."""
        return tuple(m.paper_id for m in self.memberships if m.state.is_in_flight)

    @property
    def is_queryable(self) -> bool:
        """One indexed paper is enough (FR-W5).

        Queries never block on in-flight ingestion; they answer over what is
        ready and say what was left out.
        """
        return bool(self.queryable_paper_ids)

    @property
    def can_compare(self) -> bool:
        return len(self.queryable_paper_ids) >= MIN_PAPERS_FOR_COMPARISON

    def assert_can_compare(self) -> None:
        if not self.can_compare:
            raise InsufficientPapersForComparison(len(self.queryable_paper_ids))

    def add_paper(
        self, paper_id: str, *, now: datetime, state: PaperState = PaperState.QUEUED
    ) -> PaperMembership:
        """Add a paper, or return the existing membership unchanged.

        Idempotent by design (FR-I7): re-adding must not duplicate work or reset
        a paper that is already indexed.

        `state` exists because §5.4.2 has two entry points, not one. A new upload
        enters QUEUED; a paper the registry has already parsed *and* embedded
        enters INDEXED with no work queued at all — "create membership only". The
        second case cannot be expressed by adding at QUEUED and transitioning,
        because QUEUED -> INDEXED is not a legal move and should not become one:
        skipping the pipeline is a property of the *paper* already being done,
        not of a membership fast-forwarding through stages it never ran.
        """
        if state not in _ENTRY_STATES:
            raise InvariantViolation(
                f"a membership may only be created QUEUED or INDEXED, not {state}"
            )
        for m in self.memberships:
            if m.paper_id == paper_id and m.state is not PaperState.REMOVED:
                return m

        if len(self.active_memberships) >= MAX_PAPERS_PER_WORKSPACE:
            raise WorkspaceCapacityExceeded(MAX_PAPERS_PER_WORKSPACE)

        membership = PaperMembership(
            workspace_id=self.id,
            paper_id=paper_id,
            state=state,
            added_at=now,
        )
        self.memberships.append(membership)
        self.refresh_state()
        return membership

    def remove_paper(self, paper_id: str) -> None:
        for m in self.memberships:
            if m.paper_id == paper_id and m.state is not PaperState.REMOVED:
                m.transition_to(PaperState.REMOVED)
                self.refresh_state()
                return
        raise InvariantViolation(f"paper {paper_id} is not a member of workspace {self.id}")

    # --- lifecycle --------------------------------------------------------

    def refresh_state(self) -> None:
        """Re-derive workspace state from its memberships.

        Public because the ingestion worker owns the other half of the flow: it
        transitions a membership (QUEUED -> ... -> INDEXED) and then asks the
        workspace to re-derive. Keeping this explicit beats an observer pattern
        for two reasons — the call site reads honestly, and state is *derived*
        rather than assigned, which removes the drift bug where the aggregate
        claims ACTIVE while nothing is actually indexed.
        """
        if self.state is WorkspaceState.DELETING:
            return
        if not self.active_memberships:
            target = WorkspaceState.EMPTY
        elif self.is_queryable:
            target = WorkspaceState.ACTIVE
        else:
            target = WorkspaceState.POPULATING
        if workspace_transition_allowed(self.state, target):
            self.state = target

    def mark_deleting(self, *, now: datetime) -> None:
        """Enter the persisted, resumable deletion state (§5.6.4).

        Deletion spans three stores with no distributed transaction, so it is a
        real state rather than an instantaneous operation.

        Idempotent: the deletion job arrives on an at-least-once queue, so a
        redelivery must not fail. The recorded timestamp is set once and never
        moved — retention and audit logic keys off *when deletion began*, and a
        retry three days later must not rewrite that.
        """
        if self.state is WorkspaceState.DELETING:
            return
        if not workspace_transition_allowed(self.state, WorkspaceState.DELETING):
            raise IllegalStateTransition(
                f"Workspace({self.id})", self.state, WorkspaceState.DELETING
            )
        self.state = WorkspaceState.DELETING
        self.deleted_at = now
