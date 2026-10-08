"""Domain error taxonomy.

Errors are *classified*, not stringly-typed, because two separate decisions hang
off the class: whether the operation is retried, and what the user is told
(FR-I8, FR-I9, NFR-U2). A bare `except Exception: mark_failed()` collapses both.

Doc 4 §4.4.1 defines this taxonomy normatively; this module is its executable form.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique


@unique
class FailureClass(StrEnum):
    """Why an ingestion attempt failed.

    Retryability is a property of the *class*, never a per-call judgement — that
    is what stops "retry everything" and "retry nothing" from both being wrong.
    """

    #: Upstream has no open-access full text (404, paywall, host down).
    SOURCE_UNAVAILABLE = "source_unavailable"
    #: Timeout, 5xx, rate limit, worker died. Expected to succeed later.
    TRANSIENT = "transient"
    #: Encrypted, corrupt, or structurally invalid PDF.
    UNPARSEABLE = "unparseable"
    #: Valid PDF with no text layer — a scan. OCR is out of scope (FR-I12).
    SCANNED_IMAGE = "scanned_image"
    #: Exceeds the upload size limit (NFR-P8).
    TOO_LARGE = "too_large"
    #: GPU OOM. Retryable, but only with a smaller batch.
    RESOURCE_EXHAUSTED = "resource_exhausted"

    @property
    def retryable(self) -> bool:
        return self in _RETRYABLE

    @property
    def user_message(self) -> str:
        """Actionable text for the user (NFR-U2).

        Deliberately says what the user can *do*, not what the system failed at.
        "Parse error in stage 3" is a log line, not a user message.
        """
        return _USER_MESSAGES[self]


_RETRYABLE: frozenset[FailureClass] = frozenset(
    {FailureClass.TRANSIENT, FailureClass.RESOURCE_EXHAUSTED}
)

_USER_MESSAGES: dict[FailureClass, str] = {
    FailureClass.SOURCE_UNAVAILABLE: (
        "This paper's full text isn't openly available. Upload the PDF to continue."
    ),
    # D143: a membership carries a class only once it is terminal (a retry clears
    # it), so the retryable classes speak after retrying has stopped.
    FailureClass.TRANSIENT: (
        "Something went wrong on our side, and retrying didn't fix it. "
        "Remove the paper and add it again later."
    ),
    FailureClass.UNPARSEABLE: (
        "This PDF couldn't be read — it may be password-protected or corrupted."
    ),
    FailureClass.SCANNED_IMAGE: (
        "This looks like a scanned document with no text layer. "
        "Text recognition isn't supported yet, so it can't be indexed."
    ),
    FailureClass.TOO_LARGE: "This file is too large. The limit is 50 MB.",
    FailureClass.RESOURCE_EXHAUSTED: (
        "The system was too busy to process this paper, and retrying didn't help. "
        "Remove it and add it again later."
    ),
}


class DomainError(Exception):
    """Base class for rule violations originating in the domain.

    Distinct from infrastructure failures: a DomainError means the *request* was
    invalid, so retrying it unchanged will fail identically.
    """


class InvariantViolation(DomainError):
    """An entity was asked to enter a state its rules forbid."""


class IllegalStateTransition(InvariantViolation):
    """A lifecycle transition that the state machine does not permit."""

    def __init__(self, entity: str, current: str, requested: str) -> None:
        super().__init__(f"{entity} cannot move from {current!r} to {requested!r}")
        self.entity = entity
        self.current = current
        self.requested = requested


class WorkspaceNotAvailable(DomainError):
    """The target workspace is not one this scope can act on.

    Deliberately does not distinguish "does not exist" from "belongs to someone
    else": telling them apart is how an API leaks which ids are real, which is
    why FR-A2 asks for 404 rather than 403. Its own class rather than a generic
    invariant violation, because the two map to different HTTP statuses and
    conflating them would answer a missing workspace with a 500.
    """

    def __init__(self, workspace_id: str) -> None:
        super().__init__(f"Workspace {workspace_id!r} was not found.")
        self.workspace_id = workspace_id


class WorkspaceNotQueryable(DomainError):
    """A query arrived before any paper finished indexing (FR-W5).

    Answering from parametric knowledge here would be a correctness failure, not
    a degraded experience — so this is an error, not a warning.
    """

    def __init__(self, pending: int) -> None:
        super().__init__(f"No papers are ready to query yet ({pending} still processing).")
        self.pending = pending


class InsufficientPapersForComparison(DomainError):
    """Comparison needs at least two indexed papers (§4.8)."""

    def __init__(self, available: int) -> None:
        super().__init__(
            f"Comparison needs at least 2 indexed papers; this workspace has {available}."
        )
        self.available = available


class WorkspaceCapacityExceeded(DomainError):
    """The per-workspace paper cap (NFR-S1) would be exceeded."""

    def __init__(self, cap: int) -> None:
        super().__init__(
            f"This workspace is at its limit of {cap} papers. "
            "Remove some, or create another workspace."
        )
        self.cap = cap


@dataclass(frozen=True, slots=True)
class IngestionFailure:
    """A classified ingestion failure, ready to persist and to show.

    `detail` is for logs and debugging; `failure_class.user_message` is what the
    user sees. Keeping them separate is what stops stack traces reaching the UI.
    """

    failure_class: FailureClass
    detail: str
    #: Seconds the failing service said to wait before trying again, when it said
    #: (D142). None means it said nothing, not "retry immediately".
    retry_after: float | None = None

    @property
    def retryable(self) -> bool:
        return self.failure_class.retryable

    @property
    def user_message(self) -> str:
        return self.failure_class.user_message
