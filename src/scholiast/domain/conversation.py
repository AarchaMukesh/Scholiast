"""Conversations and the turns that make them up (§5.4.3, FR-Q6, FR-Q9).

A turn is not just text. Doc 5 §5.4.3 ends the QA flow with "PERSIST (message +
evidence set + timings + token cost)", and each of those exists for a reason that
outlives the request:

  evidence     FR-Q6 makes the retrieved set inspectable. Without it stored, the
               only way to show a user what an answer was based on is to re-run
               retrieval and hope it was deterministic.

  citations    the audit result, not the raw labels — so a fabricated label that
               was stripped from the prose is still visible as *having been
               stripped*. A disclosure that vanishes on persistence is not one.

  timings      per stage (AP-8). This is what makes "it's slow" answerable as
               "retrieval is 60% of p95", and it has to be recorded from the
               first turn or M2's latency profiling has nothing to profile.

  degradation  every rung of §5.11's ladder that was used. An answer that was
               served without synthesis, or by the secondary provider, says so
               permanently — not only in the response that scrolled past.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.retrieval import Citation, ScoredChunk


@unique
class TurnRole(StrEnum):
    """Who produced a turn. Mirrors the `ck_messages_role` CHECK."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class StageTimings:
    """Where the wall clock went (AP-8, FR-Q9).

    Milliseconds per stage rather than one total, because a total cannot be
    acted on. `retrieval` covers embedding the query and the ANN search;
    `generation` is time to the last token, and `first_token` is the number
    NFR-P3 actually budgets — a fast total with a slow first token is a bad
    experience that a single figure would hide.
    """

    retrieval_ms: int = 0
    assembly_ms: int = 0
    first_token_ms: int = 0
    generation_ms: int = 0

    @property
    def total_ms(self) -> int:
        return self.retrieval_ms + self.assembly_ms + self.generation_ms

    def as_dict(self) -> dict[str, int]:
        return {
            "retrieval_ms": self.retrieval_ms,
            "assembly_ms": self.assembly_ms,
            "first_token_ms": self.first_token_ms,
            "generation_ms": self.generation_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True, slots=True)
class Turn:
    """One message in a conversation, with everything that explains it."""

    role: TurnRole
    content: str
    citations: tuple[Citation, ...] = ()
    evidence: tuple[ScoredChunk, ...] = ()
    degraded: tuple[str, ...] = ()
    prompt_version: str | None = None
    provider: str | None = None
    model: str | None = None
    timings: StageTimings | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.role is TurnRole.USER and (self.citations or self.evidence):
            # A question cites nothing. Allowing it would make the evidence
            # panel ambiguous about which turn the evidence belongs to.
            raise InvariantViolation("a user turn cannot carry citations or evidence")
        if not self.content.strip() and self.role is TurnRole.USER:
            raise InvariantViolation("a user turn cannot be blank")

    @property
    def was_degraded(self) -> bool:
        return bool(self.degraded)


@dataclass(slots=True)
class Conversation:
    """A workspace-scoped thread of turns."""

    id: str
    workspace_id: str
    title: str | None = None
    turns: list[Turn] = field(default_factory=list)
    created_at: datetime | None = None

    @property
    def is_empty(self) -> bool:
        return not self.turns
