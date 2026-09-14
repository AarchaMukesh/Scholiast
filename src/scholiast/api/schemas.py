"""Request and response DTOs (NFR-SEC2, NFR-M6).

Typed at the boundary and reject-by-default, so a malformed request is a 422
problem document rather than a `KeyError` three layers in. These are *not* the
domain entities: exposing those would make every internal rename a breaking API
change, and would leak fields (`content_hash`, `owner_id`, `tei_key`) that no
client has any business seeing.

`model_config = ConfigDict(extra="forbid")` on inputs is the reject-by-default
half. A client sending `{"nmae": "typo"}` gets told, instead of silently creating
a workspace called nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


class _Input(BaseModel):
    """Base for anything a client sends."""

    model_config = ConfigDict(extra="forbid")


#: Stripped, then required to be non-empty. Both halves matter: the domain
#: entities reject blank names and blank queries with an `InvariantViolation`,
#: which the error handlers correctly map to a 500 because a violated invariant
#: means a bug. A question of three spaces is not a bug — it is a bad request,
#: and it has to be refused here or it arrives as a server error.
NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CreateWorkspace(_Input):
    name: NonBlank = Field(max_length=200)


class RenameWorkspace(_Input):
    name: NonBlank = Field(max_length=200)


class AskQuestion(_Input):
    question: NonBlank = Field(max_length=4000)


class CreateConversation(_Input):
    title: str | None = Field(default=None, max_length=200)


class WorkspaceSummary(BaseModel):
    id: str
    name: str
    state: str
    paper_count: int
    #: Exactly the corpus a query may retrieve from. Surfaced because FR-W5's
    #: disclosure only means something if the client can tell "nothing indexed
    #: yet" from "nothing matched".
    queryable_count: int
    created_at: datetime | None = None


class PaperSummary(BaseModel):
    """One membership row, as the workspace view needs it (FR-W4's M1 subset)."""

    paper_id: str
    title: str
    state: str
    #: Present exactly when the state is a failure. §4.4.1's taxonomy reaches the
    #: client so the UI can say what went wrong and whether retrying would help.
    failure_class: str | None = None
    failure_detail: str | None = None
    #: The actionable half (NFR-U2) — what the user can *do*, not what broke.
    user_message: str | None = None


class JobStatus(BaseModel):
    """Ingestion progress, for the UI's per-paper state (NFR-U1)."""

    job_id: str
    paper_id: str
    state: str
    attempts: int
    last_error: str | None = None


class UploadAccepted(BaseModel):
    """`202` from the upload endpoint (§5.4.2)."""

    paper_id: str
    job_id: str | None
    state: str
    #: True when an existing paper was reused rather than re-ingested (FR-I7).
    deduplicated: bool


class RemovalResult(BaseModel):
    paper_id: str
    removed: bool
    #: True when no workspace held the paper any more and its derived data was
    #: deleted (FR-W2). False means another workspace still has it (AD-8).
    purged: bool
    vectors_deleted: int


class ConversationCreated(BaseModel):
    conversation_id: str


class CitationOut(BaseModel):
    label: str
    chunk_id: str | None
    status: str
    claim_text: str


class EvidenceOut(BaseModel):
    """One passage, carrying everything a click needs (NFR-U3).

    The span is the one FR-I5's round trip proved indexes `sections.text`, so
    the reader view can highlight it without a second lookup deciding where it is.
    """

    label: str
    chunk_id: str
    paper_id: str
    paper_title: str
    section_path: str
    page: int
    char_start: int
    char_end: int
    text: str
    score: float
    describes_other_work: bool


class TurnOut(BaseModel):
    """One stored turn, for conversation history (FR-Q6, FR-Q9)."""

    role: str
    content: str
    citations: list[CitationOut] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    latency_ms: dict[str, int] | None = None


class SectionOut(BaseModel):
    """A section with its canonical text — the citation's landing target (D42)."""

    id: str
    path: str
    raw_title: str
    kind: str
    page_start: int
    page_end: int
    text: str


class PaperContent(BaseModel):
    paper_id: str
    title: str
    sections: list[SectionOut]
