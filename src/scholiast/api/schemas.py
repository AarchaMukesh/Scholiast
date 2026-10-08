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

from datetime import date, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from scholiast.application.use_cases.add_discovered import MAX_ADD_PAPERS


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


class RegisterRequest(BaseModel):
    """Credentials for a new account (FR-A1).

    Length bounds are stated here as well as in the domain so a 10,000-character
    password is refused at the boundary, before Argon2id is asked to hash it —
    reject-by-default applied to a denial-of-service vector rather than to a typo.
    The exact policy still lives in `domain.identity`; this is the cheap outer
    check, not the authority.
    """

    model_config = ConfigDict(extra="forbid")

    email: Annotated[str, StringConstraints(min_length=3, max_length=320)]
    password: Annotated[str, StringConstraints(min_length=12, max_length=1024)]


class LoginRequest(BaseModel):
    """Credentials presented for a session.

    Deliberately NOT constrained the way registration is: a login attempt with a
    short password must fail as a failed login (401), not as a validation error
    (422), or the difference tells an attacker which passwords are worth trying.
    """

    model_config = ConfigDict(extra="forbid")

    email: str
    password: str


class RefreshRequest(BaseModel):
    """A refresh secret, presented for rotation or revocation."""

    model_config = ConfigDict(extra="forbid")

    refresh_token: str


class TokenPair(BaseModel):
    """What every successful authentication returns.

    The refresh token appears here once and is never retrievable again: the
    server keeps only its fingerprint (D84).
    """

    access_token: str
    refresh_token: str
    # S105 reads "token_type" as a credential name. It is the OAuth scheme
    # label — the word "bearer" — and the actual credential is the field above.
    token_type: str = "bearer"  # noqa: S105
    #: Seconds until the ACCESS token expires — the conventional OAuth field, and
    #: the one a client actually schedules against.
    expires_in: int
    user_id: str


# -- discovery (FR-D1, FR-D6, D122) ---------------------------------------------


class DiscoverySearch(_Input):
    #: A topic or a question. Searchability (a letter or digit) is the domain's
    #: rule, so a query of punctuation is a 400 from the use case, not a 422 here.
    query: NonBlank = Field(max_length=500)


class OpenAccessLinkOut(BaseModel):
    url: str
    #: What the SOURCE claims the link is ("pdf" or "landing"). The claim was wrong
    #: in both directions in the spike, so a client must not treat it as a fact.
    kind: str
    licence: str | None = None


class WorkMemberOut(BaseModel):
    """One source's record of a work: where that source put it."""

    source: str
    source_id: str
    #: The position in that source's own results, 1-based.
    source_rank: int


class PaperRefOut(BaseModel):
    """A record to send to the add endpoint as it is: `{"papers": [<this>]}`."""

    source: str
    source_id: str


class WorkOut(BaseModel):
    """One paper, however many sources returned it (FR-D1, FR-D3, D147).

    Title, authors, venue and published date are the `add` record's, so what is shown
    is what will be added. The rest is merged across `members`: a publisher DOI, the
    arXiv id, the earliest year, the longest abstract, the largest citation count and
    every open-access link once.
    """

    #: Every source record of this paper, in the search's interleaved order.
    members: list[WorkMemberOut]
    #: The record to add this paper by: one the fetcher can follow, if any is.
    add: PaperRefOut
    title: str
    authors: list[str]
    year: int | None = None
    published: date | None = None
    venue: str | None = None
    abstract: str | None = None
    #: None when the source reports none, which arXiv never does. Never 0 for unknown.
    citation_count: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    open_access: list[OpenAccessLinkOut] = Field(default_factory=list)


class SourceStatusOut(BaseModel):
    """What happened to one source in this search (FR-D6)."""

    source: str
    status: Literal["ok", "timeout", "rate_limited", "error", "circuit_open"]
    results: int
    total_hits: int | None = None
    #: When the source said, or our pacing knows, when it will accept work again.
    retry_after_seconds: float | None = None
    #: Whether this source's page came from the cache rather than the source (D127).
    #: Per user, so it reveals nothing about anyone else's searches.
    cached: bool = False
    #: How old a cached page is, in seconds; at most a day (D127).
    age_seconds: float | None = None


class DiscoveryResults(BaseModel):
    #: One entry per paper, not per source record (FR-D3).
    results: list[WorkOut]
    source_status: list[SourceStatusOut]
    #: True whenever any source did not answer (FR-D6).
    partial: bool
    #: How `results` is ordered. Until phase 8 ranks, it is each source's own order,
    #: interleaved, with each paper where its earliest record was (D122, D147).
    ordering: Literal["source_rank"] = "source_rank"


# -- adding discovered papers (D133, D137) ---------------------------------------


class DiscoveredPaperRef(_Input):
    """One paper to add, exactly as search returned it: never a URL, never metadata."""

    source: NonBlank = Field(max_length=32)
    source_id: NonBlank = Field(max_length=64)


class AddDiscoveredPapers(_Input):
    #: All added, or none (D133).
    papers: list[DiscoveredPaperRef] = Field(min_length=1, max_length=MAX_ADD_PAPERS)


class AddedPaperOut(BaseModel):
    source: str
    source_id: str
    paper_id: str
    #: None when the paper was already here, or already indexed, and nothing was queued.
    job_id: str | None
    state: str
    #: True when an existing paper of yours was reused rather than created (FR-I7).
    deduplicated: bool


class AddDiscoveredAccepted(BaseModel):
    """`202` from the add endpoint: one entry per paper sent, in the order sent."""

    papers: list[AddedPaperOut]
