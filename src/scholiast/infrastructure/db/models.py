"""Relational schema (doc 5 §5.6.3). Postgres is the source of truth (AD-3).

Two things here are load-bearing beyond ordinary data modelling.

**The privacy carve-out is enforced by the database, not by the application.**
Two partial unique indexes plus a CHECK constraint encode §4.4.2: open-access
papers deduplicate globally so they are parsed and embedded once, while a user
upload is never shared even on a content-hash collision. An unpublished
manuscript must not become a registry entry reachable through another tenant's
membership. Application code can have a bug; the index cannot be argued with.

**Defaults are server-side, not ORM-side.** A bare `default=` is applied by
SQLAlchemy in Python, so any insert that does not go through the ORM — a
migration backfill, a repair script, psql — violates NOT NULL instead of
getting the default. Same reasoning as the CHECK constraints: the database
holds the invariant, because the database is what every path goes through.

**CHECK constraints are generated from the domain enums.** `_one_of()` builds
the SQL from `SectionKind`, `PaperState` and friends, so adding a member without
a migration fails loudly in the schema test rather than silently storing a value
nothing downstream can interpret.

Deliberately deferred, with reasons, so the absences are choices rather than
oversights:

  comparisons        M5. Dead schema until then.
  chunks FTS index   M2's lexical arm. Nothing in M1 reads it, and a GIN index
                     is cheapest to build before the eval corpus lands.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from scholiast.application.ports.queue import JobPriority
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import (
    ChunkRole,
    PaperOrigin,
    PaperVisibility,
    ParseStatus,
    SectionKind,
)
from scholiast.domain.workspace import PaperState, WorkspaceState


def _one_of(column: str, enum: type[StrEnum]) -> str:
    """A CHECK constraint derived from a domain enum.

    Keeps the database's notion of a taxonomy identical to the domain's. If a
    member is added without a migration, inserting it raises — which is the
    failure we want, rather than a row nothing downstream can classify.
    """
    values = ", ".join(f"'{member.value}'" for member in enum)
    return f"{column} IN ({values})"


class Base(DeclarativeBase):
    pass


def _pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class User(Base):
    """An account. M1 seeded one; M3 phase 1 gave it a password (D85)."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _pk()
    email: Mapped[str] = mapped_column(String(320), unique=True, nullable=False)
    # Argon2id, NOT NULL since migration 0005: an account that cannot
    # authenticate is not an account, and a nullable hash invites a login path
    # that treats "no password set" as "no password needed" (D85).
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class RefreshToken(Base):
    """One issued refresh token — the stateful half of D84's split.

    Stored as a fingerprint, never as the secret, so a database leak yields
    nothing usable. `family_id` groups the rotation chain: refreshing revokes the
    presented token and issues its successor in the same family, and presenting
    an already-used token revokes the family entire, because a second use means
    either a replay or a theft and there is no third reading.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = _pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: The rotation chain this token belongs to. Not a foreign key: the first
    #: token in a family has no predecessor, and pointing at oneself to satisfy a
    #: constraint is how a schema starts lying.
    family_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    #: SHA-256 hex of the secret. Unique, so the same secret cannot be recorded
    #: twice even if two requests race.
    fingerprint: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    issued_at: Mapped[datetime] = _created_at()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    #: Set on logout, on rotation, and on family revocation. A row is usable only
    #: while this is NULL and `expires_at` is in the future.
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        # Refresh presents a fingerprint; expiry sweeps scan by expiry. Both are
        # hot enough to index, and the fingerprint's unique constraint already
        # provides its own.
        Index("ix_refresh_tokens_user", "user_id"),
        Index("ix_refresh_tokens_family", "family_id"),
        Index("ix_refresh_tokens_expires_at", "expires_at"),
    )


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[uuid.UUID] = _pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()
    #: Soft delete supports the resumable DELETING state (§4.3, §5.6.4).
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        CheckConstraint(_one_of("state", WorkspaceState), name="ck_workspaces_state"),
        Index("ix_workspaces_owner_deleted", "owner_id", "deleted_at"),
    )


class Paper(Base):
    """A work and its metadata. Identity is content-addressed (FR-I7)."""

    __tablename__ = "papers"

    id: Mapped[uuid.UUID] = _pk()
    visibility: Mapped[str] = mapped_column(Text, nullable=False)
    #: NOT NULL iff visibility='private' — see the CHECK below.
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    #: NULL only while a discovered paper awaits its PDF (D132); see
    #: ck_papers_hash_or_awaiting below.
    content_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Identity by external id (FR-I7, D132). Columns, not `external_ids` paths,
    #: because `update()` rewrites `external_ids` on every parse and identity
    #: under a unique index must sit where an update cannot move it.
    doi: Mapped[str | None] = mapped_column(Text, nullable=True)
    arxiv_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Whose metadata wins on a parse (D135).
    origin: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=PaperOrigin.UPLOAD.value
    )
    external_ids: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=sa_text("'{}'::jsonb")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    authors: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list, server_default=sa_text("'[]'::jsonb")
    )
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parse_status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=ParseStatus.PENDING.value
    )
    #: Set exactly when parse_status is 'failed' (D44). Reuses the §4.4.1
    #: FailureClass taxonomy rather than duplicating three of its values, so the
    #: retry decision and the user message come from one place. Lives on the
    #: paper because parsing happens once per paper: a second workspace adding an
    #: already-failed paper must see why without re-parsing, and without reading
    #: another tenant's membership row.
    parse_failure_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    parser_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Object-store key for the retained GROBID TEI (D43). Kept so re-chunking
    #: ablations are instant and bit-for-bit reproducible across a GROBID
    #: version change, which is what FR-E4 needs and re-parsing cannot give.
    tei_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_one_of("visibility", PaperVisibility), name="ck_papers_visibility"),
        CheckConstraint(_one_of("parse_status", ParseStatus), name="ck_papers_parse_status"),
        CheckConstraint(
            f"parse_failure_class IS NULL OR {_one_of('parse_failure_class', FailureClass)}",
            name="ck_papers_parse_failure_class",
        ),
        # The equivalence, mirroring the domain invariant on Paper. A failure
        # with no class cannot produce a retry decision or a user message; a
        # class with no failure claims a problem that did not happen.
        CheckConstraint(
            "(parse_status = 'failed') = (parse_failure_class IS NOT NULL)",
            name="ck_papers_parse_failure_iff_failed",
        ),
        # The §4.4.2 invariant, in the database. Written as an equivalence so it
        # catches both directions: a private paper with no owner, and a public
        # paper that is owner-scoped.
        CheckConstraint(
            "(visibility = 'private') = (owner_id IS NOT NULL)",
            name="ck_papers_private_iff_owned",
        ),
        CheckConstraint(_one_of("origin", PaperOrigin), name="ck_papers_origin"),
        # D132. A row without its PDF has parsed nothing and can be found by an
        # identifier; the domain entity enforces the same rule.
        CheckConstraint(
            "content_hash IS NOT NULL OR "
            "(parse_status = 'pending' AND (doi IS NOT NULL OR arxiv_id IS NOT NULL))",
            name="ck_papers_hash_or_awaiting",
        ),
        # One spelling per identity, or the unique indexes below enforce nothing:
        # lowercase, and never arXiv's DataCite DOI, which is an arXiv id.
        CheckConstraint(
            "doi IS NULL OR (doi = lower(doi) AND doi LIKE '10.%/%' "
            "AND doi NOT LIKE '10.48550/arxiv.%')",
            name="ck_papers_doi_canonical",
        ),
        CheckConstraint(
            "arxiv_id IS NULL OR (arxiv_id = lower(arxiv_id) AND arxiv_id !~ 'v[0-9]+$')",
            name="ck_papers_arxiv_id_canonical",
        ),
        # Open-access papers deduplicate globally: parsed and embedded once.
        Index(
            "uq_papers_public_hash",
            "content_hash",
            unique=True,
            postgresql_where=("visibility = 'public'"),
        ),
        # Uploads deduplicate only within one owner. A colliding hash across two
        # tenants creates two rows — deliberately. This index is the reason an
        # unpublished manuscript cannot leak through a shared registry entry.
        Index(
            "uq_papers_private_owner_hash",
            "owner_id",
            "content_hash",
            unique=True,
            postgresql_where=("visibility = 'private'"),
        ),
        # A user holds one row per work (D132). Per owner, because discovered
        # papers are private (D131): two users adding one paper get two rows.
        Index(
            "uq_papers_private_owner_doi",
            "owner_id",
            "doi",
            unique=True,
            postgresql_where=("visibility = 'private' AND doi IS NOT NULL"),
        ),
        Index(
            "uq_papers_private_owner_arxiv",
            "owner_id",
            "arxiv_id",
            unique=True,
            postgresql_where=("visibility = 'private' AND arxiv_id IS NOT NULL"),
        ),
    )


class WorkspacePaper(Base):
    """Membership, and where user-visible lifecycle state lives.

    State is here rather than on `papers` because the same open-access paper may
    be INDEXED globally while a given workspace has only just queued it. Parse
    state stays on `papers`, because parsing happens once per paper.
    """

    __tablename__ = "workspace_papers"

    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), primary_key=True
    )
    state: Mapped[str] = mapped_column(Text, nullable=False)
    failure_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sa_text("0")
    )
    added_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_one_of("state", PaperState), name="ck_workspace_papers_state"),
        # Same gap as parse_status had: an unconstrained taxonomy column. Applied
        # here too so every taxonomy in the schema is generated from its domain
        # enum rather than trusted to the writer.
        CheckConstraint(
            f"failure_class IS NULL OR {_one_of('failure_class', FailureClass)}",
            name="ck_workspace_papers_failure_class",
        ),
        # A failure without a class cannot produce a retry decision or a user
        # message, which defeats the entire taxonomy (doc 4 §4.4.1).
        CheckConstraint(
            "(state NOT IN ('failed', 'unsupported')) OR (failure_class IS NOT NULL)",
            name="ck_workspace_papers_failure_classified",
        ),
    )


class Section(Base):
    """A node in the document's section hierarchy, with its text (D42).

    `text` is the canonical parsed section text, and every chunk's
    `char_start`/`char_end` indexes *this* string. Chunking rule 1 guarantees no
    chunk crosses a section boundary, so those offsets are always well-defined —
    resolving a citation is this row plus a slice.
    """

    __tablename__ = "sections"

    id: Mapped[uuid.UUID] = _pk()
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sections.id", ondelete="CASCADE"), nullable=True
    )
    #: Not in §5.6.3, added deliberately: the domain `Section` requires it and
    #: every attribution decision depends on the normalised value, never the raw
    #: heading. Its absence from the doc is a gap, not a choice.
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    raw_title: Mapped[str] = mapped_column(Text, nullable=False)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)

    __table_args__ = (
        CheckConstraint(_one_of("kind", SectionKind), name="ck_sections_kind"),
        Index("ix_sections_paper_order", "paper_id", "order_index"),
    )


class Chunk(Base):
    """A retrieval unit with full provenance and version stamps (FR-I5, NFR-M5).

    The id is a TEXT primary key, not a UUID, because the chunker derives it
    deterministically (`{paper_id}:p{n}:c{m}`). That makes re-chunking idempotent
    — the same input yields the same ids, so a re-index updates rather than
    duplicating.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sections.id", ondelete="CASCADE"), nullable=False
    )
    #: Parent windows are returned as context; children and tables are embedded.
    parent_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("chunks.id", ondelete="CASCADE"), nullable=True
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int] = mapped_column(Integer, nullable=False)
    #: Offsets into `sections.text` (D42), not the document and not the PDF.
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    #: NFR-M5. These are what make "was it the reranker or the re-chunk?"
    #: answerable, and what let a re-index target only what actually changed.
    chunker_version: Mapped[str] = mapped_column(Text, nullable=False)
    embedder_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint(_one_of("role", ChunkRole), name="ck_chunks_role"),
        CheckConstraint("char_end > char_start", name="ck_chunks_span_non_empty"),
        CheckConstraint("page >= 1", name="ck_chunks_page_one_indexed"),
        # A child chunk with no parent window cannot be parent-expanded, which
        # is the whole point of small-to-big retrieval.
        CheckConstraint(
            "(role <> 'child') OR (parent_id IS NOT NULL)",
            name="ck_chunks_child_has_parent",
        ),
        Index("ix_chunks_paper_ordinal", "paper_id", "ordinal"),
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = _pk()
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _created_at()


class Message(Base):
    """One turn, with everything needed to explain how it was produced.

    The JSONB columns are not incidental metadata: `evidence` and `citations`
    make an answer auditable after the fact (FR-Q6), `latency_ms` per stage is
    what turns "it's slow" into "reranking is 60% of p95" (AP-8), and
    `prompt_version` is the NFR-M5 stamp that lets a quality change be attributed
    to a prompt rather than to retrieval.
    """

    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = _pk()
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    #: Position within the conversation, 0-based. Not in §5.6.3, added
    #: deliberately: `created_at`'s server default is `now()`, which is
    #: *transaction* time in Postgres, so the two turns of one QA exchange share
    #: a timestamp exactly and ordering falls through to a random UUID. A
    #: conversation is a sequence and is stored as one, the same way
    #: `chunks.ordinal` is.
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    citations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Numeric(10, 6), nullable=True)
    #: Degradation is disclosed, never silent (AP-7, NFR-U4).
    degraded: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant')", name="ck_messages_role"),
        CheckConstraint("ordinal >= 0", name="ck_messages_ordinal_non_negative"),
        # A concurrent double-append must fail loudly rather than quietly
        # producing two messages that claim the same position.
        Index("uq_messages_conversation_ordinal", "conversation_id", "ordinal", unique=True),
    )


class IngestionJob(Base):
    """Durable job state, claimed with `FOR UPDATE SKIP LOCKED`.

    §5.6.3 specifies `visible_at` and an index on `(state, visible_at)` labelled
    "claim query", so a lease-based claim is the schema's own design rather than
    an embellishment. The M1 worker loop runs in-process as doc 8 requires; only
    the *storage* is durable, which is what makes §5.4.2's rule — commit the job
    row before enqueueing — mean anything, and what makes the M3 swap to Celery
    a transport change rather than a rewrite.
    """

    __tablename__ = "ingestion_jobs"

    id: Mapped[uuid.UUID] = _pk()
    workspace_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("workspaces.id", ondelete="CASCADE"), nullable=False
    )
    paper_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False
    )
    state: Mapped[str] = mapped_column(Text, nullable=False)
    attempts: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=sa_text("0")
    )
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: FR-O1's correlation id, following the WORK rather than the message (D107).
    #: Nullable because every row written before migration 0006 has no request
    #: behind it, and because a job the sweep created has no request either.
    correlation_id: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: The open-access links the FETCHING stage tries, best first (D133, D134).
    #: On the job, not the paper: input to this unit of work, irrelevant once the
    #: paper has its PDF. NULL for uploads, which have their bytes already.
    source_links: Mapped[list[str] | None] = mapped_column(JSONB, nullable=True)
    #: What the claim orders by first (D139): an upload before a discovered paper.
    #: Defaults to interactive so a row written without it is never deprioritised.
    priority: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, server_default=sa_text(str(int(JobPriority.INTERACTIVE)))
    )
    #: The lease. A crashed worker's job becomes visible again rather than being
    #: lost, which is the at-least-once half of AD-5.
    visible_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    __table_args__ = (
        CheckConstraint(_one_of("state", PaperState), name="ck_ingestion_jobs_state"),
        CheckConstraint(
            "priority IN (" + ", ".join(str(int(p)) for p in sorted(JobPriority)) + ")",
            name="ck_ingestion_jobs_priority",
        ),
        Index("ix_ingestion_jobs_claim", "state", "visible_at"),
    )
