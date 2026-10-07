"""Tenant-scoped repository adapters (doc 5 §5.8 layer 2).

The mechanism is deliberately awkward: `ScopedRepository.__init__` takes a
`Scope` positionally, so a repository that cannot filter by tenant is one the
type checker refuses to build. Forgetting the predicate stops being a
code-review question and becomes a compile-time one.

Every scoped read funnels through `_visible_workspaces()` or
`_reachable_papers()`. Neither is optional and neither is parameterised by the
caller — a caller cannot widen its own scope, because it never supplies the
predicate in the first place.

**Where reachability comes from.** §4.4.2 adopted a global registry plus
per-workspace membership, and named the cost explicitly: "isolation now depends
on filtering by the caller's permitted `paper_id` set." So a paper is reachable
when the scope owns a workspace that has it as a member — *not* when the paper
happens to be public. The one deliberate exception is the dedup lookup, and it is
safe for the reason given on `find_by_content_hash`.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import CursorResult, Executable, Select, delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.tenancy import Scope, TenantIsolationViolation
from scholiast.domain.conversation import Conversation, StageTimings, Turn, TurnRole
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperOrigin,
    PaperVisibility,
    ParseStatus,
    Provenance,
    Section,
    SectionKind,
)
from scholiast.domain.retrieval import Citation, CitationStatus, RetrievalArm, ScoredChunk
from scholiast.domain.workspace import (
    PaperMembership,
    PaperState,
    Workspace,
    WorkspaceState,
)
from scholiast.infrastructure.db import models, pending


def _as_uuid(value: str) -> uuid.UUID | None:
    """Domain ids are strings; the database uses UUIDs.

    A malformed id is a miss, not a crash: ids arrive from URLs, and letting a
    `ValueError` escape would turn a probe for `/workspaces/../etc/passwd` into a
    500 that says more than a 404 does.
    """
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        return None


class ScopedRepository:
    """Base for every repository that touches tenant data."""

    def __init__(self, session: AsyncSession, scope: Scope) -> None:
        self._session = session
        self._scope = scope

    @property
    def scope(self) -> Scope:
        return self._scope

    def _visible_workspaces(self) -> Select[tuple[uuid.UUID]]:
        """Workspace ids this scope may act on.

        Soft-deleted workspaces are excluded: §5.6.4 marks DELETING and sweeps
        asynchronously, so a deleted workspace's rows outlive the user's action
        and must not keep serving reads in the meantime.
        """
        stmt = select(models.Workspace.id).where(models.Workspace.deleted_at.is_(None))
        if self._scope.is_system:
            return stmt
        return stmt.where(models.Workspace.owner_id == _as_uuid(self._tenant_id()))

    def _owned_workspaces(self) -> Select[tuple[uuid.UUID]]:
        """Workspace ids this scope OWNS, **including soft-deleted ones** (D92).

        One predicate different from `_visible_workspaces`, and the cascade is
        what the difference is for: by the time a deletion sweep runs, the
        workspace is soft-deleted *by construction*, so a visibility-scoped read
        would find nothing to delete and the sweep would silently do nothing.

        The split is the same one D92 drew for the queue. Ownership is the right
        question for work that is ABOUT a deletion; visibility is the right
        question for everything a user can still see. Using visibility here
        would not leak anything — it would strand data forever, which FR-W3
        forbids more loudly than any read would.
        """
        stmt = select(models.Workspace.id)
        if self._scope.is_system:
            return stmt
        return stmt.where(models.Workspace.owner_id == _as_uuid(self._tenant_id()))

    def _require_system(self, table: str, operation: str) -> None:
        """Refuse a cross-tenant sweep read to anything but the sweeper.

        The sweep legitimately spans tenants — residue has no owner — so these
        reads cannot carry a tenant predicate. That makes them exactly the shape
        a caller would want if it were trying to enumerate the whole system, so
        the scope check is the thing standing in for the predicate. Same line
        `PostgresQueue.claim` draws, for the same reason (D50).
        """
        if not self._scope.is_system:
            raise TenantIsolationViolation(table, operation, repr(self._scope))

    def _reachable_papers(self) -> Select[tuple[uuid.UUID]]:
        """Paper ids reachable through a workspace this scope owns."""
        return select(models.WorkspacePaper.paper_id).where(
            models.WorkspacePaper.workspace_id.in_(self._visible_workspaces()),
            models.WorkspacePaper.state != PaperState.REMOVED.value,
        )

    async def _affected_rows(self, statement: Executable) -> int:
        """Run a DML statement and report how many rows it touched.

        `session.execute()` is typed as returning `Result`, which does not expose
        `rowcount`; DML actually returns a `CursorResult`. The cast narrows to
        what the statement genuinely produces rather than silencing the error —
        and the row count is what tells a scoped update "not found" from
        "not yours", both of which the caller reports identically.
        """
        result = await self._session.execute(statement)
        return cast("CursorResult[Any]", result).rowcount

    def _tenant_id(self) -> str:
        # Only reached on the non-system path; `Scope` is structural, so the
        # attribute is read defensively rather than assumed.
        user_id = getattr(self._scope, "user_id", None)
        if not isinstance(user_id, str) or not user_id:
            raise TenantIsolationViolation("Scope", repr(self._scope), "missing user_id")
        return user_id


class SqlWorkspaceRepository(ScopedRepository):
    async def create(self, workspace: Workspace) -> Workspace:
        row = models.Workspace(
            id=_as_uuid(workspace.id) or uuid.uuid4(),
            owner_id=_as_uuid(workspace.owner_id),
            name=workspace.name,
            state=workspace.state.value,
        )
        self._session.add(row)
        await self._session.flush()
        return _to_workspace(row, memberships=[])

    async def get(self, workspace_id: str) -> Workspace | None:
        target = _as_uuid(workspace_id)
        if target is None:
            return None
        row = (
            await self._session.execute(
                select(models.Workspace).where(
                    models.Workspace.id == target,
                    models.Workspace.id.in_(self._visible_workspaces()),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None
        return _to_workspace(row, await self._memberships(row.id))

    async def list_all(self) -> list[Workspace]:
        rows = (
            (
                await self._session.execute(
                    select(models.Workspace)
                    .where(models.Workspace.id.in_(self._visible_workspaces()))
                    .order_by(models.Workspace.created_at)
                )
            )
            .scalars()
            .all()
        )
        return [_to_workspace(r, await self._memberships(r.id)) for r in rows]

    async def rename(self, workspace_id: str, name: str) -> bool:
        target = _as_uuid(workspace_id)
        if target is None:
            return False
        affected = await self._affected_rows(
            update(models.Workspace)
            .where(
                models.Workspace.id == target,
                models.Workspace.id.in_(self._visible_workspaces()),
            )
            .values(name=name)
        )
        return affected > 0

    async def soft_delete(self, workspace_id: str) -> bool:
        target = _as_uuid(workspace_id)
        if target is None:
            return False
        affected = await self._affected_rows(
            update(models.Workspace)
            .where(
                models.Workspace.id == target,
                models.Workspace.id.in_(self._visible_workspaces()),
            )
            .values(state=WorkspaceState.DELETING.value, deleted_at=datetime.now(UTC))
        )
        if affected:
            # The mark is now part of the caller's transaction, so the wake-up
            # has to wait for it to commit — publishing here would announce a
            # deletion a rollback then un-marked (D99, and D95's reasoning for
            # ingestion verbatim). The composition root drains this after the
            # commit; if it never does, `list_deleting` finds the row anyway.
            pending.record(self._session, pending.PENDING_DELETION_DISPATCH, workspace_id)
        return affected > 0

    async def list_deleting(self) -> tuple[str, ...]:
        """Workspaces whose cascade has not finished (§5.6.4 step 3, D99).

        Keyed on `deleted_at` rather than `state == DELETING` because the two
        are written together and only one of them is a timestamp: ordering by it
        sweeps the oldest unfinished cascade first, which bounds how long any
        residue can survive instead of leaving it to whichever row a scan
        happens to reach. A row whose state drifted but whose timestamp stands
        is still found, which is the safer direction to be wrong in.
        """
        self._require_system("workspaces", "list_deleting")
        rows = (
            (
                await self._session.execute(
                    select(models.Workspace.id)
                    .where(models.Workspace.deleted_at.is_not(None))
                    .order_by(models.Workspace.deleted_at)
                )
            )
            .scalars()
            .all()
        )
        return tuple(str(r) for r in rows)

    async def is_deleting(self, workspace_id: str) -> bool:
        """Whether the workspace carries a deletion mark (D99's safety check)."""
        target = _as_uuid(workspace_id)
        if target is None:
            return False
        found = (
            await self._session.execute(
                select(models.Workspace.id).where(
                    models.Workspace.id == target,
                    models.Workspace.id.in_(self._owned_workspaces()),
                    models.Workspace.deleted_at.is_not(None),
                )
            )
        ).scalar_one_or_none()
        return found is not None

    async def purge(self, workspace_id: str) -> bool:
        """Hard-delete the workspace row — the cascade's last step.

        Scoped by ownership, not visibility: this row is soft-deleted by the
        time anything calls it, so `_visible_workspaces` would match nothing and
        the cascade would never complete. The database cascades from here to
        conversations, messages, memberships and ingestion jobs.
        """
        target = _as_uuid(workspace_id)
        if target is None:
            return False
        affected = await self._affected_rows(
            delete(models.Workspace).where(
                models.Workspace.id == target,
                models.Workspace.id.in_(self._owned_workspaces()),
            )
        )
        return affected > 0

    async def _memberships(self, workspace_id: uuid.UUID) -> list[PaperMembership]:
        rows = (
            (
                await self._session.execute(
                    select(models.WorkspacePaper).where(
                        models.WorkspacePaper.workspace_id == workspace_id
                    )
                )
            )
            .scalars()
            .all()
        )
        return [
            PaperMembership(
                workspace_id=str(r.workspace_id),
                paper_id=str(r.paper_id),
                state=PaperState(r.state),
                added_at=r.added_at,
                failure_class=r.failure_class,
                failure_detail=r.failure_detail,
                attempts=r.attempts,
            )
            for r in rows
        ]


class SqlPaperRepository(ScopedRepository):
    async def get(self, paper_id: str) -> Paper | None:
        target = _as_uuid(paper_id)
        if target is None:
            return None
        row = (
            await self._session.execute(
                select(models.Paper).where(
                    models.Paper.id == target,
                    models.Paper.id.in_(self._reachable_papers()),
                )
            )
        ).scalar_one_or_none()
        return _to_paper(row) if row is not None else None

    async def get_for_update(self, paper_id: str) -> Paper | None:
        target = _as_uuid(paper_id)
        if target is None:
            return None
        row = (
            await self._session.execute(
                select(models.Paper)
                .where(
                    models.Paper.id == target,
                    models.Paper.id.in_(self._reachable_papers()),
                )
                # The paper row only: the reachability subquery's tables are read,
                # never locked, so a membership change elsewhere does not wait on it.
                .with_for_update(of=models.Paper)
            )
        ).scalar_one_or_none()
        return _to_paper(row) if row is not None else None

    async def find_by_content_hash(self, content_hash: str) -> Paper | None:
        # Mirrors the two partial unique indexes. Public rows are registry-wide
        # by design (§4.4.2 / AD-8 — parsed and embedded once); private rows stay
        # owner-scoped, which is the half where a global lookup would leak.
        owner = _as_uuid(self._tenant_id()) if not self._scope.is_system else None
        public = (models.Paper.visibility == PaperVisibility.PUBLIC.value) & (
            models.Paper.content_hash == content_hash
        )
        if self._scope.is_system:
            predicate = models.Paper.content_hash == content_hash
        else:
            private = (
                (models.Paper.visibility == PaperVisibility.PRIVATE.value)
                & (models.Paper.owner_id == owner)
                & (models.Paper.content_hash == content_hash)
            )
            predicate = public | private
        row = (
            await self._session.execute(select(models.Paper).where(predicate).limit(1))
        ).scalar_one_or_none()
        return _to_paper(row) if row is not None else None

    async def add(self, paper: Paper) -> Paper:
        row = models.Paper(
            id=_as_uuid(paper.id) or uuid.uuid4(),
            visibility=paper.visibility.value,
            owner_id=_as_uuid(paper.owner_id) if paper.owner_id else None,
            content_hash=paper.content_hash,
            external_ids=dict(paper.external_ids),
            title=paper.title,
            authors=list(paper.authors),
            year=paper.year,
            venue=paper.venue,
            abstract=paper.abstract,
            citation_count=paper.citation_count,
            parse_status=paper.parse_status.value,
            parse_failure_class=(
                paper.parse_failure_class.value if paper.parse_failure_class else None
            ),
            parser_version=paper.parser_version,
            tei_key=paper.tei_key,
            doi=paper.doi,
            arxiv_id=paper.arxiv_id,
            origin=paper.origin.value,
        )
        self._session.add(row)
        await self._session.flush()
        return _to_paper(row)

    async def find_by_identity(self, *, arxiv_id: str | None, doi: str | None) -> Paper | None:
        """This owner's paper with the arXiv id, else with the DOI (FR-I7, D132).

        Owner-scoped, and private rows only: a discovered paper is its adder's
        (D131), so another user's row for the same work is not this user's to
        reuse. The arXiv id is tried first because it is the identity the fetch
        path runs on (D130); a DOI match on a different row is the rarer case of
        one user holding the preprint and the published version separately.
        """
        if self._scope.is_system:
            raise PermissionError("find_by_identity is per owner; a system scope has no owner")
        owner = _as_uuid(self._tenant_id())
        mine = (models.Paper.visibility == PaperVisibility.PRIVATE.value) & (
            models.Paper.owner_id == owner
        )
        for column, value in ((models.Paper.arxiv_id, arxiv_id), (models.Paper.doi, doi)):
            if value is None:
                continue
            row = (
                await self._session.execute(select(models.Paper).where(mine & (column == value)))
            ).scalar_one_or_none()
            if row is not None:
                return _to_paper(row)
        return None

    async def set_content_hash(self, paper_id: str, content_hash: str) -> bool:
        """Write a fetched paper's hash, once (D134).

        `WHERE content_hash IS NULL` makes it write-once: a hash, once set, is
        which bytes this paper IS, and the partial unique indexes encoding §4.4.2
        are not something a later write may move it across. Scoped like `update`.
        """
        target = _as_uuid(paper_id)
        if target is None:
            return False
        affected = await self._affected_rows(
            update(models.Paper)
            .where(
                models.Paper.id == target,
                models.Paper.content_hash.is_(None),
                models.Paper.id.in_(self._reachable_papers()),
            )
            .values(content_hash=content_hash)
        )
        return affected > 0

    async def update(self, paper: Paper) -> bool:
        """Write back what a parse produced. Identity columns are not touched.

        Scoped like every other access here: the ingestion worker reaches a
        paper through the membership it is ingesting for, so the predicate costs
        it nothing and closes the alternative — an unscoped write that any scope
        could aim at any paper id.

        `content_hash`, `visibility` and `owner_id` are deliberately absent from
        the SET list. Changing one would change which paper this *is*, and the
        partial unique indexes encoding §4.4.2 are not something an update path
        should be able to move a row across.
        """
        target = _as_uuid(paper.id)
        if target is None:
            return False
        affected = await self._affected_rows(
            update(models.Paper)
            .where(models.Paper.id == target, models.Paper.id.in_(self._reachable_papers()))
            .values(
                title=paper.title,
                authors=list(paper.authors),
                abstract=paper.abstract,
                year=paper.year,
                venue=paper.venue,
                citation_count=paper.citation_count,
                external_ids=dict(paper.external_ids),
                parse_status=paper.parse_status.value,
                parse_failure_class=(
                    paper.parse_failure_class.value if paper.parse_failure_class else None
                ),
                parser_version=paper.parser_version,
                tei_key=paper.tei_key,
            )
        )
        return affected > 0

    async def add_to_workspace(
        self, workspace_id: str, paper_id: str, state: PaperState
    ) -> PaperMembership | None:
        workspace = _as_uuid(workspace_id)
        paper = _as_uuid(paper_id)
        if workspace is None or paper is None:
            return None
        # Membership may only be created in a workspace this scope owns —
        # otherwise a caller could graft a paper into someone else's workspace.
        owned = (
            await self._session.execute(
                self._visible_workspaces().where(models.Workspace.id == workspace)
            )
        ).scalar_one_or_none()
        if owned is None:
            return None

        row = models.WorkspacePaper(
            workspace_id=workspace, paper_id=paper, state=state.value, attempts=0
        )
        self._session.add(row)
        await self._session.flush()
        return PaperMembership(
            workspace_id=workspace_id,
            paper_id=paper_id,
            state=state,
            added_at=row.added_at,
        )

    async def set_membership_state(
        self,
        workspace_id: str,
        paper_id: str,
        state: PaperState,
        *,
        failure_class: str | None = None,
        failure_detail: str | None = None,
    ) -> bool:
        workspace = _as_uuid(workspace_id)
        paper = _as_uuid(paper_id)
        if workspace is None or paper is None:
            return False
        affected = await self._affected_rows(
            update(models.WorkspacePaper)
            .where(
                models.WorkspacePaper.workspace_id == workspace,
                models.WorkspacePaper.paper_id == paper,
                models.WorkspacePaper.workspace_id.in_(self._visible_workspaces()),
            )
            .values(
                state=state.value,
                failure_class=failure_class,
                failure_detail=failure_detail,
            )
        )
        return affected > 0

    async def remove_from_workspace(self, workspace_id: str, paper_id: str) -> bool:
        workspace = _as_uuid(workspace_id)
        paper = _as_uuid(paper_id)
        if workspace is None or paper is None:
            return False
        affected = await self._affected_rows(
            delete(models.WorkspacePaper).where(
                models.WorkspacePaper.workspace_id == workspace,
                models.WorkspacePaper.paper_id == paper,
                models.WorkspacePaper.workspace_id.in_(self._visible_workspaces()),
            )
        )
        return affected > 0

    async def has_any_membership(self, paper_id: str) -> bool:
        """Whether any workspace anywhere still holds this paper (FR-W2, AD-8).

        The one read in this class that is deliberately NOT scope-filtered. A
        scoped answer would report "no memberships" as soon as the *caller's*
        workspaces had none, and the caller would then delete derived data
        another tenant is still retrieving from. It returns a boolean about
        existence and never a row, so it cannot disclose whose workspace, what
        it is called, or that it is the same paper.
        """
        target = _as_uuid(paper_id)
        if target is None:
            return False
        found = (
            await self._session.execute(
                select(models.WorkspacePaper.paper_id)
                .join(
                    models.Workspace,
                    models.Workspace.id == models.WorkspacePaper.workspace_id,
                )
                .where(
                    models.WorkspacePaper.paper_id == target,
                    models.WorkspacePaper.state != PaperState.REMOVED.value,
                    # A soft-deleted workspace is on its way out (§5.6.4), so it
                    # does not count as a holder — otherwise a paper could never
                    # be purged while a deleted workspace lingered.
                    models.Workspace.deleted_at.is_(None),
                )
                .limit(1)
            )
        ).scalar_one_or_none()
        return found is not None

    async def delete(self, paper_id: str) -> bool:
        """Delete the paper row; sections and chunks cascade by foreign key."""
        target = _as_uuid(paper_id)
        if target is None:
            return False
        affected = await self._affected_rows(delete(models.Paper).where(models.Paper.id == target))
        return affected > 0

    async def queryable_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        target = _as_uuid(workspace_id)
        if target is None:
            return ()
        rows = (
            (
                await self._session.execute(
                    select(models.WorkspacePaper.paper_id).where(
                        models.WorkspacePaper.workspace_id == target,
                        models.WorkspacePaper.workspace_id.in_(self._visible_workspaces()),
                        models.WorkspacePaper.state == PaperState.INDEXED.value,
                    )
                )
            )
            .scalars()
            .all()
        )
        return tuple(str(r) for r in rows)

    async def membership_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        """Every paper the workspace holds, in any state (FR-W3's input).

        Neither filter `queryable_paper_ids` applies is wanted here. The state
        filter would skip papers that failed to ingest, which still own rows and
        blobs; the visibility filter would skip the workspace itself, which is
        soft-deleted by the time the cascade runs. Scoped by ownership instead,
        so the sweep sees what it has to delete (D92).
        """
        target = _as_uuid(workspace_id)
        if target is None:
            return ()
        rows = (
            (
                await self._session.execute(
                    select(models.WorkspacePaper.paper_id).where(
                        models.WorkspacePaper.workspace_id == target,
                        models.WorkspacePaper.workspace_id.in_(self._owned_workspaces()),
                    )
                )
            )
            .scalars()
            .all()
        )
        return tuple(str(r) for r in rows)

    async def orphaned_paper_ids(self) -> tuple[str, ...]:
        """Paper rows no workspace holds any more (D102).

        `NOT EXISTS` rather than `NOT IN`: the correlated form lets Postgres
        stop at the first membership instead of materialising every one, and it
        cannot be surprised by a NULL the way `NOT IN` silently can.
        """
        self._require_system("papers", "orphaned_paper_ids")
        held = select(models.WorkspacePaper.paper_id).where(
            models.WorkspacePaper.paper_id == models.Paper.id
        )
        rows = (
            (await self._session.execute(select(models.Paper.id).where(~held.exists())))
            .scalars()
            .all()
        )
        return tuple(str(r) for r in rows)

    async def all_paper_ids(self) -> tuple[str, ...]:
        """Every paper id in Postgres — the live set the sweep compares against."""
        self._require_system("papers", "all_paper_ids")
        rows = (await self._session.execute(select(models.Paper.id))).scalars().all()
        return tuple(str(r) for r in rows)


class SqlConversationRepository(ScopedRepository):
    """Conversations and their turns, scoped through the owning workspace (D59)."""

    async def create(self, workspace_id: str, *, title: str | None = None) -> str | None:
        target = _as_uuid(workspace_id)
        if target is None:
            return None
        # Membership in a workspace this scope owns is the whole predicate: a
        # conversation grafted into someone else's workspace would make every
        # later read of it legitimate-looking.
        owned = (
            await self._session.execute(
                self._visible_workspaces().where(models.Workspace.id == target)
            )
        ).scalar_one_or_none()
        if owned is None:
            return None

        row = models.Conversation(id=uuid.uuid4(), workspace_id=target, title=title)
        self._session.add(row)
        await self._session.flush()
        return str(row.id)

    async def get(self, conversation_id: str) -> Conversation | None:
        target = _as_uuid(conversation_id)
        if target is None:
            return None
        row = (
            await self._session.execute(
                select(models.Conversation).where(
                    models.Conversation.id == target,
                    models.Conversation.workspace_id.in_(self._visible_workspaces()),
                )
            )
        ).scalar_one_or_none()
        if row is None:
            return None

        turns = (
            (
                await self._session.execute(
                    select(models.Message)
                    .where(models.Message.conversation_id == target)
                    # By ordinal, not by timestamp: `created_at` defaults to
                    # `now()`, which is transaction time, so both turns of one
                    # exchange share it exactly.
                    .order_by(models.Message.ordinal)
                )
            )
            .scalars()
            .all()
        )
        return Conversation(
            id=str(row.id),
            workspace_id=str(row.workspace_id),
            title=row.title,
            turns=[_to_turn(t) for t in turns],
            created_at=row.created_at,
        )

    async def append(self, conversation_id: str, turn: Turn) -> str | None:
        target = _as_uuid(conversation_id)
        if target is None:
            return None
        reachable = (
            await self._session.execute(
                select(models.Conversation.id).where(
                    models.Conversation.id == target,
                    models.Conversation.workspace_id.in_(self._visible_workspaces()),
                )
            )
        ).scalar_one_or_none()
        if reachable is None:
            return None

        # The next position, read inside the caller's transaction. The unique
        # index on (conversation_id, ordinal) is what turns a lost race into an
        # IntegrityError rather than two messages claiming one position.
        used = (
            await self._session.execute(
                select(func.max(models.Message.ordinal)).where(
                    models.Message.conversation_id == target
                )
            )
        ).scalar_one_or_none()

        row = models.Message(
            id=uuid.uuid4(),
            conversation_id=target,
            ordinal=0 if used is None else used + 1,
            role=turn.role.value,
            content=turn.content,
            # Stored as the scored chunks the model was shown, which is what
            # FR-Q6 makes inspectable — not the rendered passages, which are
            # reconstructable from the chunk rows and would duplicate them.
            evidence={
                "chunks": [
                    {"chunk_id": c.chunk_id, "paper_id": c.paper_id, "score": c.score}
                    for c in turn.evidence
                ]
            }
            if turn.evidence
            else None,
            citations=[
                {
                    "label": c.label,
                    "chunk_id": c.chunk_id,
                    "status": c.status.value,
                    "claim_text": c.claim_text,
                }
                for c in turn.citations
            ]
            or None,
            prompt_version=turn.prompt_version,
            provider=turn.provider,
            model=turn.model,
            latency_ms=turn.timings.as_dict() if turn.timings else None,
            token_usage=(
                {"prompt": turn.prompt_tokens, "completion": turn.completion_tokens}
                if turn.prompt_tokens is not None or turn.completion_tokens is not None
                else None
            ),
            degraded=list(turn.degraded) or None,
        )
        self._session.add(row)
        await self._session.flush()
        return str(row.id)


class SqlSectionRepository(ScopedRepository):
    async def replace_for_paper(self, paper_id: str, sections: Sequence[Section]) -> int:
        """Delete this paper's sections, then write the new ones (D52).

        The delete cascades to `chunks`, which is what makes re-parsing
        converge: section ids are UUIDs, so a second parse cannot upsert onto
        the first, and appending would leave two copies of every section with
        chunks attached to whichever the path mapping happened to choose.
        """
        target = _as_uuid(paper_id)
        if target is None:
            return 0
        # Scoped, because this one *deletes* — and cascades to `chunks`. An
        # unscoped variant would let any scope that guessed a paper id wipe
        # another tenant's parsed content, which is a worse outcome than the
        # read leaks layer 2 is usually guarding against.
        await self._session.execute(
            delete(models.Section).where(
                models.Section.paper_id == target,
                models.Section.paper_id.in_(self._reachable_papers()),
            )
        )
        if not await self._can_reach(target):
            return 0
        for section in sections:
            self._session.add(
                models.Section(
                    id=_as_uuid(section.id) or uuid.uuid4(),
                    paper_id=target,
                    parent_id=_as_uuid(section.parent_id) if section.parent_id else None,
                    kind=section.kind.value,
                    raw_title=section.raw_title,
                    path=section.path,
                    order_index=section.order_index,
                    page_start=section.page_start,
                    page_end=section.page_end,
                    text=section.text,
                )
            )
        await self._session.flush()
        return len(sections)

    async def list_for_paper(self, paper_id: str) -> list[Section]:
        target = _as_uuid(paper_id)
        if target is None:
            return []
        rows = (
            (
                await self._session.execute(
                    select(models.Section)
                    .where(
                        models.Section.paper_id == target,
                        models.Section.paper_id.in_(self._reachable_papers()),
                    )
                    .order_by(models.Section.order_index)
                )
            )
            .scalars()
            .all()
        )
        return [_to_section(row) for row in rows]

    async def _can_reach(self, paper_id: uuid.UUID) -> bool:
        """Whether this scope holds a membership for the paper.

        Checked before the insert half of `replace_for_paper`: the delete is
        already filtered, but writing rows for a paper the scope cannot reach
        would leave orphan sections nothing can read or clean up.
        """
        found = (
            await self._session.execute(
                self._reachable_papers().where(models.WorkspacePaper.paper_id == paper_id).limit(1)
            )
        ).scalar_one_or_none()
        return found is not None


class SqlChunkRepository(ScopedRepository):
    async def add_many(self, chunks: Sequence[Chunk], *, section_ids: Mapping[str, str]) -> int:
        written = 0
        for chunk in chunks:
            section_id = section_ids.get(chunk.provenance.section_path)
            if section_id is None:
                raise KeyError(
                    f"no section id supplied for path {chunk.provenance.section_path!r} "
                    f"(chunk {chunk.id!r})"
                )
            self._session.add(
                models.Chunk(
                    id=chunk.id,
                    paper_id=_as_uuid(chunk.provenance.paper_id),
                    section_id=_as_uuid(section_id),
                    parent_id=chunk.parent_id,
                    role=chunk.role.value,
                    ordinal=chunk.ordinal,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page=chunk.provenance.page,
                    char_start=chunk.provenance.char_start,
                    char_end=chunk.provenance.char_end,
                    chunker_version=chunk.chunker_version,
                    embedder_version=chunk.embedder_version,
                )
            )
            written += 1
        await self._session.flush()
        return written

    async def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        rows = (
            await self._session.execute(
                select(models.Chunk, models.Section)
                .join(models.Section, models.Section.id == models.Chunk.section_id)
                .where(
                    models.Chunk.id.in_(list(chunk_ids)),
                    models.Chunk.paper_id.in_(self._reachable_papers()),
                )
            )
        ).all()
        return [_to_chunk(chunk, section) for chunk, section in rows]

    async def stamp_embedded(self, chunk_ids: Sequence[str], version: str) -> int:
        """Mark chunks as having vectors, after the upsert succeeded (D53)."""
        if not chunk_ids:
            return 0
        return await self._affected_rows(
            update(models.Chunk)
            .where(
                models.Chunk.id.in_(list(chunk_ids)),
                models.Chunk.paper_id.in_(self._reachable_papers()),
            )
            .values(embedder_version=version)
        )

    async def list_for_paper(self, paper_id: str) -> list[Chunk]:
        target = _as_uuid(paper_id)
        if target is None:
            return []
        rows = (
            await self._session.execute(
                select(models.Chunk, models.Section)
                .join(models.Section, models.Section.id == models.Chunk.section_id)
                .where(
                    models.Chunk.paper_id == target,
                    models.Chunk.paper_id.in_(self._reachable_papers()),
                )
                .order_by(models.Chunk.ordinal)
            )
        ).all()
        return [_to_chunk(chunk, section) for chunk, section in rows]


# -- row -> domain -----------------------------------------------------------
#
# Mapping lives here rather than on the models so the ORM stays a persistence
# detail. A domain entity that knew how to load itself would make the layering
# claim false.


def _to_workspace(row: models.Workspace, memberships: list[PaperMembership]) -> Workspace:
    return Workspace(
        id=str(row.id),
        owner_id=str(row.owner_id),
        name=row.name,
        state=WorkspaceState(row.state),
        memberships=memberships,
        created_at=row.created_at,
        deleted_at=row.deleted_at,
    )


def _to_paper(row: models.Paper) -> Paper:
    return Paper(
        id=str(row.id),
        content_hash=row.content_hash,
        visibility=PaperVisibility(row.visibility),
        title=row.title,
        authors=tuple(row.authors or ()),
        year=row.year,
        venue=row.venue,
        abstract=row.abstract,
        citation_count=row.citation_count,
        external_ids=dict(row.external_ids or {}),
        owner_id=str(row.owner_id) if row.owner_id else None,
        parse_status=ParseStatus(row.parse_status),
        parse_failure_class=(
            FailureClass(row.parse_failure_class) if row.parse_failure_class else None
        ),
        parser_version=row.parser_version,
        tei_key=row.tei_key,
        doi=row.doi,
        arxiv_id=row.arxiv_id,
        origin=PaperOrigin(row.origin),
        created_at=row.created_at,
    )


def _to_section(row: models.Section) -> Section:
    return Section(
        id=str(row.id),
        paper_id=str(row.paper_id),
        kind=SectionKind(row.kind),
        raw_title=row.raw_title,
        path=row.path,
        order_index=row.order_index,
        page_start=row.page_start,
        page_end=row.page_end,
        text=row.text,
        parent_id=str(row.parent_id) if row.parent_id else None,
    )


def _to_turn(row: models.Message) -> Turn:
    """One message row back to a domain turn.

    Citations are rebuilt through the `Citation` constructor rather than around
    it, so its invariant — `chunk_id` is None exactly when the label was
    fabricated — is re-checked on the way out. A row that violated it would
    otherwise become an unresolvable citation the UI renders as clickable.
    """
    citations = tuple(
        Citation(
            label=str(c["label"]),
            chunk_id=c.get("chunk_id"),
            claim_text=str(c.get("claim_text", "")),
            status=CitationStatus(c["status"]),
        )
        for c in (row.citations or [])
    )
    evidence = tuple(
        ScoredChunk(
            chunk_id=str(c["chunk_id"]),
            paper_id=str(c["paper_id"]),
            score=float(c["score"]),
            arm=RetrievalArm.DENSE,
        )
        for c in ((row.evidence or {}).get("chunks") or [])
    )
    latency = row.latency_ms or {}
    usage = row.token_usage or {}
    return Turn(
        role=TurnRole(row.role),
        content=row.content,
        citations=citations,
        evidence=evidence,
        degraded=tuple(row.degraded or ()),
        prompt_version=row.prompt_version,
        provider=row.provider,
        model=row.model,
        timings=StageTimings(
            retrieval_ms=int(latency.get("retrieval_ms", 0)),
            assembly_ms=int(latency.get("assembly_ms", 0)),
            first_token_ms=int(latency.get("first_token_ms", 0)),
            generation_ms=int(latency.get("generation_ms", 0)),
        )
        if latency
        else None,
        prompt_tokens=usage.get("prompt"),
        completion_tokens=usage.get("completion"),
        created_at=row.created_at,
    )


def _to_chunk(row: models.Chunk, section: models.Section) -> Chunk:
    return Chunk(
        id=row.id,
        text=row.text,
        role=ChunkRole(row.role),
        ordinal=row.ordinal,
        provenance=Provenance(
            paper_id=str(row.paper_id),
            section_path=section.path,
            section_kind=SectionKind(section.kind),
            page=row.page,
            char_start=row.char_start,
            char_end=row.char_end,
        ),
        chunker_version=row.chunker_version,
        token_count=row.token_count,
        embedder_version=row.embedder_version,
        parent_id=row.parent_id,
    )
