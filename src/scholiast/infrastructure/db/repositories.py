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

from sqlalchemy import CursorResult, Executable, Select, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.tenancy import Scope, TenantIsolationViolation
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Paper,
    PaperVisibility,
    ParseStatus,
    Provenance,
    SectionKind,
)
from scholiast.domain.workspace import (
    PaperMembership,
    PaperState,
    Workspace,
    WorkspaceState,
)
from scholiast.infrastructure.db import models


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
        )
        self._session.add(row)
        await self._session.flush()
        return _to_paper(row)

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
