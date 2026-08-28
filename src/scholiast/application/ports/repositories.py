"""Repository ports.

These speak **domain entities**, never ORM rows. That is what makes "swap the
database" an adapter change, and it is the reason use cases can be tested without
a running Postgres.

Every one of these is constructed with a `Scope` (doc 5 §5.8 layer 2). There is
no unscoped variant, and that is deliberate: the realistic tenancy failure is not
a missing login page, it is a developer adding a query and forgetting the
predicate. A repository that *cannot* be built without a scope turns that from a
code-review question into a type error.

`SystemContext` is the single sanctioned way to read across tenants, and it
demands a stated reason so the full set of cross-tenant operations stays
greppable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from scholiast.domain.paper import Chunk, Paper
from scholiast.domain.workspace import PaperMembership, PaperState, Workspace


@runtime_checkable
class WorkspaceRepository(Protocol):
    """Workspaces owned by the current scope (FR-W1)."""

    async def create(self, workspace: Workspace) -> Workspace: ...

    async def get(self, workspace_id: str) -> Workspace | None:
        """The workspace with its memberships, or None if it is not in scope.

        Returns None rather than raising for a workspace owned by someone else:
        an out-of-scope id and a nonexistent id must be indistinguishable, or the
        API leaks existence (FR-A2's 404-not-403 requirement).
        """
        ...

    async def list_all(self) -> list[Workspace]: ...

    async def rename(self, workspace_id: str, name: str) -> bool: ...

    async def soft_delete(self, workspace_id: str) -> bool:
        """Mark DELETING and stamp `deleted_at` (§5.6.4 step 1)."""
        ...


@runtime_checkable
class PaperRepository(Protocol):
    """Papers reachable by the current scope, and workspace membership."""

    async def get(self, paper_id: str) -> Paper | None:
        """A paper the scope can reach through one of its workspaces.

        Reachability is membership, not ownership: §4.4.2 option B's stated cost
        is that "isolation now depends on filtering by the caller's permitted
        `paper_id` set". A public paper another tenant ingested is *not*
        retrievable here just because it is public.
        """
        ...

    async def find_by_content_hash(self, content_hash: str) -> Paper | None:
        """Ingestion-time dedup lookup (FR-I7).

        Deliberately *not* membership-scoped: re-adding a paper to a different
        workspace must not re-parse or re-embed it (§4.4.2), which requires
        seeing a public registry entry the caller has no membership for.

        This is safe rather than an exception to the rule, because the predicate
        mirrors the two partial unique indexes: a public row by hash, or a
        private row by (owner, hash). It cannot return another tenant's private
        paper — the case where a global lookup *would* be a privacy failure.
        """
        ...

    async def add(self, paper: Paper) -> Paper: ...

    async def add_to_workspace(
        self, workspace_id: str, paper_id: str, state: PaperState
    ) -> PaperMembership | None: ...

    async def set_membership_state(
        self,
        workspace_id: str,
        paper_id: str,
        state: PaperState,
        *,
        failure_class: str | None = None,
        failure_detail: str | None = None,
    ) -> bool: ...

    async def queryable_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        """Exactly the corpus a query may retrieve from — the D36 filter input.

        This is where layer 3's mandatory vector filter comes from, which is why
        it is a scoped query rather than a value the caller supplies: the id set
        is *derived from* the tenant predicate rather than passed alongside it.
        """
        ...


@runtime_checkable
class ChunkRepository(Protocol):
    """Chunks, reachable only through a paper the scope can reach."""

    async def add_many(self, chunks: Sequence[Chunk], *, section_ids: Mapping[str, str]) -> int:
        """Persist chunks, mapping each provenance path to its section id.

        `section_ids` is supplied by the caller rather than looked up by path.
        The caller has just created those sections and knows their ids, and
        resolving by path here would silently mis-assign chunks for two sections
        that share a path.
        """
        ...

    async def get_many(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        """Fetch by id, dropping any the scope cannot reach.

        Silently dropping is correct here: these ids arrive from a vector search
        that is already workspace-filtered, so an unreachable id means the two
        stores disagree — and serving it would be the leak layer 2 exists to
        stop.
        """
        ...

    async def list_for_paper(self, paper_id: str) -> list[Chunk]: ...
