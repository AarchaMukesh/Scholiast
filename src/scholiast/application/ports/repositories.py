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

from scholiast.domain.conversation import Conversation, Turn
from scholiast.domain.paper import Chunk, Paper, Section
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

    async def update(self, paper: Paper) -> bool:
        """Persist a paper's mutable metadata — what a parse produced.

        A whole entity rather than a keyword-per-column setter: a parse changes
        title, authors, abstract, `parse_status`, `parse_failure_class`,
        `parser_version` and `tei_key` together, and the domain invariant tying
        status to failure class is checked on the entity. Splitting them into
        setters would let a caller persist half a parse.

        Identity fields (`content_hash`, `visibility`, `owner_id`) are not
        written: changing one would change which paper this is, and the partial
        unique indexes that encode §4.4.2 are not something an update path
        should be able to move a row across.
        """
        ...

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

    async def remove_from_workspace(self, workspace_id: str, paper_id: str) -> bool:
        """Drop a membership, reporting whether one was there (FR-W2).

        False for "not a member" rather than an error: removal is idempotent
        (NFR-R5), and a retried delete must not fail for having already
        succeeded.
        """
        ...

    async def has_any_membership(self, paper_id: str) -> bool:
        """Whether ANY workspace still holds this paper — across all tenants.

        Deliberately not scope-filtered, and it is the one read here that must
        not be. §4.4.2's registry shares a public paper across tenants and AD-8
        exists so it is parsed once; a scoped answer would say "no memberships"
        whenever the *caller's* workspaces had none, and the caller would then
        delete a paper another tenant is still reading from. Safe because it
        returns a boolean about existence, never a row: it cannot leak what the
        other workspace is, who owns it, or that it is the same paper.
        """
        ...

    async def delete(self, paper_id: str) -> bool:
        """Delete the paper row, cascading to its sections and chunks.

        Only called once `has_any_membership` is False. The cascade is the
        database's (`ON DELETE CASCADE`), so there is no window where sections
        outlive their paper.
        """
        ...

    async def queryable_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        """Exactly the corpus a query may retrieve from — the D36 filter input.

        This is where layer 3's mandatory vector filter comes from, which is why
        it is a scoped query rather than a value the caller supplies: the id set
        is *derived from* the tenant predicate rather than passed alongside it.
        """
        ...


@runtime_checkable
class ConversationRepository(Protocol):
    """Conversations and their turns (FR-Q6, FR-Q9, §5.4.3, D59).

    A turn is persisted in full — evidence, citations, prompt version, provider
    and model, per-stage latency and token usage. The accounting is not
    decoration: AP-8's claim is that a stage breakdown turns "it's slow" into
    "reranking is 60% of p95", and M2's latency profiling needs those numbers to
    have been recorded all along rather than retrofitted onto messages that
    never had them.
    """

    async def create(self, workspace_id: str, *, title: str | None = None) -> str | None:
        """Start a conversation, returning its id, or None if out of scope."""
        ...

    async def get(self, conversation_id: str) -> Conversation | None:
        """A conversation with its turns, or None if the scope cannot reach it."""
        ...

    async def append(self, conversation_id: str, turn: Turn) -> str | None:
        """Record one turn, returning its message id, or None if out of scope."""
        ...


@runtime_checkable
class SectionRepository(Protocol):
    """Parsed sections — the rows chunk offsets are measured against (D42).

    Separate from `PaperRepository` for the same reason `ChunkRepository` is:
    sections live under a paper but are written and read on their own schedule,
    and one repository per entity keeps the scoped-query surface small enough to
    audit.
    """

    async def replace_for_paper(self, paper_id: str, sections: Sequence[Section]) -> int:
        """Replace this paper's sections wholesale, returning how many were written.

        Replace rather than insert, because re-parsing must *converge* rather
        than accumulate (FR-I7). Section ids are UUIDs — unlike chunk ids, which
        are deterministic — so a second parse cannot upsert onto the first, and
        appending would leave the paper with two copies of every section and
        chunks pointing at whichever the mapping happened to pick. The
        `ON DELETE CASCADE` from `sections` to `chunks` makes the replacement
        total: stale chunks cannot outlive the section they indexed.
        """
        ...

    async def list_for_paper(self, paper_id: str) -> list[Section]:
        """This paper's sections in document order, or empty if out of scope.

        Carries `text`, because this is what resolves a citation: chunk → its
        section row → slice (D42), with no second store on the request path.
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

    async def stamp_embedded(self, chunk_ids: Sequence[str], version: str) -> int:
        """Record that these chunks now have vectors (NFR-M5, D53).

        The second half of a two-phase write. Chunks are persisted with
        `embedder_version` NULL, the vectors are upserted, and only then are the
        rows stamped — so `Chunk.is_indexed` means "a vector exists for this
        chunk" rather than "we intended to make one". A crash between the two
        leaves vectors with unstamped rows, which is the orphan direction
        §5.6.4's sweep detects; the reverse ordering would leave a chunk that
        claims to be searchable and is not, with nothing reporting it.
        """
        ...

    async def list_for_paper(self, paper_id: str) -> list[Chunk]: ...
