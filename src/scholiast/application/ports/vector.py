"""Vector index port.

Layer 3 of the tenancy design (§5.8) lives in this interface's *shape*, not in
its implementations: `search` takes `paper_ids` as a required argument, and an
empty set is an error rather than an unfiltered search. A caller cannot forget
the filter, because there is no overload without it.

The id set is produced by `PaperRepository.queryable_paper_ids()`, which already
applies the tenant predicate — so per D36 it is *derived from* the tenant filter
rather than passed alongside it, and a caller cannot widen it by supplying
something of its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from scholiast.domain.paper import SectionKind
from scholiast.domain.retrieval import ScoredChunk


class EmptyRetrievalScope(Exception):
    """A search was attempted with no papers in scope.

    Deliberately an error and not an empty result. An empty filter set is the
    one input that would turn a workspace-scoped ANN query into a global one, so
    it must be impossible to reach by accident — and a caller that has nothing to
    search should have checked `Workspace.is_queryable` and said so (FR-W5),
    rather than asking the index and getting a confusing silence.
    """

    def __init__(self) -> None:
        super().__init__(
            "vector search requires at least one paper id — an unfiltered ANN query "
            "would cross tenant boundaries (§5.8 layer 3, D36)"
        )


class VectorDimensionMismatch(Exception):
    """The existing collection was built for a different embedder.

    Changing the embedding model changes the vector width, and a collection is
    fixed at creation. Without this check the mismatch surfaces as a per-batch
    upsert error in the middle of ingestion; with it, the process refuses to
    start and says exactly which two numbers disagree.
    """

    def __init__(self, collection: str, existing: int, configured: int) -> None:
        super().__init__(
            f"collection {collection!r} stores {existing}-dim vectors but the "
            f"configured embedder produces {configured}-dim. Re-index into a new "
            f"collection, or point at the collection this embedder built."
        )
        self.existing = existing
        self.configured = configured


@dataclass(frozen=True, slots=True)
class VectorRecord:
    """One embedded chunk, ready to index.

    The version stamps are payload rather than metadata: NFR-M5 wants "re-embed
    only the chunks produced by embedder v1" to be a *query*, which means the
    stamp has to be filterable in the store that holds the vectors, not only in
    Postgres.
    """

    chunk_id: str
    paper_id: str
    vector: tuple[float, ...]
    section_kind: SectionKind
    chunker_version: str
    embedder_version: str


@runtime_checkable
class VectorIndex(Protocol):
    """Dense index over chunk embeddings, filterable by paper."""

    async def ensure_ready(self) -> None:
        """Create the collection and payload indexes, or validate what exists.

        Idempotent: called on startup and safe to call again. The payload index
        on `paper_id` is not optional — without it the mandatory filter degrades
        into a full scan, which is the failure mode AD-4 chose Qdrant to avoid.

        Raises `VectorDimensionMismatch` when a collection already exists with a
        different vector width than the configured embedder produces. Failing at
        startup, naming both numbers, beats the alternative: no error here and
        then every upsert failing deep inside ingestion, far from the cause.
        """
        ...

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        """Index records, replacing any that already exist.

        Upsert rather than insert because chunk ids are deterministic (D46): a
        re-index of the same paper must converge on the same points instead of
        accumulating duplicates.
        """
        ...

    async def search(
        self,
        vector: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[ScoredChunk]:
        """Nearest neighbours, restricted to `paper_ids`.

        Raises `EmptyRetrievalScope` when `paper_ids` is empty.
        """
        ...

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        """Remove every vector belonging to these papers (FR-W2, §5.6.4)."""
        ...

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        """Points in the index, optionally restricted to some papers.

        `None` means the whole collection — used by the reconciliation sweep,
        which legitimately spans tenants, never by a request path.
        """
        ...
