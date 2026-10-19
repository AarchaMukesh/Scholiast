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

from collections.abc import Mapping, Sequence
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
    #: The chunk's BM25 term weights and the version that produced them (D172).
    #: Production always supplies both; a dense-only collection ignores them, and
    #: a hybrid one writes them in the same point as the dense vector, so no point
    #: is ever half-written. None where a caller has no lexical arm (the eval,
    #: which attaches its weights in a second pass).
    lexical: Mapping[int, float] | None = None
    lexical_version: str | None = None


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

    async def ping(self) -> bool:
        """Whether the index is reachable. For FR-O2's readiness probe.

        Deliberately not `count()`: an exact count is work proportional to the
        collection, and a readiness probe runs every few seconds forever. What
        readiness needs to know is whether the store answers at all, which is a
        metadata call.

        Returns False rather than raising, because a probe that 500s tells the
        caller less than a probe that says which dependency is down.
        """
        ...

    async def count_orphans(self, live_paper_ids: Sequence[str]) -> int:
        """Vectors whose paper is no longer in Postgres (§5.6.4 step 4, D102).

        Separate from the delete so the sweep can report residue it did not
        create, which is the difference between a safety net and a silent one.
        """
        ...

    async def delete_orphans(self, live_paper_ids: Sequence[str]) -> int:
        """Remove every vector whose paper is not in `live_paper_ids`.

        Phrased as "delete what is NOT in this set" rather than enumerate-diff-
        delete because the index can answer it with a negated filter in one
        round trip, while enumerating pages the whole collection to find a
        difference that is normally empty.

        **An empty set means every vector is an orphan**, because Postgres is
        the source of truth (NFR-R6) and it is saying there are no papers at
        all. An adapter that reads empty as "nothing to do" inverts the
        guarantee this method exists to enforce.
        """
        ...


@dataclass(frozen=True, slots=True)
class SparseRecord:
    """One chunk's lexical term weights, ready to index (doc 6 section 6.5, D73).

    `weights` maps a term id to its BM25 term-frequency contribution. The IDF
    factor is deliberately absent: the store applies it from collection
    statistics, so the weights here stay correct as the corpus grows.
    """

    chunk_id: str
    paper_id: str
    weights: dict[int, float]


@runtime_checkable
class LexicalIndex(Protocol):
    """Sparse BM25 retrieval over the same chunks (FR-Q3, ML-6).

    **A separate interface over the SAME store, not a second store.** Doc 6
    section 6.5 adopted sparse-in-the-vector-store over Elasticsearch precisely
    for "single store, single filter path, single query round-trip" -- so this is
    implemented by the same adapter as `VectorIndex`, against the same collection,
    with the same mandatory paper filter. It is a distinct Protocol rather than
    extra methods on `VectorIndex` because the dense-only path must not acquire
    methods it never calls: production retrieves dense-only unless
    `retrieval_mode` says hybrid (D172), and a dense-only collection has no lexical
    vector to search.

    The same tenancy shape applies for the same reason (section 5.8 layer 3, D36):
    `paper_ids` is required and an empty set is an error, never an unfiltered scan.
    """

    async def upsert_sparse(self, records: Sequence[SparseRecord]) -> int:
        """Index lexical weights, replacing any that already exist."""
        ...

    async def search_lexical(
        self,
        weights: dict[int, float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[ScoredChunk]:
        """Best BM25 matches, restricted to `paper_ids`.

        Raises `EmptyRetrievalScope` when `paper_ids` is empty.
        """
        ...

    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        """Every chunk id whose point carries lexical weights of this version.

        For the completeness check a hybrid read waits on (D172): compared by id
        against the chunks Postgres records as embedded, so one missing point is
        found, not averaged away. Read-only, and proportional to the collection,
        so it runs at startup and in a sweep, never per request.
        """
        ...

    async def count_lexical(self, *, lexical_version: str) -> int:
        """How many points carry lexical weights of this version.

        The cheap half of the same check, for the readiness probe: a count, not
        an enumeration.
        """
        ...
