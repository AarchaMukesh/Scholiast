"""Qdrant adapter — single collection, filtered by paper (AD-4, D36, D46).

**One collection, not one per workspace.** §5.10: tenant count is the dimension
expected to grow, and thousands of tiny HNSW graphs waste memory badly.
Filterable ANN over one index is the capability Qdrant was chosen for, and the
payload index on `paper_id` is what makes the filter selective rather than a
scan — without it the choice of Qdrant buys nothing.

**The filter is not optional.** `search` refuses an empty paper set instead of
falling through to a global query, which is §5.8 layer 3 expressed as control
flow rather than as a convention someone has to remember.

**Point ids are `uuid5(namespace, chunk_id)`** (D46). Qdrant takes only integers
or UUIDs; chunk ids are deterministic strings. UUID5 keeps re-indexing
idempotent — the same chunk always lands on the same point — and the payload
carries `chunk_id` because the mapping is one-way.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from qdrant_client import AsyncQdrantClient, models

from scholiast.application.ports.vector import (
    EmptyRetrievalScope,
    SparseRecord,
    VectorDimensionMismatch,
    VectorRecord,
)
from scholiast.domain.retrieval import RetrievalArm, ScoredChunk

#: Fixed namespace for D46's mapping. It must never change: a new namespace
#: silently relocates every point, so a re-index would duplicate the entire
#: corpus rather than replace it.
POINT_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

#: Payload keys. Named constants because a typo in a filter key does not error —
#: Qdrant simply matches nothing, and a retrieval that returns nothing looks
#: exactly like a workspace with no relevant chunks.
PAYLOAD_CHUNK_ID = "chunk_id"
PAYLOAD_PAPER_ID = "paper_id"
PAYLOAD_SECTION_KIND = "section_kind"
PAYLOAD_CHUNKER_VERSION = "chunker_version"
PAYLOAD_EMBEDDER_VERSION = "embedder_version"
#: Written on hybrid points only (D172); a dense-only collection never carries it.
PAYLOAD_LEXICAL_VERSION = "lexical_version"

#: Named-vector keys, used ONLY when the adapter is constructed with `sparse=True`.
#: A dense-only collection keeps M1's unnamed vector, so production's schema is
#: untouched by the lexical arm existing (D73).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "lexical"


def point_id_for(chunk_id: str) -> str:
    """Deterministic point id for a chunk (D46)."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


#: Stands in for "no live papers" in the orphan filter. Not a UUID, so it
#: cannot collide with a real paper id, and it is never written to the index.
_NO_SUCH_PAPER = "__no_live_papers__"


class CollectionSchemaMismatch(Exception):
    """An existing collection lacks a vector this adapter reads, or its shape is unreadable.

    Raised by `existing_count`, whose callers read a collection exactly as it was
    built (D168): querying one that is missing the lexical vector, or that has a
    dense vector of the wrong form, would fail or silently differ, so it is
    refused before anything is read from it.
    """

    def __init__(self, collection: str, problems: Sequence[str]) -> None:
        self.collection = collection
        self.problems = tuple(problems)
        super().__init__(f"collection {collection!r}: " + "; ".join(self.problems))


class CollectionNotFound(Exception):
    """A collection this process reads or mirrors to, but must not create, is absent (D172)."""

    def __init__(self, collection: str) -> None:
        self.collection = collection
        super().__init__(
            f"collection {collection!r} does not exist, and this process does not create "
            "it: `scholiast-reindex --hybrid` builds it from Postgres (D172)"
        )


class QdrantVectorIndex:
    """The `VectorIndex` adapter."""

    def __init__(
        self,
        url: str,
        *,
        collection: str = "scholiast_chunks",
        dimensions: int = 768,
        timeout: float = 30.0,
        sparse: bool = False,
    ) -> None:
        self._url = url
        self._timeout = int(timeout)
        self._collection = collection
        self._dimensions = dimensions
        # Opt-in, defaulting OFF, and it changes the collection SCHEMA: a sparse
        # collection uses named vectors, a dense-only one uses the unnamed vector
        # M1 built. Two code paths rather than one conditional, so
        # `scholiast_chunks` keeps the exact shape it already has; production's
        # hybrid arm reads a separate collection built with this on (D172).
        self._sparse = sparse
        self._connection: AsyncQdrantClient | None = None

    @property
    def collection(self) -> str:
        return self._collection

    @property
    def sparse(self) -> bool:
        return self._sparse

    @property
    def _client(self) -> AsyncQdrantClient:
        """Connect on first use, not at construction.

        `AsyncQdrantClient.__init__` performs a server version check, so
        building one eagerly turns constructing an adapter into a network call —
        which NFR-T3 forbids in the default test run, and which would make
        dependency wiring fail at import time if the store were briefly down.

        The version check itself is worth keeping: a client/server minor gap
        works for simple calls and fails on the ones added in between, which is
        how it caught the 1.15-vs-1.18 mismatch during setup.
        """
        if self._connection is None:
            self._connection = AsyncQdrantClient(url=self._url, timeout=self._timeout)
        return self._connection

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def ensure_ready(self) -> None:
        if await self._client.collection_exists(self._collection):
            self._assert_dimensions(await self._client.get_collection(self._collection))
        elif self._sparse:
            await self._client.create_collection(
                self._collection,
                # Named vectors, because one collection now holds two
                # representations of the same chunk — doc §6.5's "single store,
                # single filter path, single query round-trip", which is the whole
                # reason sparse-in-Qdrant beat running Elasticsearch alongside.
                vectors_config={
                    DENSE_VECTOR: models.VectorParams(
                        size=self._dimensions, distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    SPARSE_VECTOR: models.SparseVectorParams(
                        # The store supplies IDF from its own statistics (D73).
                        # Computing it client-side would freeze every document's
                        # weights against the corpus as it stood at index time, so
                        # each later insert would quietly make older scores wrong.
                        modifier=models.Modifier.IDF,
                    )
                },
            )
        else:
            await self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimensions,
                    # BGE is trained for cosine similarity; using dot or
                    # euclidean here would silently degrade every ranking.
                    distance=models.Distance.COSINE,
                ),
                # Deliberately NOT quantized in M1. §6.4: at demo scale int8 is
                # not required, "which is exactly why it should be measured and
                # reported as a scaling capability rather than adopted on
                # faith". D33 makes it required for the deployed instance, where
                # it becomes a measured change against this baseline.
            )

        # The index that makes the mandatory filter selective. Creating it is
        # idempotent, and doing it here rather than at collection creation means
        # an index added later still gets built on an existing deployment.
        await self._client.create_payload_index(
            self._collection,
            field_name=PAYLOAD_PAPER_ID,
            field_schema=models.PayloadSchemaType.KEYWORD,
            wait=True,
        )

    def _assert_dimensions(self, info: models.CollectionInfo) -> None:
        """Refuse to run against a collection built for a different embedder.

        Read defensively: Qdrant's config shape differs between unnamed and
        named vector configurations, and a check that silently found nothing
        would be worse than no check at all — it would look like validation.
        """
        params = getattr(getattr(info.config, "params", None), "vectors", None)
        # Unnamed config: `vectors` IS the VectorParams. Named config (what the
        # sparse path creates): `vectors` is a dict keyed by vector name, and
        # reaching for `.size` on it returns None — which would make this method
        # look like validation while checking nothing, the exact failure the
        # docstring above warns about. Both shapes are read.
        if isinstance(params, dict):
            existing = getattr(params.get(DENSE_VECTOR), "size", None)
        else:
            existing = getattr(params, "size", None)
        if isinstance(existing, int) and existing != self._dimensions:
            raise VectorDimensionMismatch(self._collection, existing, self._dimensions)

    def _assert_schema(self, info: models.CollectionInfo) -> None:
        """Strictly: the collection holds exactly the vectors this adapter reads.

        For `existing_count`. `_assert_dimensions`, which `ensure_ready` uses, is
        left lenient on purpose, so production's path is unchanged; this one
        refuses an unreadable shape instead of passing it, because a check that
        finds nothing must not look like validation.

        - Dense-only: one unnamed vector of `dimensions`, what `search` queries.
        - Sparse: a named `dense` vector of `dimensions` and a named `lexical`
          sparse vector with Qdrant's IDF modifier (D73), what `search` and
          `search_lexical` query. Without the modifier the lexical arm would score
          by term frequency alone and not be the arm M2 measured.
        """
        params = getattr(info.config, "params", None)
        vectors = getattr(params, "vectors", None)
        problems: list[str] = []
        if self._sparse:
            dense = vectors.get(DENSE_VECTOR) if isinstance(vectors, dict) else None
            if dense is None:
                problems.append(f"no named {DENSE_VECTOR!r} vector")
            lexical = (getattr(params, "sparse_vectors", None) or {}).get(SPARSE_VECTOR)
            if lexical is None:
                problems.append(f"no {SPARSE_VECTOR!r} sparse vector")
            elif getattr(lexical, "modifier", None) != models.Modifier.IDF:
                problems.append(f"the {SPARSE_VECTOR!r} sparse vector is not IDF-weighted (D73)")
        else:
            dense = None if isinstance(vectors, dict) else vectors
            if dense is None:
                problems.append("no unnamed dense vector")
        size = getattr(dense, "size", None)
        if dense is not None and not isinstance(size, int):
            problems.append("its dense vector size is unreadable")
        if problems:
            raise CollectionSchemaMismatch(self._collection, problems)
        # No problems means a dense vector with an integer size was found above.
        if isinstance(size, int) and size != self._dimensions:
            raise VectorDimensionMismatch(self._collection, size, self._dimensions)

    async def existing_count(self) -> int | None:
        """The collection's exact point count, or None if it does not exist. Read-only.

        For a caller that must not create or alter a collection, which
        `ensure_ready` does: it creates a missing collection and re-asserts the
        `paper_id` payload index on every call. This makes three reads and no
        write: `GET .../exists`, then `GET /collections/{name}` for a strict
        schema check (`_assert_schema`: every vector this adapter reads is
        present, including the hybrid collection's lexical vector, at the right
        size; a collection built otherwise is refused, not read), then
        `POST .../points/count`. Existence is asked first rather than inferred
        from `count()` failing, so what the server does when counting a missing
        collection never matters (D168).
        """
        if not await self._client.collection_exists(self._collection):
            return None
        self._assert_schema(await self._client.get_collection(self._collection))
        return await self.count()

    async def require_existing(self) -> None:
        """Validate a collection this process must not create. Read-only (D172).

        What the API and worker call for the hybrid collection, in place of
        `ensure_ready`: only `scholiast-reindex --hybrid` creates it, filled, so a
        process that found it missing and created it would serve or extend an
        empty index. Missing is refused, and so is any shape `_assert_schema`
        refuses.
        """
        if not await self._client.collection_exists(self._collection):
            raise CollectionNotFound(self._collection)
        self._assert_schema(await self._client.get_collection(self._collection))

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        points = [
            models.PointStruct(
                id=point_id_for(record.chunk_id),
                vector=self._point_vector(record),
                payload=self._payload(record),
            )
            for record in records
        ]
        await self._client.upsert(self._collection, points=points, wait=True)
        return len(points)

    def _point_vector(self, record: VectorRecord) -> models.VectorStruct:
        if not self._sparse:
            # A dense-only collection has nowhere to put lexical weights, so they
            # are ignored and the point is exactly what it was before D172.
            return list(record.vector)
        vector: dict[str, models.Vector] = {DENSE_VECTOR: list(record.vector)}
        if record.lexical:
            # In the same point as the dense vector, so no point is ever half-written.
            # An empty weight set (a chunk with no tokens) gets no sparse vector,
            # which the store would reject; its version is still stamped below.
            vector[SPARSE_VECTOR] = models.SparseVector(
                indices=list(record.lexical), values=list(record.lexical.values())
            )
        return vector

    def _payload(self, record: VectorRecord) -> dict[str, str]:
        payload = {
            PAYLOAD_CHUNK_ID: record.chunk_id,
            PAYLOAD_PAPER_ID: record.paper_id,
            PAYLOAD_SECTION_KIND: record.section_kind.value,
            PAYLOAD_CHUNKER_VERSION: record.chunker_version,
            PAYLOAD_EMBEDDER_VERSION: record.embedder_version,
        }
        if self._sparse and record.lexical_version is not None:
            # What the completeness check counts (D172): a point whose weights came
            # from an earlier version is not a current one.
            payload[PAYLOAD_LEXICAL_VERSION] = record.lexical_version
        return payload

    async def lexical_chunk_ids(self, *, lexical_version: str) -> set[str]:
        """Chunk ids of every point stamped with this lexical version. Read-only."""
        self._require_sparse()
        found: set[str] = set()
        offset: models.ExtendedPointId | None = None
        while True:
            points, offset = await self._client.scroll(
                self._collection,
                scroll_filter=self._lexical_filter(lexical_version),
                limit=1000,
                offset=offset,
                with_payload=[PAYLOAD_CHUNK_ID],
                with_vectors=False,
            )
            found.update(str((p.payload or {}).get(PAYLOAD_CHUNK_ID, "")) for p in points)
            if offset is None:
                return found

    async def count_lexical(self, *, lexical_version: str) -> int:
        """Points stamped with this lexical version: the readiness probe's count."""
        self._require_sparse()
        result = await self._client.count(
            self._collection, count_filter=self._lexical_filter(lexical_version), exact=True
        )
        return int(result.count)

    @staticmethod
    def _lexical_filter(lexical_version: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=PAYLOAD_LEXICAL_VERSION, match=models.MatchValue(value=lexical_version)
                )
            ]
        )

    async def search(
        self,
        vector: Sequence[float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[ScoredChunk]:
        if not paper_ids:
            # The one input that would turn a scoped query into a global one.
            raise EmptyRetrievalScope

        response = await self._client.query_points(
            self._collection,
            query=list(vector),
            using=DENSE_VECTOR if self._sparse else None,
            limit=limit,
            query_filter=self._paper_filter(paper_ids),
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=str((point.payload or {}).get(PAYLOAD_CHUNK_ID, "")),
                paper_id=str((point.payload or {}).get(PAYLOAD_PAPER_ID, "")),
                score=float(point.score),
                arm=RetrievalArm.DENSE,
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    async def upsert_sparse(self, records: Sequence[SparseRecord]) -> int:
        """Attach lexical weights to points that already carry dense vectors.

        `set_vectors` rather than `upsert`, deliberately: the sparse pass runs
        after the dense one over the same chunk ids (D46 makes them
        deterministic), and a full upsert would rewrite the payload and the dense
        vector from a record that carries neither — silently emptying the
        collection of everything except term weights.
        """
        self._require_sparse()
        if not records:
            return 0
        await self._client.update_vectors(
            self._collection,
            points=[
                models.PointVectors(
                    id=point_id_for(r.chunk_id),
                    vector={
                        SPARSE_VECTOR: models.SparseVector(
                            indices=list(r.weights), values=list(r.weights.values())
                        )
                    },
                )
                for r in records
            ],
            wait=True,
        )
        return len(records)

    async def search_lexical(
        self,
        weights: dict[int, float],
        *,
        paper_ids: Sequence[str],
        limit: int,
    ) -> list[ScoredChunk]:
        """BM25 matches. The store multiplies in IDF; these weights are TF only."""
        self._require_sparse()
        if not paper_ids:
            # Same rule as the dense arm, for the same reason (§5.8 layer 3).
            raise EmptyRetrievalScope
        if not weights:
            # A query whose every term was tokenised away matches nothing. An
            # empty sparse vector is rejected by the store, and returning [] here
            # says the same thing without a round trip — RRF then simply fuses one
            # arm, which is the correct degenerate behaviour.
            return []

        response = await self._client.query_points(
            self._collection,
            query=models.SparseVector(indices=list(weights), values=list(weights.values())),
            using=SPARSE_VECTOR,
            limit=limit,
            query_filter=self._paper_filter(paper_ids),
            with_payload=True,
        )
        return [
            ScoredChunk(
                chunk_id=str((point.payload or {}).get(PAYLOAD_CHUNK_ID, "")),
                paper_id=str((point.payload or {}).get(PAYLOAD_PAPER_ID, "")),
                score=float(point.score),
                arm=RetrievalArm.LEXICAL,
                rank=rank,
            )
            for rank, point in enumerate(response.points, start=1)
        ]

    def _require_sparse(self) -> None:
        if not self._sparse:
            raise RuntimeError(
                "this adapter was built dense-only; construct it with sparse=True "
                "to use the lexical arm. A dense-only collection has no lexical "
                "vector: production's hybrid arm reads the hybrid collection (D172)"
            )

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        """Remove a paper's vectors (FR-W2, and step 2 of §5.6.4's sweep)."""
        if not paper_ids:
            return 0
        before = await self.count(paper_ids=paper_ids)
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=self._paper_filter(paper_ids)),
            wait=True,
        )
        return before

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        result = await self._client.count(
            self._collection,
            count_filter=self._paper_filter(paper_ids) if paper_ids else None,
            exact=True,
        )
        return int(result.count)

    async def ping(self) -> bool:
        """Reachability, for FR-O2. A metadata call, not a count."""
        try:
            await self._client.get_collections()
        except Exception:
            # Deliberately broad: connection refused, DNS, auth and a protocol
            # mismatch all mean the same thing to a readiness probe, and a probe
            # that raised would report "the API is broken" for "Qdrant is down".
            return False
        return True

    async def count_orphans(self, live_paper_ids: Sequence[str]) -> int:
        """Points whose paper is not in `live_paper_ids` (§5.6.4 step 4)."""
        result = await self._client.count(
            self._collection,
            count_filter=self._orphan_filter(live_paper_ids),
            exact=True,
        )
        return int(result.count)

    async def delete_orphans(self, live_paper_ids: Sequence[str]) -> int:
        """Remove every point whose paper is gone from Postgres (D102).

        Counted first so the sweep can report residue it did not create: a
        number above zero here means something upstream failed to clean up, and
        a sweeper that silently fixes that is indistinguishable from one with
        nothing to fix.
        """
        before = await self.count_orphans(live_paper_ids)
        if not before:
            return 0
        await self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=self._orphan_filter(live_paper_ids)),
            wait=True,
        )
        return before

    @staticmethod
    def _orphan_filter(live_paper_ids: Sequence[str]) -> models.Filter:
        """Everything whose `paper_id` is NOT one of the live ones.

        **The empty case is the dangerous one and is handled by construction
        rather than by a branch.** An empty live set means Postgres holds no
        papers at all, so every point is an orphan — the exact opposite of the
        "nothing to do" an empty-list guard would produce. Substituting a
        sentinel that cannot be a paper id keeps one code path: `must_not
        MatchAny([sentinel])` excludes nothing and therefore matches the whole
        collection. Paper ids are UUIDs, so the sentinel can never collide with
        one, and it never reaches the index as data.
        """
        return models.Filter(
            must_not=[
                models.FieldCondition(
                    key=PAYLOAD_PAPER_ID,
                    match=models.MatchAny(any=list(live_paper_ids) or [_NO_SUCH_PAPER]),
                )
            ]
        )

    @staticmethod
    def _paper_filter(paper_ids: Sequence[str]) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=PAYLOAD_PAPER_ID,
                    match=models.MatchAny(any=list(paper_ids)),
                )
            ]
        )
