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


def point_id_for(chunk_id: str) -> str:
    """Deterministic point id for a chunk (D46)."""
    return str(uuid.uuid5(POINT_NAMESPACE, chunk_id))


class QdrantVectorIndex:
    """The `VectorIndex` adapter."""

    def __init__(
        self,
        url: str,
        *,
        collection: str = "scholiast_chunks",
        dimensions: int = 768,
        timeout: float = 30.0,
    ) -> None:
        self._url = url
        self._timeout = int(timeout)
        self._collection = collection
        self._dimensions = dimensions
        self._connection: AsyncQdrantClient | None = None

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

    @property
    def collection(self) -> str:
        return self._collection

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()
            self._connection = None

    async def ensure_ready(self) -> None:
        if await self._client.collection_exists(self._collection):
            self._assert_dimensions(await self._client.get_collection(self._collection))
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
        existing = getattr(params, "size", None)
        if isinstance(existing, int) and existing != self._dimensions:
            raise VectorDimensionMismatch(self._collection, existing, self._dimensions)

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        if not records:
            return 0
        points = [
            models.PointStruct(
                id=point_id_for(record.chunk_id),
                vector=list(record.vector),
                payload={
                    PAYLOAD_CHUNK_ID: record.chunk_id,
                    PAYLOAD_PAPER_ID: record.paper_id,
                    PAYLOAD_SECTION_KIND: record.section_kind.value,
                    PAYLOAD_CHUNKER_VERSION: record.chunker_version,
                    PAYLOAD_EMBEDDER_VERSION: record.embedder_version,
                },
            )
            for record in records
        ]
        await self._client.upsert(self._collection, points=points, wait=True)
        return len(points)

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
