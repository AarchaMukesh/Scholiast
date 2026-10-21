"""One construction of production's vector indexes, for every process (D172).

The API, the Celery worker and `scholiast-reindex` each used to build their own
`QdrantVectorIndex`. That was harmless with one dense collection. With a hybrid
collection beside it, three hand-written constructions would be three chances to
disagree - a worker that does not mirror while the API reads the mirror is
exactly how the hybrid collection would go stale - so all three call this.

- `write`: what every writer uses. The dense collection alone, as before, or that
  collection mirrored to the hybrid one when `qdrant_hybrid_collection` is set.
- `read` and `lexical`: what QA retrieves from. Dense mode reads the dense
  collection; hybrid mode reads both arms from the hybrid collection.
- `hybrid`: the hybrid collection's adapter, for the completeness check, or None.

`embedded_chunks` is what lets a starting API or worker create a missing hybrid
collection, for a corpus with nothing embedded yet (D174). The reindex command
does not pass it: it builds the collection itself, filled.

Nothing here touches the network: adapters connect on first use.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from scholiast.config.settings import Settings
from scholiast.domain.retrieval import RetrievalMode
from scholiast.infrastructure.vector.mirrored import MirroredVectorIndex
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex


@dataclass(frozen=True, slots=True)
class VectorIndexes:
    """Production's indexes, as one process should hold them."""

    mode: RetrievalMode
    primary: QdrantVectorIndex
    hybrid: QdrantVectorIndex | None
    write: QdrantVectorIndex | MirroredVectorIndex

    @property
    def read(self) -> QdrantVectorIndex:
        """The collection QA's dense arm searches."""
        return self.hybrid if self.mode is RetrievalMode.HYBRID and self.hybrid else self.primary

    @property
    def lexical(self) -> QdrantVectorIndex | None:
        """The collection QA's lexical arm searches; None in dense mode."""
        return self.hybrid if self.mode is RetrievalMode.HYBRID else None

    async def close(self) -> None:
        await self.write.close()


def build_vector_indexes(
    settings: Settings, *, embedded_chunks: Callable[[], Awaitable[int]] | None = None
) -> VectorIndexes:
    mode = RetrievalMode(settings.retrieval_mode)
    primary = QdrantVectorIndex(
        settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
    )
    if not settings.qdrant_hybrid_collection:
        # Settings refuses hybrid mode without a hybrid collection; asserted here
        # as well, because a read that silently fell back to dense would be a
        # rollback nobody asked for.
        if mode is RetrievalMode.HYBRID:
            raise ValueError("hybrid retrieval needs qdrant_hybrid_collection (D172)")
        return VectorIndexes(mode=mode, primary=primary, hybrid=None, write=primary)
    hybrid = QdrantVectorIndex(
        settings.qdrant_url,
        collection=settings.qdrant_hybrid_collection,
        dimensions=settings.embedding_dimensions,
        sparse=True,
    )
    return VectorIndexes(
        mode=mode,
        primary=primary,
        hybrid=hybrid,
        write=MirroredVectorIndex(primary, hybrid, embedded_chunks=embedded_chunks),
    )
