"""Every write to two collections, every read from one (D172).

While production moves from the dense-only `scholiast_chunks` to the hybrid
collection, both have to stay complete: the dense one because a rollback reads it,
the hybrid one because a hybrid read is refused unless it holds every embedded
chunk. So every process that writes - ingestion, the in-process and Celery
workers, deletion, the reconciliation sweep - writes through this when the hybrid
collection is configured, and nothing has to remember to write twice.

**A failure on either side fails the write.** The caller is a job that retries
on its lease, upserts converge (D46), and chunk rows are stamped only after the
write returns (D53), so a half-done write is redone rather than recorded as done.
Swallowing the mirror's failure instead would leave a hybrid collection that
quietly lacks points while every row says they exist.

**A delete goes to the mirror first, a write to the primary first**, so a partial
failure always leaves its gap where a check can see it. A half-done upsert leaves
points whose rows are unstamped, which no read reaches; a half-done delete leaves
the hybrid collection short, which the completeness check reports, and the dense
collection that production reads intact.

**It never creates the mirror.** `ensure_ready` creates the primary as it always
has, and only validates the mirror (`require_existing`): a mirror created here
would be empty, and only `scholiast-reindex --hybrid` creates one, filled.
"""

from __future__ import annotations

from collections.abc import Sequence

from scholiast.application.ports.vector import VectorRecord
from scholiast.domain.retrieval import ScoredChunk
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex


class MirroredVectorIndex:
    """A `VectorIndex` over a primary collection and a mirror of it."""

    def __init__(self, primary: QdrantVectorIndex, mirror: QdrantVectorIndex) -> None:
        if primary.collection == mirror.collection:
            # Writing every point twice to one collection would look like a mirror
            # and be none.
            raise ValueError(f"a collection cannot mirror itself: {primary.collection!r}")
        self._primary = primary
        self._mirror = mirror

    @property
    def primary(self) -> QdrantVectorIndex:
        return self._primary

    @property
    def mirror(self) -> QdrantVectorIndex:
        return self._mirror

    async def ensure_ready(self) -> None:
        await self._primary.ensure_ready()
        await self._mirror.require_existing()

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        written = await self._primary.upsert(records)
        await self._mirror.upsert(records)
        return written

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        # Reads are the primary's: whoever reads the hybrid collection holds it
        # directly, behind the completeness check this wrapper does not make.
        return await self._primary.search(vector, paper_ids=paper_ids, limit=limit)

    async def delete_by_papers(self, paper_ids: Sequence[str]) -> int:
        # The MIRROR first. A delete runs before its paper row is deleted, so a
        # failure here rolls the row back and leaves the paper in place: whichever
        # side was already deleted now has a hole under a paper that still exists.
        # Mirror-first puts that hole in the hybrid collection, which the
        # completeness check, the sweep and hybrid readiness all detect, never in
        # the dense one, which production reads and nothing checks.
        await self._mirror.delete_by_papers(paper_ids)
        # The primary's count, as before: callers report vectors per paper, and
        # the mirror holds the same points.
        return await self._primary.delete_by_papers(paper_ids)

    async def count(self, *, paper_ids: Sequence[str] | None = None) -> int:
        return await self._primary.count(paper_ids=paper_ids)

    async def ping(self) -> bool:
        return await self._primary.ping() and await self._mirror.ping()

    async def count_orphans(self, live_paper_ids: Sequence[str]) -> int:
        # Summed: residue in either collection is residue, and the sweep reports it.
        return await self._primary.count_orphans(live_paper_ids) + await self._mirror.count_orphans(
            live_paper_ids
        )

    async def delete_orphans(self, live_paper_ids: Sequence[str]) -> int:
        removed = await self._primary.delete_orphans(live_paper_ids)
        return removed + await self._mirror.delete_orphans(live_paper_ids)

    async def close(self) -> None:
        await self._primary.close()
        await self._mirror.close()
