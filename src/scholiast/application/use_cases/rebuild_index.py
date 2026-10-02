"""Rebuild the vector index from Postgres — NFR-R6, D101.

> "Relational store is source of truth; vector store is a derived index and must
> be rebuildable from scratch by a documented command."

This is that command's body. It exists to make the claim **falsifiable**: if
chunk rows plus the embedder cannot reproduce the index, then the index holds
something Postgres does not, and NFR-R6 is false. Running this is the only way
to find that out.

**It re-embeds rather than copying.** There is nowhere to copy from — the
vectors live only in Qdrant, which is the store being rebuilt. That makes the
command an honest reconstruction instead of a backup restore, and it is why the
cost is a full embedding pass: at M0's measured 196 chunks/s (384-dim), D-RET's
7,854 chunks is ~40 s. Linear in corpus size, not constant, and stated rather
than discovered.

**The ML-3 contextual header has to be reproduced, not skipped.** Ingestion
embeds `chunk.embedding_input(title, year)`, not `chunk.text`: ~20 tokens of
paper context prepended so a Related Work chunk reads as describing other work.
Rebuilding from raw chunk text would produce a *different* index that looks
correct — a silent re-baseline of every measurement in M2. The title is safe to
read back because ingestion persists GROBID's title onto the paper row before
chunking (`_parsed_metadata`), so the row now holds the same string the original
embed used.

**Convergent, not destructive.** Chunk ids are deterministic (D51), so an upsert
over an existing collection replaces rather than duplicates, and orphans are
removed at the end. "From scratch" is therefore the same command run against an
empty collection — one code path for the disaster case and the drift case,
which is the only way the disaster case is ever exercised.

**The embedder must be ungated (D74, D101).** A gated embedder re-batches and
perturbs fp16 vectors, so rebuilding through one yields a collection that
differs from the one it replaced — and a rebuild that changes the index is not
a rebuild. The caller owns construction, so this module cannot enforce it; the
CLI that wires it does, and says so.
"""

from __future__ import annotations

from dataclasses import dataclass

from scholiast.application.ports.ml import Embedder, TextRole
from scholiast.application.ports.repositories import ChunkRepository, PaperRepository
from scholiast.application.ports.vector import VectorIndex, VectorRecord
from scholiast.domain.chunking import embeddable_chunks
from scholiast.domain.errors import InvariantViolation

#: Bounded for §5.7's reason, not for throughput: small batches let a
#: high-priority rerank preempt bulk work. A rebuild is bulk work by definition,
#: so it takes the same discipline as ingestion rather than a bigger batch.
DEFAULT_REBUILD_BATCH = 32


@dataclass(frozen=True, slots=True)
class RebuildResult:
    """What the rebuild reconstructed, and what it removed."""

    papers: int = 0
    chunks_embedded: int = 0
    vectors_upserted: int = 0
    #: Points that survived the rebuild without a chunk row behind them. Any
    #: number above zero means the index had drifted — the same residue the
    #: reconciliation sweep exists to catch, found here by a different route.
    orphans_deleted: int = 0


class RebuildIndexUseCase:
    """Reconstruct every vector from the chunk rows that produced it."""

    def __init__(
        self,
        *,
        papers: PaperRepository,
        chunks: ChunkRepository,
        embedder: Embedder,
        index: VectorIndex,
        batch_size: int = DEFAULT_REBUILD_BATCH,
    ) -> None:
        self._papers = papers
        self._chunks = chunks
        self._embedder = embedder
        self._index = index
        self._batch_size = batch_size

    async def execute(self) -> RebuildResult:
        paper_ids = await self._papers.all_paper_ids()

        papers = embedded = upserted = 0
        for paper_id in paper_ids:
            written = await self._rebuild_paper(paper_id)
            if written is None:
                continue
            papers += 1
            embedded += written
            upserted += written

        # Last: anything still in the index that no chunk row produced. After a
        # full re-embed the live set is exactly what was just written, so this
        # is the moment the difference is meaningful.
        orphans = await self._index.delete_orphans(paper_ids)
        return RebuildResult(
            papers=papers,
            chunks_embedded=embedded,
            vectors_upserted=upserted,
            orphans_deleted=orphans,
        )

    async def _rebuild_paper(self, paper_id: str) -> int | None:
        """Re-embed and re-index one paper. None when there is nothing to do."""
        paper = await self._papers.get(paper_id)
        if paper is None:
            # Deleted between the id listing and here. Not an error: the next
            # step removes its vectors as orphans anyway.
            return None

        # The SAME filter ingestion applies, from the same function. Rebuilding
        # every chunk row would index the parent windows too — 111 vectors where
        # ingestion wrote 67 on this corpus — and a parent would become a
        # matchable hit competing with the child inside it. Nothing would fail;
        # every retrieval number would just move.
        chunks = embeddable_chunks(await self._chunks.list_for_paper(paper_id))
        if not chunks:
            # Unparsed, failed, or still in flight — or a paper whose chunks are
            # all parents. A paper with nothing embeddable contributes no
            # vectors, which is the correct rebuilt state.
            return None

        texts = [chunk.embedding_input(paper.title, paper.year) for chunk in chunks]
        vectors = await self._embedder.embed(
            texts, role=TextRole.PASSAGE, batch_size=self._batch_size
        )
        if len(vectors) != len(chunks):
            raise InvariantViolation(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        return await self._index.upsert(
            [
                VectorRecord(
                    chunk_id=chunk.id,
                    paper_id=paper_id,
                    vector=tuple(vector),
                    section_kind=chunk.provenance.section_kind,
                    chunker_version=chunk.chunker_version,
                    embedder_version=self._embedder.version,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]
        )
