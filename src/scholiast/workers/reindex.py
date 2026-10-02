"""`scholiast-reindex` — NFR-R6's documented command (D101).

> "Relational store is source of truth; vector store is a derived index and must
> be rebuildable from scratch by a documented command."

This is that command. It re-embeds every chunk row in Postgres and upserts the
result, then removes anything left in the index that no chunk row produced.

**Run it against an empty collection and it is a disaster recovery; run it
against a populated one and it is a drift repair.** One code path for both, on
purpose: the disaster case is the one nobody exercises, so it had better be the
same code the ordinary case runs.

    uv run scholiast-reindex            # rebuild the configured collection
    uv run scholiast-reindex --dry-run  # report the work without doing it

**The embedder is built ungated, and that is not a default — it is a
requirement** (D74, D101). A gated embedder re-batches, which perturbs fp16
vectors; rebuilding through one produces a collection that differs from the one
it replaced, and a rebuild that changes the index is not a rebuild. The same
trap is why `eval --contention` refuses to build a gated embedder against the
eval collection.

**This is worker-side batch work, not a layer of its own.** It lives under
`workers` because that is where the layers contract already permits wiring
adapters to a use case, and because it is the same composition the maintenance
tasks do — a separate top-level module would be a second composition root with
nothing else to justify it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from scholiast.application.ports.ml import Precision
from scholiast.application.ports.tenancy import REBUILD_REASON, SystemContext
from scholiast.application.use_cases.rebuild_index import RebuildIndexUseCase, RebuildResult
from scholiast.config.settings import Settings, get_settings
from scholiast.domain.chunking import embeddable_chunks
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.db.repositories import SqlChunkRepository, SqlPaperRepository
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scholiast-reindex",
        description="Rebuild the vector index from Postgres (NFR-R6).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many papers and chunks would be re-embedded, and change nothing",
    )
    return parser


async def _dry_run(settings: Settings) -> str:
    """Count the work without doing it — the cheap half of the same traversal."""
    engine = build_engine(settings)
    scope = SystemContext(reason=REBUILD_REASON)
    try:
        async with build_session_factory(engine)() as session:
            papers = SqlPaperRepository(session, scope)
            chunks = SqlChunkRepository(session, scope)
            paper_ids = await papers.all_paper_ids()
            total = 0
            for paper_id in paper_ids:
                # The same filter the rebuild applies, or the estimate is of
                # work that will not happen: parent windows are never indexed.
                total += len(embeddable_chunks(await chunks.list_for_paper(paper_id)))
    finally:
        await engine.dispose()
    return f"would re-embed {total} chunks across {len(paper_ids)} papers"


async def _rebuild(settings: Settings) -> RebuildResult:
    engine = build_engine(settings)
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=settings.qdrant_collection,
        dimensions=settings.embedding_dimensions,
    )
    scope = SystemContext(reason=REBUILD_REASON)
    try:
        await index.ensure_ready()
        async with build_session_factory(engine)() as session:
            use_case = RebuildIndexUseCase(
                papers=SqlPaperRepository(session, scope),
                chunks=SqlChunkRepository(session, scope),
                embedder=LocalEmbedder(
                    settings.embedding_model,
                    device=settings.embedding_device,
                    precision=Precision(settings.embedding_precision),
                    batch_size=settings.embedding_batch_size,
                    # Ungated, and this is the requirement rather than the
                    # default: a gate re-batches and perturbs fp16 vectors
                    # (D74), so a gated rebuild would not reproduce the index
                    # it replaced.
                    gate=None,
                ),
                index=index,
                batch_size=settings.embedding_batch_size,
            )
            result = await use_case.execute()
    finally:
        await index.close()
        await engine.dispose()
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.dry_run:
        print(asyncio.run(_dry_run(settings)))
        return 0

    result = asyncio.run(_rebuild(settings))
    print(
        f"rebuilt {result.vectors_upserted} vectors "
        f"from {result.chunks_embedded} chunks across {result.papers} papers"
    )
    if result.orphans_deleted:
        # Not a success message. The index held points no chunk row produced,
        # which means something upstream leaked — the rebuild removed them, and
        # saying so is the only way anyone finds out it happened.
        print(f"removed {result.orphans_deleted} orphaned vectors the index should not have held")
    return 0


if __name__ == "__main__":
    sys.exit(main())
