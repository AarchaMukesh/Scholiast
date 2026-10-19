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
    uv run scholiast-reindex --hybrid   # build or rebuild the hybrid collection (D172)
    uv run scholiast-reindex --verify   # is the hybrid collection complete? read-only

**`--hybrid` is the only thing that creates the hybrid collection** (D172). The
API and the workers validate it and refuse to start without it, so it never
exists empty: it is created here and filled from Postgres in the same run, dense
and lexical vectors in each point. Run against an existing one it converges, as
the plain rebuild does, which is how it catches up after the migration's gap.

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
from scholiast.application.ports.tenancy import (
    INDEX_CHECK_REASON,
    REBUILD_REASON,
    SystemContext,
)
from scholiast.application.use_cases.rebuild_index import RebuildIndexUseCase, RebuildResult
from scholiast.application.use_cases.verify_hybrid_index import (
    HybridCoverage,
    VerifyHybridIndexUseCase,
)
from scholiast.config.settings import Settings, get_settings
from scholiast.domain.chunking import embeddable_chunks
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.db.repositories import SqlChunkRepository, SqlPaperRepository
from scholiast.infrastructure.ml.embedder import LocalEmbedder
from scholiast.infrastructure.vector.factory import build_vector_indexes
from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scholiast-reindex",
        description="Rebuild the vector index from Postgres (NFR-R6).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="report how many papers and chunks would be re-embedded, and change nothing",
    )
    mode.add_argument(
        "--hybrid",
        action="store_true",
        help=(
            "build the hybrid collection named by qdrant_hybrid_collection: create it if "
            "absent and write dense and lexical vectors from Postgres (D172)"
        ),
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help=(
            "check, read-only, that the hybrid collection holds every embedded chunk at "
            "the current lexical version; exit 1 if it does not"
        ),
    )
    return parser


def _hybrid_index(settings: Settings) -> QdrantVectorIndex:
    """The hybrid collection's adapter, built as every other process builds it."""
    hybrid = build_vector_indexes(settings).hybrid
    if hybrid is None:
        raise SystemExit(
            "qdrant_hybrid_collection is not set: there is no hybrid collection to build "
            "or check (D172)"
        )
    return hybrid


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


async def _rebuild(settings: Settings, *, hybrid: bool = False) -> RebuildResult:
    # Exactly one collection per run: the dense one, or with --hybrid the hybrid
    # one, whose `ensure_ready` is the only place it is ever created (D172).
    index = _hybrid_index(settings) if hybrid else build_vector_indexes(settings).primary
    engine = build_engine(settings)
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


async def _verify(settings: Settings) -> HybridCoverage:
    """The id-level completeness check, read-only: what a hybrid read waits on."""
    index = _hybrid_index(settings)
    engine = build_engine(settings)
    try:
        await index.require_existing()
        async with build_session_factory(engine)() as session:
            return await VerifyHybridIndexUseCase(
                chunks=SqlChunkRepository(session, SystemContext(reason=INDEX_CHECK_REASON)),
                lexical=index,
            ).execute()
    finally:
        await index.close()
        await engine.dispose()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()

    if args.dry_run:
        print(asyncio.run(_dry_run(settings)))
        return 0

    if args.verify:
        coverage = asyncio.run(_verify(settings))
        print(
            f"{coverage.present} of {coverage.expected} embedded chunks have a point at "
            f"lexical version {coverage.lexical_version}; {len(coverage.missing)} missing, "
            f"{coverage.unaccounted} points no embedded chunk accounts for"
        )
        for chunk_id in coverage.missing[:20]:
            print(f"  missing {chunk_id}")
        return 0 if coverage.complete else 1

    result = asyncio.run(_rebuild(settings, hybrid=args.hybrid))
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
