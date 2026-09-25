"""Rerank latency under concurrent ingestion (doc 8 deliverable 6, §5.7, D79).

Doc 9 §9.8 calls this "the single most design-specific number in this project" —
the empirical answer to *"you have one GPU and two consumers, how do you keep the
user-facing path fast?"*. Phase 4 measured only the ungated half (rerank p95
578-707 ms under contention, 3.2x), which establishes that contention hurts and
says nothing about whether the §5.7 scheme fixes it. **The comparison is the
deliverable**, so this module runs the same query workload under four conditions
that differ only in how the GPU is shared.

**The ingestion arm embeds and writes nothing (D79).** What is being measured is
GPU contention, and an upsert would add store I/O while mutating the collection
every published metric depends on — D74 measured that a gated embedder perturbs
fp16 vectors, so writing from here could silently move phases 2-5. The honest cost
is that this is not an end-to-end ingestion measurement: NFR-P6's 90 s/paper covers
parse, chunk, embed and write, and is not this number.

**Retrieval results must not move between conditions.** Ordering does not depend on
timing, so every condition should produce identical metrics; the report carries them
per condition precisely so a difference would be visible rather than assumed away.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import VectorIndex
from scholiast.eval.dret import DRet, Query
from scholiast.eval.harness import EmbedderLike, LexicalLike, RerankerLike, RunReport, run_hybrid

#: How many times the query set runs per condition. p95 over 53 queries moves too
#: much to compare conditions with; three passes give 159 samples, which is enough
#: for p50 and p95 and still leaves p99 indicative rather than solid (D79).
DEFAULT_REPEATS = 3


class GateLike(Protocol):
    """Only the counter this report needs; the gate itself is an adapter (D74)."""

    acquisitions: Mapping[Any, int]


@dataclass(frozen=True, slots=True)
class Condition:
    """One way of sharing the GPU between the two consumers."""

    label: str
    ingesting: bool
    gated: bool
    batch_size: int
    #: True for a cell that exists to explain a result rather than to propose a
    #: configuration — the batch-16 cell, which §5.7's "bounded batches" premise
    #: needs and which production does not run (D79).
    diagnostic: bool = False
    note: str = ""


class IngestionLoad:
    """A saturating embedding workload: §5.7's low-priority consumer.

    Cycles the corpus rather than stopping at the end of it, because a loop that
    quietly ran out of work would leave the query arm measuring an idle GPU and
    reporting it as contention.
    """

    def __init__(self, embedder: EmbedderLike, texts: Sequence[str], *, batch_size: int) -> None:
        if not texts:
            raise ValueError("an ingestion load with no texts would measure an idle GPU")
        self._embedder = embedder
        self._texts = texts
        self._batch_size = batch_size
        self._stopping = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self.batches = 0
        self.chunks = 0

    async def start(self) -> None:
        self._task = asyncio.create_task(self._pump())
        # Let the first batch reach the GPU before the query arm starts timing, so
        # the first measured query meets a busy device rather than an idle one.
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Stop the loop and re-raise anything it died of.

        A load task that failed silently is the worst outcome available here: the
        conditions would still produce numbers, and they would be idle numbers
        wearing a contention label.
        """
        self._stopping.set()
        if self._task is not None:
            await self._task
            self._task = None

    async def _pump(self) -> None:
        cursor = 0
        while not self._stopping.is_set():
            batch = [self._texts[(cursor + i) % len(self._texts)] for i in range(self._batch_size)]
            cursor = (cursor + self._batch_size) % len(self._texts)
            await self._embedder.embed(batch, role=TextRole.PASSAGE, batch_size=self._batch_size)
            self.batches += 1
            self.chunks += len(batch)
            # Yield explicitly. The real embedder suspends inside `to_thread`, but a
            # loop that relies on its collaborator suspending is one cheap batch
            # away from starving the arm it is supposed to compete with — and the
            # symptom would be a query workload that never runs rather than an error.
            await asyncio.sleep(0)


@dataclass(slots=True)
class ConditionReport:
    """What one condition produced. Latency, the cost side, and the proof it ran."""

    condition: Condition
    rerank_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    embed_ms: list[float] = field(default_factory=list)
    search_ms: list[float] = field(default_factory=list)
    queries: int = 0
    seconds: float = 0.0
    #: Ingestion work completed WHILE the queries were being measured — the
    #: evidence that a "contended" condition was actually contended.
    batches_during: int = 0
    chunks_during: int = 0
    #: Gate acquisitions by priority name, empty when ungated. A gated condition
    #: whose QUERY count is not the number of GPU calls the query arm made would
    #: mean the gate was bypassed somewhere.
    acquisitions: dict[str, int] = field(default_factory=dict)
    #: Retrieval quality, which must be identical across conditions.
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def ingestion_chunks_per_second(self) -> float | None:
        if not self.condition.ingesting or self.seconds <= 0:
            return None
        return round(self.chunks_during / self.seconds, 1)

    def percentiles(self, values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        return {
            name: round(ordered[min(len(ordered) - 1, int(len(ordered) * q))], 2)
            for name, q in (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))
        }


async def measure_condition(
    condition: Condition,
    dataset: DRet,
    queries: Sequence[Query],
    index: VectorIndex,
    lexical: LexicalLike,
    embedder: EmbedderLike,
    reranker: RerankerLike,
    *,
    paper_ids: Sequence[str],
    load_texts: Sequence[str],
    repeats: int = DEFAULT_REPEATS,
    depth: int = 50,
    gate: GateLike | None = None,
) -> ConditionReport:
    """Run the query workload under one sharing regime.

    The query arm is the real retrieval stage — `run_hybrid`'s `hybrid_rerank`,
    with its own warm-ups — rather than a bare rerank call, because NFR-P2's budget
    is the whole stage and the query embed contends for the same device.
    """
    report = ConditionReport(condition=condition)
    # Load both models BEFORE anything is timed. Between them they are ~1.5 GB, and
    # whichever condition touches them first would otherwise spend that inside its
    # own window — harmless for the per-query percentiles, which have their own
    # warm-ups, and badly misleading for INGESTION THROUGHPUT, which divides work
    # by wall-clock. The first measured run deflated the gated arm's throughput by
    # about a fifth that way, which is the cost side the gate is judged on.
    await embedder.embed(["warm"], role=TextRole.PASSAGE, batch_size=condition.batch_size)
    await reranker.rerank("warm", ["warm"])
    taken = dict(gate.acquisitions) if gate is not None else {}

    load = (
        IngestionLoad(embedder, load_texts, batch_size=condition.batch_size)
        if condition.ingesting
        else None
    )
    if load is not None:
        await load.start()
    started = time.perf_counter()
    before = load.batches if load else 0
    chunks_before = load.chunks if load else 0
    try:
        runs: list[RunReport] = []
        for _ in range(repeats):
            runs.append(
                await run_hybrid(
                    dataset,
                    queries,
                    index,
                    lexical,
                    embedder,
                    label=condition.label,
                    paper_ids=paper_ids,
                    arms="hybrid_rerank",
                    depth=depth,
                    reranker=reranker,
                )
            )
    finally:
        report.seconds = time.perf_counter() - started
        if load is not None:
            report.batches_during = load.batches - before
            report.chunks_during = load.chunks - chunks_before
            await load.stop()

    for run in runs:
        report.rerank_ms.extend(run.rerank_ms)
        report.total_ms.extend(run.latencies_ms)
        report.embed_ms.extend(run.embed_ms)
        report.search_ms.extend(run.search_ms)
        report.queries += len(run.latencies_ms)
    # Read off the last pass: every pass ranks the same queries against the same
    # collection, so these are a constant that a broken condition would change.
    report.metrics = {name: round(value, 6) for name, value in runs[-1].metrics.means().items()}
    if gate is not None:
        # Deltas, not totals: the gated conditions share one gate, so totals would
        # carry the previous condition's acquisitions into this block. Taken after
        # the warm-up, so the count is exactly the measured work.
        report.acquisitions = {
            priority.name: count - taken.get(priority, 0)
            for priority, count in gate.acquisitions.items()
        }
    verify_contended(report)
    return report


def verify_contended(report: ConditionReport) -> None:
    """A contended condition that did no ingestion measured an idle GPU.

    Raised rather than warned: the numbers are plausible either way, and a report
    claiming a gate helped when neither condition was contended would be worse
    than no measurement at all.
    """
    if report.condition.ingesting and report.batches_during <= 0:
        raise ValueError(
            f"condition {report.condition.label!r} claims concurrent ingestion but "
            "completed no embedding batches while the queries ran — the load never "
            "reached the GPU, so these would be idle numbers under a contention label"
        )
    if not report.condition.ingesting and report.batches_during:
        raise ValueError(
            f"condition {report.condition.label!r} is the idle reference but "
            f"{report.batches_during} embedding batches ran during it"
        )


def verify_same_retrieval(reports: Sequence[ConditionReport]) -> None:
    """Every condition must retrieve identically — only timing may differ.

    Ranking does not depend on how the GPU was shared, so a difference here means
    the harness changed something it had no business changing (the gate touching
    query vectors, a different collection, a different depth). Checking is free and
    the failure would otherwise be invisible in a latency table.
    """
    if not reports:
        return
    reference = reports[0]
    for other in reports[1:]:
        if other.metrics != reference.metrics:
            raise ValueError(
                f"condition {other.condition.label!r} retrieved differently from "
                f"{reference.condition.label!r}: {other.metrics} vs "
                f"{reference.metrics} — only timing may vary between conditions"
            )
