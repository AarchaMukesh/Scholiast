"""`uv run eval` — the one command that reproduces the report (doc 9 §9.12, FR-E4).

§9.12's requirement is literal: *"One command: `make eval` (or `uv run eval`)
reproduces the published report."* This is that command.

**What it emits, and why both.** A JSON artifact and a terminal table. The JSON is
the archive — §9.13 keeps every milestone's run date-stamped so the trend line
across milestones becomes an artifact in itself, and that needs a machine-readable
file rather than scrollback. The table is what a human reads while working.

**Every run records what produced it.** Dataset versions, embedder and reranker
versions, collection name, retrieval scope, query counts and the grade scale go into
the report next to the numbers. §9.12 calls this a "full config snapshot"; the reason it
sits *inside* the report rather than beside it is that a number separated from the
configuration that produced it is not reproducible, only repeatable.

**Reports CORE and FULL in separate blocks** (D69). CORE is primary and carries the
statistical claims; FULL is the larger-sample robustness check. They are never
merged into one table, and FULL's nDCG is omitted — binary judgments do not produce
the graded nDCG CORE reports, and one shared column would overstate precision.

**The plain command measures what production runs** — the hybrid RRF arm since
D174, archived as `retrieval-hybrid`. `--dense` is the D56 dense baseline, still
archived as `retrieval`, so neither date-stamped series mixes in the other's arm.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scholiast.config.settings import get_settings
from scholiast.domain.lexical import LEXICAL_VERSION
from scholiast.eval import answerability
from scholiast.eval.abstention import (
    ABSTENTION_VERSION,
    REPRODUCE_SAMPLE,
    WITHHELD_SIZES,
    analyse,
    instances,
    largest_differences,
    progress_writer,
    read_progress,
    reproduce_sample,
    score_all,
    score_one,
)
from scholiast.eval.abstention import RULE as ABSTENTION_RULE
from scholiast.eval.answerability import (
    build_sets,
    candidate_sheet,
    load_sets,
    read_verdicts,
    sheet_json,
    sheet_markdown,
    write_sets,
)
from scholiast.eval.contention import (
    DEFAULT_REPEATS,
    Condition,
    ConditionReport,
    measure_condition,
    verify_same_retrieval,
)
from scholiast.eval.dedup import DedupReport, run_dedup
from scholiast.eval.dedup import render as _render_dedup
from scholiast.eval.dret import DRet, Query, build_dret, verify_core, verify_split
from scholiast.eval.fusion import FusionFit, fit_dense_weight, training_queries
from scholiast.eval.gate import (
    DECIDING_SIZE,
    DEPTH,
    DRAWS,
    GATE_KEY,
    RULE,
    SIZES,
    ScopedRuns,
    compare_with_archive,
    decide,
    drawn,
    measure_scoped,
    whole_corpus,
)
from scholiast.eval.harness import (
    EmbedderLike,
    IndexReport,
    RunReport,
    build_index,
    build_lexical_index,
    collect_arm_hits,
    collection_name,
    run_hybrid,
    run_retrieval,
)
from scholiast.eval.metrics import (
    NDCG_CUTOFF,
    bootstrap_interval,
    paired_bootstrap_delta,
)
from scholiast.eval.pdfcheck import (
    SAMPLE_SEED,
    SAMPLE_SIZE,
    comparable_queries,
    diagnose_mapping,
    fetch_and_parse,
    normalise_for_pdf_match,
    paper_from_document,
    queries_by_id,
    sample_papers,
)
from scholiast.eval.ranking import (
    DENSE_SCORES,
    Latency,
    RankingReport,
    compute_dense_scores,
    load_searches,
    run_ranking,
)
from scholiast.eval.ranking import render as _render_ranking
from scholiast.eval.smoke import (
    BASELINE_PATH,
    SMOKE_PAPERS,
    SMOKE_SEED,
    TOLERANCE,
    compare,
    load_baseline,
    save_baseline,
)
from scholiast.eval.snapshot import load_snapshot

if TYPE_CHECKING:
    # For the annotation only: at runtime the adapter is imported inside
    # `_build_dependencies`, so `--help` and the pure-logic tests stay light.
    from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

#: Where a run lands. Date-stamped rather than overwritten (§9.13): the trend line
#: across milestones is itself a deliverable, and it cannot exist if each run
#: destroys the last.
DEFAULT_REPORT_DIR = Path("var/eval")
#: Where `--withheld-sheet` writes D177's hand-check sheet: gitignored, like the
#: reports, because it is generated from the vendored snapshot.
DEFAULT_SHEET_DIR = Path("var/labelling")

#: Where D68's fetched PDFs live. Under `var/` so the repository never carries
#: third-party papers, and cached so a re-run costs arXiv nothing.
PDF_CACHE_DIR = Path("var/pdf")

#: The two builds the D68 check compares. Named once so the block labels, the
#: control pointer and the renderer cannot disagree about them.
JSON_SIDE = "json (published build)"
PDF_SIDE = "pdf (GROBID)"


@dataclass(frozen=True, slots=True)
class EmbedderSpec:
    """One encoder in the ML-4 comparison."""

    label: str
    model: str
    dimensions: int
    #: True for the encoder M1 actually shipped. Every delta is measured against
    #: it, and it is named in the report so no reader has to infer which arm is
    #: the incumbent.
    baseline: bool = False


#: ML-4: *"base-size retrieval encoder, chosen by benchmark on own eval set"* —
#: the requirement is that the choice follows a measurement, not that any
#: particular model wins. REV-7 names the live alternative and doc 9 §9.3 lists
#: it as a genuine ablation axis rather than a formality.
#:
#: Both arms run in ONE invocation against ONE `DRet` instance. That is what makes
#: the corpus, chunking, queries, judgments, scope, cutoffs, bootstrap seed and
#: methodology controlled by construction rather than by remembering to pass the
#: same flags twice.
COMPARED_EMBEDDERS: tuple[EmbedderSpec, ...] = (
    EmbedderSpec("bge-base (M1 baseline)", "BAAI/bge-base-en-v1.5", 768, baseline=True),
    EmbedderSpec("bge-small", "BAAI/bge-small-en-v1.5", 384),
)


#: REV-4's candidate depths, plus D77's diagnostic. 100 is NOT a production
#: candidate — doc 6 §6.6 fixes the budget at top-50 and rerank latency scales with
#: depth — it is there to say whether phase 4's null was a weak reranker or a
#: starved one, by showing where the pre-rerank ceiling stops climbing.
DEPTH_SWEEP: tuple[int, ...] = (10, 20, 30, 50, 100)
DIAGNOSTIC_DEPTH = 100
#: The depth every other depth is compared against: doc 6 §6.6's adopted budget and
#: what phases 3 and 4 measured, so the sweep's deltas are read against the
#: incumbent rather than against its own best cell.
INCUMBENT_DEPTH = 50


@dataclass(frozen=True, slots=True)
class Cell:
    """One cell of the ablation table: an encoder plus the knobs a phase varies.

    Replaces the `(spec, arm)` pairs earlier phases planned with. Depth (D77) and
    the fusion rule (D76) are configuration in exactly the same sense as the arm,
    and threading them as extra tuple positions is how a plan entry becomes
    unreadable and a block ends up labelled with the wrong one.
    """

    label: str
    spec: EmbedderSpec
    arm: str = "dense"
    depth: int = INCUMBENT_DEPTH
    dense_weight: float | None = None
    #: The cell this one's deltas are measured against, empty when it is itself a
    #: control. Per cell rather than per run, because the depth sweep compares each
    #: depth with the SAME ARM at depth 50 — one run-wide control would put a
    #: pre-rerank cell against a reranked one.
    control_label: str = ""
    is_m1_baseline: bool = False

    @property
    def fusion(self) -> str:
        return "rrf" if self.dense_weight is None else f"weighted(dense={self.dense_weight})"


@dataclass(slots=True)
class EvalReport:
    """One complete run, as archived."""

    generated_at: str
    config: dict[str, Any]
    blocks: dict[str, dict[str, Any]]

    def to_json(self) -> str:
        return json.dumps(
            {"generated_at": self.generated_at, "config": self.config, "blocks": self.blocks},
            indent=2,
            sort_keys=True,
        )


def _summarise(run: RunReport, *, graded: bool) -> dict[str, Any]:
    """Point estimates with intervals, never one without the other (§9.4).

    `graded` gates nDCG out of the FULL block. The gate is here, in the code that
    builds the report, rather than in the printer: a value that must not be quoted
    should not be in the artifact a future reader greps.
    """
    metrics: dict[str, Any] = {}
    # The run's own cutoffs, not the module's: a shallow candidate depth cannot
    # report recall@50, and `cutoffs_for` already dropped what would be degenerate
    # (D77). Iterating RECALL_CUTOFFS here would raise on the depth-10 cell.
    for k in sorted(run.metrics.recall):
        interval = bootstrap_interval(run.metrics.recall[k])
        metrics[f"recall@{k}"] = {
            "point": interval.point,
            "ci95": [interval.low, interval.high],
        }
    mrr = bootstrap_interval(run.metrics.rr)
    metrics["mrr"] = {"point": mrr.point, "ci95": [mrr.low, mrr.high]}
    if graded:
        ndcg = bootstrap_interval(run.metrics.ndcg)
        metrics[f"ndcg@{NDCG_CUTOFF}"] = {"point": ndcg.point, "ci95": [ndcg.low, ndcg.high]}
    diagnostics: dict[str, Any] = {}
    if run.ceiling_recall:
        ceiling = bootstrap_interval(run.ceiling_recall)
        # Doc 9 §9.3's reranking ceiling. A diagnostic rather than a metric: it is
        # recall over the whole candidate list, so it answers "could the reranker
        # have found it at all?" — not "how well is the list ordered?".
        diagnostics["ceiling_recall_at_depth"] = {
            "point": ceiling.point,
            "ci95": [ceiling.low, ceiling.high],
        }
    if run.own_paper_in_top10:
        own = bootstrap_interval(run.own_paper_in_top10)
        # Kept out of `metrics` on purpose: it is not one of §9.3's four, and a
        # reader scanning the metrics block should not mistake it for one.
        diagnostics["own_paper_share_of_top10"] = {
            "point": own.point,
            "ci95": [own.low, own.high],
        }
    return {
        "n": run.metrics.n,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "retrieve_latency_ms": run.latency_percentiles(),
    }


def _render_fit(fit: dict[str, Any]) -> list[str]:
    """The weight curve, printed so the reader can see how flat the optimum is."""
    lines = [
        f"  fusion fit weight {fit['weight_on_dense']} on dense, by {fit['objective']}"
        f"  ({fit['fitted_on']})",
        f"             MRR would choose {fit['mrr_would_choose']}"
        + ("  (agrees)" if fit["objectives_agree"] else "  (DISAGREES — the curve is flat)"),
        "             weight   recall   mrr",
    ]
    for score in fit["curve"]:
        chosen = " <-" if score["weight"] == fit["weight_on_dense"] else ""
        lines.append(
            f"             {score['weight']:>6.1f}   {score['recall']:.3f}   "
            f"{score['mrr']:.3f}{chosen}"
        )
    return [*lines, ""]


def _render(report: EvalReport) -> str:
    """The terminal view: one section per cell, deltas under the challenger."""
    cfg = report.config
    out: list[str] = [
        f"Scholiast retrieval eval  ({report.generated_at})",
        f"  dataset    {cfg['dataset_versions']}",
        f"  scope      {cfg['retrieval_scope']}",
        f"  grades     {cfg['grade_scale']}",
        f"  comparison {cfg['comparison']}",
        f"  latency    {cfg['latency_measured']}",
        "",
    ]
    if cfg.get("fusion_fit"):
        out.extend(_render_fit(cfg["fusion_fit"]))
    for label, arm in report.blocks.items():
        marker = "  [M1 DENSE BASELINE]" if arm.get("is_m1_baseline") else ""
        res = arm["resources"]
        out.append(f"=== {label}{marker}")
        out.append(f"    {arm['model']}  {res['dimensions']}d  ->  {arm['collection']}")
        out.append(
            f"    arm {arm['arm']}   depth {arm['depth']}   fusion {arm['fusion']}"
            + (f"   vs {arm['control']}" if arm.get("control") else "")
        )
        if arm.get("reranker_version"):
            out.append(f"    reranker {arm['reranker_version']}")
        built = res["index_build_seconds"]
        out.append(
            f"    vectors {res['vector_bytes_computed'] / 1e6:.1f} MB (computed)   "
            + (
                f"index built in {built}s ({res['index_chunks_per_second']} chunks/s)"
                if built is not None
                else "index reused"
            )
        )
        for name, block in arm["blocks"].items():
            out.append(f"  {name}  (n={block['n']})  -- {block['note']}")
            for metric, values in sorted(block["metrics"].items()):
                low, high = values["ci95"]
                line = f"    {metric:<12}{values['point']:>7.3f}  [{low:.3f}, {high:.3f}]"
                delta = (arm.get("deltas_vs_control") or {}).get(name, {}).get(metric)
                if delta:
                    flag = "within noise" if delta["within_noise"] else "SIGNIFICANT"
                    dl, dh = delta["ci95"]
                    line += f"   d {delta['delta']:+.3f} [{dl:+.3f}, {dh:+.3f}] {flag}"
                vs_hybrid = (arm.get("deltas_vs_hybrid") or {}).get(name, {}).get(metric)
                if vs_hybrid:
                    flag = "within noise" if vs_hybrid["within_noise"] else "SIGNIFICANT"
                    hl, hh = vs_hybrid["ci95"]
                    line += (
                        f"   | vs hybrid {vs_hybrid['delta']:+.3f} [{hl:+.3f}, {hh:+.3f}] {flag}"
                    )
                out.append(line)
            for diagnostic, values in sorted((block.get("diagnostics") or {}).items()):
                low, high = values["ci95"]
                out.append(
                    f"    {diagnostic:<24}{values['point']:>6.3f}  "
                    f"[{low:.3f}, {high:.3f}]  (diagnostic)"
                )
            lat = arm["latency"][name]
            total = lat["total"]
            out.append(
                f"    latency      p50 {total.get('p50', 0):.0f}ms  p95 {total.get('p95', 0):.0f}ms"
                f"  p99 {total.get('p99', 0):.0f}ms"
                f"   (embed p50 {lat['embed_p50_ms']}ms  search p50 {lat['search_p50_ms']}ms"
                + (
                    f"  rerank p50 {lat['rerank_p50_ms']}ms p95 {lat['rerank_p95_ms']}ms)"
                    if lat.get("rerank_p50_ms") is not None
                    else ")"
                )
            )
        out.append("")
    return "\n".join(out)


def _build_dependencies(
    dataset: DRet, spec: EmbedderSpec, *, sparse: bool = False
) -> tuple[EmbedderLike, QdrantVectorIndex, str]:
    """Wire the real adapters for one encoder. Imported here, not at module load.

    Importing the embedder pulls in torch and sentence-transformers, which costs
    seconds and a CUDA context — unacceptable for a module that `--help` should be
    able to reach, and for the tests that import this module's pure parts.

    The collection name already derives from the embedder version, so each arm of
    the ML-4 comparison indexes into its own collection without anyone naming one:
    a 384-dim encoder physically cannot be pointed at the 768-dim collection, and
    `VectorDimensionMismatch` would say so if it were.
    """
    from scholiast.application.ports.ml import Precision
    from scholiast.infrastructure.ml.embedder import LocalEmbedder
    from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

    settings = get_settings()
    embedder = LocalEmbedder(
        model_name=spec.model,
        device=settings.embedding_device,
        # Settings validates this string against the same two values (D18); the
        # conversion is here rather than in Settings because `config` must not
        # import an application port.
        precision=Precision(settings.embedding_precision),
    )
    # The arm is part of the collection identity: a sparse collection has a
    # different SCHEMA (named vectors), so reusing one name for both would make
    # `ensure_ready` validate against the wrong shape.
    collection = collection_name(dataset, embedder.version) + ("_hybrid" if sparse else "")
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=collection,
        dimensions=spec.dimensions,
        sparse=sparse,
    )
    return embedder, index, collection


def _build_reranker() -> Any:
    """The local cross-encoder, holding a GPU gate at query priority (D74).

    Imported here for the same reason as the embedder: `--help` and the tests that
    import this module must not pay for torch and a 1.1 GB model.
    """
    from scholiast.application.ports.ml import Precision
    from scholiast.infrastructure.ml.gpu import GpuPriorityGate
    from scholiast.infrastructure.ml.reranker import LocalReranker

    settings = get_settings()
    return LocalReranker(
        device=settings.embedding_device,
        precision=Precision(settings.embedding_precision),
        gate=GpuPriorityGate(),
    )


def _contention_conditions(batch_size: int) -> tuple[Condition, ...]:
    """The four ways of sharing the GPU that D79 compares.

    Idle runs through the UNGATED adapters on purpose: it is the row every other
    row is read against, and phases 4 and 5 published their idle numbers from that
    same configuration, so keeping it identical makes the comparison continuous.
    """
    return (
        Condition(
            label="idle",
            ingesting=False,
            gated=False,
            batch_size=batch_size,
            note="reference; the configuration phases 4 and 5 measured idle",
        ),
        Condition(
            label=f"contended, ungated (batch {batch_size})",
            ingesting=True,
            gated=False,
            batch_size=batch_size,
            note="phase 4 measured this half only: 3.2x at fp16 (D74)",
        ),
        Condition(
            label=f"contended, gated (batch {batch_size})",
            ingesting=True,
            gated=True,
            batch_size=batch_size,
            note="the §5.7 scheme as production would run it",
        ),
        Condition(
            label="contended, gated (batch 16)",
            ingesting=True,
            gated=True,
            batch_size=16,
            diagnostic=True,
            note=(
                "DIAGNOSTIC, not a proposed configuration: §5.7 pairs the gate with "
                "batches bounded to ~50-100 ms and D74 measured batch 32 at p50 103 / "
                "p95 135 ms, above that target. The default stays 32 (D74, D79)"
            ),
        ),
    )


def _build_contention(
    dataset: DRet, spec: EmbedderSpec, *, gated: bool
) -> tuple[Any, Any, Any, Any]:
    """Adapters for one sharing regime: embedder, index, reranker, gate.

    Built once per regime rather than per condition, so the batch-16 diagnostic
    reuses the gated models instead of loading 1.5 GB again — `batch_size` is a
    per-call argument, not part of the adapter's identity.

    **One gate, shared.** A gate per adapter would coordinate nothing: priority
    only means something when both consumers queue on the same primitive.
    """
    from scholiast.application.ports.ml import Precision
    from scholiast.infrastructure.ml.embedder import LocalEmbedder
    from scholiast.infrastructure.ml.gpu import GpuPriorityGate
    from scholiast.infrastructure.ml.reranker import LocalReranker
    from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

    settings = get_settings()
    precision = Precision(settings.embedding_precision)
    gate = GpuPriorityGate() if gated else None
    embedder = LocalEmbedder(
        model_name=spec.model,
        device=settings.embedding_device,
        precision=precision,
        gate=gate,
    )
    collection = collection_name(dataset, embedder.version) + "_hybrid"
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=collection,
        dimensions=spec.dimensions,
        sparse=True,
    )
    reranker = LocalReranker(
        device=settings.embedding_device,
        precision=precision,
        gate=gate,
    )
    return embedder, index, reranker, gate


def _resources(spec: EmbedderSpec, index_report: IndexReport, chunks: int) -> dict[str, Any]:
    """The cost side of the ML-4 tradeoff, measured where measurable.

    `vector_bytes` is **computed**, not measured — dimensions x count x 4 bytes for
    fp32 storage — and labelled as such. Qdrant's on-disk footprint includes HNSW
    graph and payload, so quoting a directory size as "the vector memory" would
    attribute the index structure to the embedding width. The arithmetic is the
    honest version of the claim REV-7 actually makes: half the dimensions is half
    the vector memory.
    """
    built = index_report.chunks_indexed > 0
    return {
        "dimensions": spec.dimensions,
        "vector_bytes_computed": spec.dimensions * chunks * 4,
        "index_build_seconds": round(index_report.seconds, 1) if built else None,
        "index_chunks_per_second": (
            round(index_report.chunks_indexed / index_report.seconds, 1)
            if built and index_report.seconds > 0
            else None
        ),
        "index_reused": not built,
    }


QUERY_SETS: tuple[tuple[str, str, bool, str], ...] = (
    ("D-RET-CORE", "core", True, "primary benchmark (D69)"),
    ("D-RET-FULL", "full", False, "robustness check; nDCG omitted - binary judgments (D69)"),
)


async def _measure(
    dataset: DRet, cell: Cell, scope: list[str]
) -> tuple[dict[str, Any], dict[str, RunReport], dict[str, Any]]:
    """Run one cell over every query set. Returns blocks, raw runs, config.

    `cell.arm` selects the retrieval configuration: "dense" is phase 2's baseline
    and goes through `run_retrieval` unchanged, so adding the lexical arm cannot
    move the number it is measured against. "lexical", "hybrid" and "hybrid_rerank"
    need sparse vectors, which means a sparse-capable adapter and its own
    collection — production's dense-only schema is never touched (D73).

    `cell.depth` and `cell.dense_weight` are phase 5's two knobs (D77, D76). Both
    are query-time: every cell reads the collection the first one built, so a
    sweep measures retrieval configurations rather than differently-built indexes.

    **Only the reranker takes the GPU gate here, deliberately (D74).** Indexing
    stays ungated: a gated embedder re-batches and measurably perturbs fp16
    vectors (108 of 256 changed), and every sparse cell shares the `_hybrid`
    collection — so gating the build would let whichever cell built it first decide
    every other cell's vectors. The embedder's gating is exercised under real
    contention in phase 6, where it is the point of the measurement.
    """
    sparse = cell.arm != "dense"
    embedder, index, collection = _build_dependencies(dataset, cell.spec, sparse=sparse)
    index_report = await build_index(dataset, index, embedder, collection=collection)
    if sparse:
        # Idempotent: `update_vectors` over deterministic point ids (D46). Runs
        # after the dense pass because it attaches to points that must already exist.
        await build_lexical_index(dataset, index)
    reranker = _build_reranker() if cell.arm == "hybrid_rerank" else None

    blocks: dict[str, Any] = {}
    runs: dict[str, RunReport] = {}
    for name, attribute, graded, note in QUERY_SETS:
        queries: list[Query] = getattr(dataset, attribute)
        if cell.arm == "dense":
            run = await run_retrieval(
                dataset, queries, index, embedder, label=name, paper_ids=scope, depth=cell.depth
            )
        else:
            run = await run_hybrid(
                dataset,
                queries,
                index,
                index,
                embedder,
                label=name,
                paper_ids=scope,
                arms=cell.arm,
                depth=cell.depth,
                reranker=reranker,
                dense_weight=cell.dense_weight,
            )
        runs[name] = run
        blocks[name] = {**_summarise(run, graded=graded), "note": note}
    return (
        blocks,
        runs,
        {
            "embedder_version": embedder.version,
            # NFR-M5: the embedder's stamp alone would present a cross-encoder's
            # ordering as the embedder's. None rather than absent for arms that do
            # not rerank, so every block has the same shape.
            "reranker_version": reranker.version if reranker is not None else None,
            "collection": collection,
            "arm": cell.arm,
            # The two phase 5 knobs, on every block rather than only on the cells
            # that vary them: a reader comparing two artifacts needs to see that
            # the other one ran at depth 50 with RRF, not infer it from silence.
            "depth": cell.depth,
            "fusion": cell.fusion,
            "resources": _resources(cell.spec, index_report, len(dataset.chunks)),
        },
    )


def _deltas(treatment: RunReport, baseline: RunReport, *, graded: bool) -> dict[str, Any]:
    """Paired deltas, treatment minus baseline (doc 9 section 9.4 point 2).

    Paired because both arms answered the SAME queries: an unpaired comparison
    would discard that and need a larger effect before anything looked real. An
    interval straddling zero is reported as within noise rather than as a win for
    whichever point estimate happens to be higher -- which is the entire
    discipline doc 9 section 9.4 exists to impose.
    """
    out: dict[str, Any] = {}
    # Only cutoffs both sides actually reported: a depth-10 cell against the
    # depth-50 control shares recall@10 and nothing deeper, and inventing the
    # missing rows would compare recall@50 against recall@10 under one label.
    shared = sorted(set(treatment.metrics.recall) & set(baseline.metrics.recall))
    pairs: list[tuple[str, list[float], list[float]]] = [
        (f"recall@{k}", treatment.metrics.recall[k], baseline.metrics.recall[k]) for k in shared
    ]
    pairs.append(("mrr", treatment.metrics.rr, baseline.metrics.rr))
    if graded:
        pairs.append((f"ndcg@{NDCG_CUTOFF}", treatment.metrics.ndcg, baseline.metrics.ndcg))
    for name, arm, control in pairs:
        delta = paired_bootstrap_delta(arm, control)
        out[name] = {
            "delta": delta.point,
            "ci95": [delta.low, delta.high],
            "within_noise": delta.low <= 0.0 <= delta.high,
        }
    return out


def _latency(run: RunReport) -> dict[str, Any]:
    return {
        "total": run.latency_percentiles(),
        "embed_p50_ms": _percentile(run.embed_ms, 0.50),
        "search_p50_ms": _percentile(run.search_ms, 0.50),
        # None for arms that do not rerank — absent rather than zero, for the same
        # reason unbuilt stages are absent from `latency_measured`.
        "rerank_p50_ms": _percentile(run.rerank_ms, 0.50),
        "rerank_p95_ms": _percentile(run.rerank_ms, 0.95),
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * q))], 2)


async def _fit_fusion(dataset: DRet, spec: EmbedderSpec, scope: list[str]) -> FusionFit:
    """Fit the weighted-fusion weight on the TRAIN split (D76).

    Reads the same collection the evaluation reads and retrieves through the same
    two calls, so the weight is fitted on the system that will be measured. The
    training queries never appear in a reported block: they exist so the ablation
    can put a tuned weighted fusion against RRF's untuned constant without the
    delta measuring the tuning.
    """
    embedder, index, collection = _build_dependencies(dataset, spec, sparse=True)
    await build_index(dataset, index, embedder, collection=collection)
    await build_lexical_index(dataset, index)
    hits = await collect_arm_hits(
        dataset, training_queries(dataset), index, index, embedder, paper_ids=scope
    )
    return fit_dense_weight(hits)


def _plan(
    base: EmbedderSpec,
    *,
    compare: bool,
    compare_arms: bool,
    compare_depths: bool,
    weight: float | None,
    dense: bool = False,
) -> list[Cell]:
    """The cells this run measures, and what each one is compared against.

    Every cell names its own control rather than the run naming one for all of
    them: the depth sweep compares each depth against the SAME ARM at depth 50, so
    a single run-wide control would either compare a pre-rerank cell against a
    reranked one or drop half the comparisons.
    """
    model = base.model.split("/")[-1]
    if weight is not None:
        control = f"rrf hybrid ({model})"
        return [
            Cell(control, base, arm="hybrid"),
            Cell(
                f"weighted hybrid ({model})",
                base,
                arm="hybrid",
                dense_weight=weight,
                control_label=control,
            ),
        ]
    if compare_depths:
        return [
            Cell(
                f"{arm} @{depth}",
                base,
                arm=arm,
                depth=depth,
                control_label="" if depth == INCUMBENT_DEPTH else f"{arm} @{INCUMBENT_DEPTH}",
            )
            for depth in DEPTH_SWEEP
            for arm in ("hybrid", "hybrid_rerank")
        ]
    if compare_arms:
        control = f"dense ({model})"
        return [
            Cell(
                f"{arm} ({model})",
                base,
                arm=arm,
                control_label="" if arm == "dense" else control,
                is_m1_baseline=arm == "dense",
            )
            for arm in ("dense", "lexical", "hybrid", "hybrid_rerank")
        ]
    if compare:
        return [
            Cell(
                spec.label,
                spec,
                control_label="" if spec.baseline else base.label,
                is_m1_baseline=spec.baseline,
            )
            for spec in COMPARED_EMBEDDERS
        ]
    if dense:
        return [Cell(base.label, base, is_m1_baseline=True)]
    # What production retrieves with (D174): RRF over both arms, no reranker,
    # at the incumbent depth - the same cell `--compare-arms` labels hybrid.
    return [Cell(f"hybrid ({model})", base, arm="hybrid")]


def _validate(plan: Sequence[Cell]) -> None:
    """Refuse a plan that cannot be reported — before any measuring happens.

    Both failures are quiet in the artifact and expensive in wall-clock: two cells
    sharing a label overwrite each other's block, and a control the plan does not
    contain drops a comparison. A depth sweep is ten cells and minutes of GPU
    time, so the check belongs before the first cell rather than after the last.
    """
    labels = [cell.label for cell in plan]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two cells share a label, so one would overwrite the other: {labels}")
    unknown = {cell.control_label for cell in plan if cell.control_label} - set(labels)
    if unknown:
        raise ValueError(
            f"cells are measured against {sorted(unknown)}, which this run does not "
            f"contain: {sorted(labels)}"
        )


def _comparison(plan: Sequence[Cell], *, compare: bool, fit: FusionFit | None) -> str:
    """What this run compared — built from the plan, never written by hand.

    A hand-written header went stale the moment the rerank arm was added, and the
    artifact then described a three-arm comparison it had not run.
    """
    if fit is not None:
        return (
            f"ML-7 fusion ablation (D76): RRF vs weighted min-max score fusion at "
            f"dense weight {fit.weight}, fitted on {fit.n_queries} train-split "
            f"queries by {fit.objective}"
        )
    depths = sorted({cell.depth for cell in plan})
    if len(depths) > 1:
        return (
            "REV-4 candidate-depth sweep (D77): "
            + " / ".join(str(depth) for depth in depths)
            + f"; {DIAGNOSTIC_DEPTH} is a diagnostic, not a production candidate; "
            f"each cell is compared with the same arm at depth {INCUMBENT_DEPTH}"
        )
    arms = [cell.arm for cell in plan]
    if len(set(arms)) > 1:
        return "FR-Q3 retrieval-arm comparison (D73, D74): " + " / ".join(arms)
    return "ML-4 embedding-model comparison" if compare else "single arm"


async def _run(
    *,
    compare: bool,
    compare_arms: bool,
    compare_depths: bool = False,
    compare_fusion: bool = False,
    dense: bool = False,
) -> EvalReport:
    dataset = build_dret(load_snapshot())
    verify_split(dataset)
    verify_core(dataset)
    # D70: the whole corpus, every query. Stated here, once, where it is passed.
    scope = sorted(dataset.papers)

    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    # Fitting happens BEFORE the plan exists, because the fitted weight is part of
    # the configuration the plan describes (D76).
    fit = await _fit_fusion(dataset, base, scope) if compare_fusion else None
    plan = _plan(
        base,
        compare=compare,
        compare_arms=compare_arms,
        compare_depths=compare_depths,
        weight=fit.weight if fit else None,
        dense=dense,
    )

    _validate(plan)

    cells: dict[str, Any] = {}
    raw: dict[str, dict[str, RunReport]] = {}
    for cell in plan:
        blocks, runs, cell_config = await _measure(dataset, cell, scope)
        raw[cell.label] = runs
        cells[cell.label] = {
            **cell_config,
            "model": cell.spec.model,
            "is_m1_baseline": cell.is_m1_baseline,
            # Named per block, because different cells in one run can have
            # different controls (the depth sweep's do).
            "control": cell.control_label or None,
            "blocks": blocks,
            "latency": {name: _latency(run) for name, run in runs.items()},
        }

    for cell in plan:
        if not cell.control_label:
            continue
        cells[cell.label]["deltas_vs_control"] = {
            name: _deltas(raw[cell.label][name], raw[cell.control_label][name], graded=graded)
            for name, _attribute, graded, _note in QUERY_SETS
        }

    # The reranker's OWN contribution. Its delta against dense-only mixes two
    # effects — fusion and reranking — so on its own it cannot say whether the
    # cross-encoder earned its latency. Against hybrid, the only difference is
    # the rerank step, on the identical fused shortlist.
    hybrid_label = next((lbl for lbl in cells if lbl.startswith("hybrid (")), None)
    rerank_label = next((lbl for lbl in cells if lbl.startswith("hybrid_rerank (")), None)
    if hybrid_label and rerank_label:
        cells[rerank_label]["deltas_vs_hybrid"] = {
            name: _deltas(raw[rerank_label][name], raw[hybrid_label][name], graded=graded)
            for name, _attribute, graded, _note in QUERY_SETS
        }

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "configuration": (
                "per cell: each block's `arm`, `depth`, `fusion`, `embedder_version` "
                "and `reranker_version`; every cell reads one collection, built once"
                if len(plan) > 1
                else "dense-only (D56 ablation baseline)"
            ),
            "dataset_versions": dataset.versions,
            "retrieval_scope": f"whole corpus, {len(scope)} papers, "
            f"{len(dataset.chunks)} chunks (D70)",
            "grade_scale": dataset.grade_scale,
            "corpus_chunks": len(dataset.chunks),
            "comparison": _comparison(plan, compare=compare, fit=fit),
            # The fitted weight and the curve behind it, so a reader can see how
            # weakly or strongly the choice was determined (D76). None when this
            # run fitted nothing.
            "fusion_fit": (
                None
                if fit is None
                else {
                    "weight_on_dense": fit.weight,
                    "objective": fit.objective,
                    "fitted_on": f"D-RET train split, {fit.n_queries} queries, "
                    "disjoint from the evaluation papers (D76)",
                    "mrr_would_choose": fit.mrr_weight,
                    "objectives_agree": fit.objectives_agree,
                    "curve": [
                        {"weight": score.weight, "recall": score.recall, "mrr": score.mrr}
                        for score in fit.curve
                    ],
                }
            ),
            # Scope of the latency figures, stated in the artifact rather than
            # only in doc 9 §9.8. A reader meets these numbers here, and a table
            # that says "latency" without saying WHICH stages invites the
            # assumption that it covers the pipeline. Rows for unbuilt stages
            # would be fabrication, so they are absent rather than zero.
            "latency_measured": (
                "retrieval stage only, idle system, CUDA init and model load excluded "
                "by untimed warm-ups. embed = query preparation: dense encode (dense), "
                "tokenise (lexical), tokenise + dense encode (hybrid, hybrid_rerank). "
                "search = store queries: one (dense, lexical), two plus fusion by each "
                "block's `fusion` rule (hybrid, hybrid_rerank). rerank = cross-encoder "
                "over the fused candidate list at each block's `depth` (hybrid_rerank "
                "only). total = embed + search + rerank, NFR-P2's stage. context/TTFT/"
                "complete (generation path) and the two non-idle conditions, including "
                "rerank under concurrent ingestion, are NOT measured yet"
            ),
        },
        blocks=cells,
    )


async def run_contention(*, repeats: int = DEFAULT_REPEATS) -> EvalReport:
    """Doc 8 deliverable 6: the number that says whether §5.7's scheme works.

    **This harness never builds the collection.** Every condition reads the index
    phases 3-5 built, and a gated embedder re-batches and perturbs fp16 vectors
    (D74, 108 of 256 changed) — so building here could silently move numbers other
    phases published. An incomplete collection is refused with the command that
    builds it properly instead.
    """
    dataset = build_dret(load_snapshot())
    verify_split(dataset)
    verify_core(dataset)
    scope = sorted(dataset.papers)
    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    settings = get_settings()

    adapters: dict[bool, tuple[Any, Any, Any, Any]] = {}
    reports: list[ConditionReport] = []
    blocks: dict[str, Any] = {}
    # Real corpus text, in a fixed order, so every condition ingests the same work.
    load_texts = [dataset.chunks[cid].text for cid in sorted(dataset.chunks)]

    for condition in _contention_conditions(settings.embedding_batch_size):
        if condition.gated not in adapters:
            adapters[condition.gated] = _build_contention(dataset, base, gated=condition.gated)
        embedder, index, reranker, gate = adapters[condition.gated]
        await index.ensure_ready()
        present = await index.count()
        if present != len(dataset.chunks):
            raise ValueError(
                f"the eval collection holds {present} of {len(dataset.chunks)} chunks. "
                "This harness must not build it — a gated embedder perturbs fp16 "
                "vectors (D74) and every published metric reads this collection. "
                "Run `uv run eval --compare-arms` first."
            )
        report = await measure_condition(
            condition,
            dataset,
            dataset.core,
            index,
            index,
            embedder,
            reranker,
            paper_ids=scope,
            load_texts=load_texts,
            repeats=repeats,
            gate=gate,
        )
        reports.append(report)
        blocks[condition.label] = {
            "ingesting": condition.ingesting,
            "gated": condition.gated,
            "batch_size": condition.batch_size,
            "diagnostic": condition.diagnostic,
            "note": condition.note,
            "queries": report.queries,
            "seconds": round(report.seconds, 1),
            "rerank_ms": report.percentiles(report.rerank_ms),
            "stage_total_ms": report.percentiles(report.total_ms),
            "embed_ms": report.percentiles(report.embed_ms),
            "search_ms": report.percentiles(report.search_ms),
            "ingestion": {
                "batches_during_queries": report.batches_during,
                "chunks_during_queries": report.chunks_during,
                "chunks_per_second": report.ingestion_chunks_per_second,
            },
            "gate_acquisitions": report.acquisitions,
            "retrieval_metrics": report.metrics,
        }

    # Only timing may vary. Checked across all four before anything is archived.
    verify_same_retrieval(reports)

    embedder, _index, reranker, _gate = adapters[False]
    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "measurement": (
                "rerank latency under concurrent ingestion — doc 8 deliverable 6, "
                "doc 9 §9.8's second condition, the test of §5.7's priority scheme (D79)"
            ),
            "conditions": (
                "idle / contended-ungated / contended-gated at the production batch "
                f"size ({settings.embedding_batch_size}), plus contended-gated at batch "
                "16 as a DIAGNOSTIC. Idle runs the ungated adapters, the configuration "
                "phases 4 and 5 measured idle with"
            ),
            "ingestion_load": (
                f"real corpus chunks embedded continuously at {settings.embedding_precision}, "
                "cycling the corpus, NOTHING WRITTEN — the contention is GPU-side and a "
                "gated embedder would perturb the collection every metric reads (D74, D79)"
            ),
            "query_workload": (
                f"D-RET-CORE (n={len(dataset.core)}) x {repeats} passes through the "
                "hybrid_rerank arm at depth 50, the whole retrieval stage including the "
                "query embed, because NFR-P2's budget is the stage"
            ),
            "budget": "NFR-P2: hybrid + fusion + rerank at p95 <= 800 ms",
            "embedder_version": embedder.version,
            "reranker_version": reranker.version,
            "dataset_versions": dataset.versions,
            "not_measured": (
                "end-to-end ingestion (NFR-P6 covers parse + chunk + embed + write and is "
                "a different number), concurrent QUERY load (NFR-S4's 20 users), and the "
                "generation path"
            ),
            "reproducibility": (
                "latency is not reproducible in FR-E4's sense; the retrieval metrics beside "
                "each condition are, and are verified identical across all conditions — a "
                "difference there would mean the harness changed more than the timing"
            ),
        },
        blocks=blocks,
    )


def _pdf_check_index(spec: EmbedderSpec, collection: str) -> Any:
    from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

    return QdrantVectorIndex(
        get_settings().qdrant_url,
        collection=collection,
        dimensions=spec.dimensions,
        sparse=False,
    )


async def run_pdf_check(*, count: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> EvalReport:
    """D68's validity check: the same papers, parsed from real PDFs (D81).

    The two sides differ in exactly one thing — where the text came from. Same
    rows, same builder, same chunker, same matcher, same queries, same scope.
    """
    import httpx

    from scholiast.infrastructure.parsing.grobid import GrobidParser

    settings = get_settings()
    rows = load_snapshot()
    published = build_dret(rows)
    verify_split(published)
    sampled = sample_papers(published, count=count, seed=seed)
    subset = [row for row in rows if str(row["id"]) in set(sampled)]

    parser = GrobidParser(settings.grobid_url)
    async with httpx.AsyncClient() as client:
        documents = await fetch_and_parse(
            sampled, cache_dir=PDF_CACHE_DIR, client=client, parser=parser
        )

    def from_pdf(row: dict[str, Any], *, normaliser: Any) -> Any:
        return paper_from_document(row, documents[str(row["id"])], normaliser=normaliser)

    # One matcher on both sides (D81): the check varies the parse, not the lookup.
    json_side = build_dret(subset, normaliser=normalise_for_pdf_match)
    pdf_side = build_dret(subset, paper_factory=from_pdf, normaliser=normalise_for_pdf_match)
    comparable = comparable_queries(json_side, pdf_side)
    if not comparable.shared:
        raise ValueError(
            "no query carries judgments on both builds, so there is nothing to "
            "compare — the PDF parse resolved no evidence at all"
        )

    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    embedder, _unused, _collection = _build_dependencies(published, base)
    # The sample and the parser are part of the corpus identity: two runs over
    # different samples, or the same sample under a different GROBID, must not
    # share a collection and silently reuse the wrong vectors.
    digest = hashlib.sha256(("|".join(sampled) + parser.version).encode("utf-8")).hexdigest()[:8]

    blocks: dict[str, Any] = {}
    runs: dict[str, RunReport] = {}
    for label, side, note in (
        (JSON_SIDE, json_side, "the published build: QASPER full_text JSON (D68 primary)"),
        (PDF_SIDE, pdf_side, f"real arXiv PDFs through {parser.version} and the same chunker"),
    ):
        collection = f"{collection_name(side, embedder.version)}_pdfcheck_{digest}_{label[:4]}"
        index = _pdf_check_index(base, collection)
        index_report = await build_index(side, index, embedder, collection=collection)
        run = await run_retrieval(
            side,
            queries_by_id(side, comparable.shared),
            index,
            embedder,
            label=label,
            paper_ids=sampled,
        )
        runs[label] = run
        blocks[label] = {
            "note": note,
            "collection": collection,
            "chunks": len(side.chunks),
            "queries_scored": run.metrics.n,
            "judgments": len(side.judgments),
            "evidence": side.evidence.as_dict(),
            # WHY evidence failed to reach a chunk. Without it, an unresolved count
            # is uninterpretable: a parse that lost the text and two sources that
            # render the same sentence differently look identical in a total.
            "mapping_diagnosis": diagnose_mapping(subset, side.papers).as_dict(),
            **_summarise(run, graded=True),
            "latency": _latency(run),
            "resources": _resources(base, index_report, len(side.chunks)),
        }
    blocks[PDF_SIDE]["control"] = JSON_SIDE
    blocks[PDF_SIDE]["deltas_vs_control"] = {
        "paper-scoped": _deltas(runs[PDF_SIDE], runs[JSON_SIDE], graded=True)
    }

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "measurement": (
                "D68's real-PDF validity check: does retrieval hold when the corpus is "
                "parsed from PDFs by the production path instead of QASPER's JSON? (D81)"
            ),
            "sample": (
                f"{len(sampled)} eval-split papers chosen by seed {seed}: {', '.join(sampled)}"
            ),
            "varied": (
                "the PARSE only. Both sides run through the same `build_dret`, chunker, "
                "evidence location, grading, split and CORE rules; the PDF side swaps in a "
                "paper factory whose text came from GROBID"
            ),
            "matching": (
                "both sides matched with one normaliser (NFKC + de-hyphenation across line "
                "breaks + whitespace collapse + casefold), used for this check ONLY — the "
                "vendored D-RET keeps its published `normalise_for_match` (D81)"
            ),
            "comparison_basis": (
                "the delta covers queries judged on BOTH builds; a query judged on one side "
                "only measures the matcher rather than retrieval, and is counted separately"
            ),
            "comparable": comparable.report,
            "retrieval_scope": (
                f"paper-scoped: both sides retrieve within the {len(sampled)} sampled papers, "
                "because the PDF build contains only those. THIS DEPARTS FROM D70's "
                "whole-corpus scope, so these absolute numbers are NOT the headline ones — a "
                "top-50 over a few hundred chunks compresses Recall@50 toward 1.0. Recall@10 "
                "and MRR carry the comparison"
            ),
            "arm": "dense-only (D56), the path M1 ships",
            "grade_scale": published.grade_scale,
            "dataset_versions": published.versions,
            "parser_version": parser.version,
            "embedder_version": embedder.version,
        },
        blocks=blocks,
    )


async def run_smoke() -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Measure the smoke subset on CPU (FR-E6, D82). Returns (report, metrics).

    **CPU and fp32 deliberately.** CI has no GPU, and the adapter forces fp32 there
    (D18) — so these values are a different configuration from the published ones
    and exist to detect change, not to be quoted. The device is passed explicitly
    rather than read from settings: a smoke run that silently used CUDA on a dev
    machine would compare against a baseline no CI runner could reproduce.

    One collection serves both arms. The full eval keeps dense and hybrid apart so
    the published baseline cannot drift; here the point is speed, and the sparse
    schema answers a dense search the same way.
    """
    from scholiast.application.ports.ml import Precision
    from scholiast.infrastructure.ml.embedder import LocalEmbedder
    from scholiast.infrastructure.vector.qdrant import QdrantVectorIndex

    rows = load_snapshot()
    published = build_dret(rows)
    verify_split(published)
    sampled = sample_papers(published, count=SMOKE_PAPERS, seed=SMOKE_SEED)
    dataset = build_dret([row for row in rows if str(row["id"]) in set(sampled)])
    queries = dataset.full

    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    embedder = LocalEmbedder(base.model, device="cpu", precision=Precision.FP32)
    digest = hashlib.sha256("|".join(sampled).encode("utf-8")).hexdigest()[:8]
    collection = f"{collection_name(dataset, embedder.version)}_smoke_{digest}"
    # `sparse=True`: the hybrid arm needs the named-vector schema, and the dense
    # arm reads the same collection rather than paying for a second index build.
    index = QdrantVectorIndex(
        get_settings().qdrant_url,
        collection=collection,
        dimensions=base.dimensions,
        sparse=True,
    )
    await build_index(dataset, index, embedder, collection=collection)
    await build_lexical_index(dataset, index)

    dense = await run_retrieval(dataset, queries, index, embedder, label="dense", paper_ids=sampled)
    hybrid = await run_hybrid(
        dataset, queries, index, index, embedder, label="hybrid", paper_ids=sampled, arms="hybrid"
    )
    metrics = {
        "dense": {name: round(value, 6) for name, value in dense.metrics.means().items()},
        "hybrid": {name: round(value, 6) for name, value in hybrid.metrics.means().items()},
    }
    report = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": {
            "purpose": (
                "FR-E6 regression gate (D82). These numbers are a fixed point for "
                "detecting CHANGE on a 10-paper subset at CPU/fp32 — they are not the "
                "published results, which are whole-corpus at GPU/fp16"
            ),
            "papers": sampled,
            "seed": SMOKE_SEED,
            "chunks": len(dataset.chunks),
            "queries": len(queries),
            "arms": ["dense", "hybrid"],
            "reranking": "absent: on CPU it costs minutes, which is the whole job budget",
            "scope": "paper-scoped to the subset, so absolute values sit above D70's",
            "device": "cpu",
            "precision": "fp32 (forced on CPU by D18)",
            "embedder_version": embedder.version,
            "dataset_versions": dataset.versions,
            "tolerance": TOLERANCE,
            "regenerate": "uv run eval --smoke --update-baseline",
        },
        "metrics": metrics,
    }
    return report, metrics


#: The arms run D168's precondition must reproduce, and the blocks it compares.
#: Named, not discovered: "the latest arms artifact" would quietly move the
#: reference the day anyone re-ran `--compare-arms`.
GATE_REPRODUCES = DEFAULT_REPORT_DIR / "arms-20260924T053549+0000.json"
GATE_DENSE_BLOCK = "dense (bge-base-en-v1.5)"
GATE_HYBRID_BLOCK = "hybrid (bge-base-en-v1.5)"
#: SHA-256 of that file's bytes, pinned. `var/` is gitignored, so the name alone
#: says which file and not what is in it: an edited or replaced copy would make
#: the precondition reproduce that copy rather than the numbers M2 published.
GATE_REFERENCE_SHA256 = "53accec29c820489d9d7290a56aa8b3dd5e72ce49b161567df57f694cc4d51b8"


def _load_reference(path: Path) -> dict[str, Any]:
    """The arms artifact the precondition reproduces, read before anything is measured.

    Refused unless its bytes hash to `GATE_REFERENCE_SHA256`, and the gate calls
    this first, so a modified or replaced reference stops the run before any
    Qdrant call or measurement.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is the arms run D168's precondition reproduces, and it is missing"
        )
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != GATE_REFERENCE_SHA256:
        raise ValueError(
            f"{path} does not match the pinned M2 reference (D168): sha256 {digest}, "
            f"expected {GATE_REFERENCE_SHA256}. The precondition reproduces the arms "
            "run M2 published, not whatever file now has its name."
        )
    loaded: dict[str, Any] = json.loads(raw.decode("utf-8"))
    return loaded


def _identity_mismatches(
    archived: dict[str, Any], *, collections: dict[str, str], embedder_version: str
) -> list[str]:
    """Where the collections this run would read differ from the ones M2's arms read.

    D168 (as amended): each arm is reproduced against the collection the archive
    records for it. The names are derived from the dataset and embedder versions,
    so a changed precision or device setting names a different collection; this
    says so before any Qdrant call, instead of letting the run refuse later for a
    reason that would look like a missing build.
    """
    blocks = archived.get("blocks", {})
    found: list[str] = []
    for arm, label in (("dense", GATE_DENSE_BLOCK), ("hybrid", GATE_HYBRID_BLOCK)):
        recorded = blocks.get(label, {})
        for key, ours in (("collection", collections[arm]), ("embedder_version", embedder_version)):
            theirs = recorded.get(key)
            if ours != theirs:
                found.append(f"{arm}: {key} would be {ours!r}, the archive records {theirs!r}")
    return found


def _gate_block(runs: ScopedRuns, *, graded: bool) -> dict[str, Any]:
    """One query set at one scope size: both arms, the paired deltas, the spread."""

    def quality(run: RunReport) -> dict[str, Any]:
        summary = _summarise(run, graded=graded)
        # The gate measures quality only (gate.py); its clocks include per-call
        # warm-ups and would be misread as retrieval latency.
        summary.pop("retrieve_latency_ms")
        return summary

    spread = {
        metric: {"per_draw": values, "min": min(values), "max": max(values)}
        for metric, values in runs.draw_deltas.items()
        if values and (graded or metric == "recall@10")
    }
    return {
        "dense": quality(runs.dense),
        "hybrid": quality(runs.hybrid),
        "deltas_hybrid_vs_dense": _deltas(runs.hybrid, runs.dense, graded=graded),
        "draw_spread_of_delta": spread,
    }


async def run_workspace_gate(*, reproduces: Path = GATE_REPRODUCES) -> EvalReport:
    """D167's gate, as D168 fixed it: the precondition first, then three scope sizes.

    **Read-only against Qdrant.** Dense reads the dense-only collection and hybrid
    the sparse one, exactly as `--compare-arms` did, and nothing is created,
    indexed or written:

    1. The collection names and the embedder version are checked against the ones
       the archive records for each arm (D168 as amended) before any Qdrant call.
    2. Each collection is checked with `existing_count`, which only reads. Not
       `ensure_ready`, which creates a missing collection and re-asserts its
       payload index; a missing or incomplete collection is refused, never built,
       because a rebuilt one is not the one M2 measured.
    3. If the precondition fails, no scoped measurement runs: there is no verdict,
       and no scoped number exists to be mistaken for a result.
    """
    archived = _load_reference(reproduces)

    dataset = build_dret(load_snapshot())
    verify_split(dataset)
    verify_core(dataset)
    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    embedder, dense_index, dense_collection = _build_dependencies(dataset, base)
    _, hybrid_index, hybrid_collection = _build_dependencies(dataset, base, sparse=True)
    collections = {"dense": dense_collection, "hybrid": hybrid_collection}
    # Constructing the adapters made no Qdrant call (it connects on first use), so
    # this refusal happens before the gate has touched the store at all.
    wrong = _identity_mismatches(
        archived, collections=collections, embedder_version=embedder.version
    )
    if wrong:
        raise ValueError(
            "the gate would not read the collections M2's arms read (D168):\n  "
            + "\n  ".join(wrong)
            + "\nThe names derive from the dataset and embedder versions, so check the "
            "embedding precision and device settings before anything else."
        )
    for index, collection in ((dense_index, dense_collection), (hybrid_index, hybrid_collection)):
        present = await index.existing_count()
        if present is None:
            raise ValueError(
                f"{collection} does not exist. The gate reads the collections M2's arms "
                "measured and never creates one (D168); a new collection would not be "
                "the one the archive was measured on."
            )
        if present != len(dataset.chunks):
            raise ValueError(
                f"{collection} holds {present} of {len(dataset.chunks)} chunks. The gate "
                "reads the collections M2's arms measured and must not build them; a "
                "rebuilt one would make the reproduction precondition compare different "
                "vectors (D168)."
            )

    async def measure(name: str, attribute: str, **how: Any) -> ScopedRuns:
        queries: list[Query] = getattr(dataset, attribute)
        return await measure_scoped(
            dataset, queries, dense_index, hybrid_index, embedder, label=name, **how
        )

    # The precondition: M2's scope, one draw (every draw of the whole corpus is the
    # same scope), compared with the archived numbers exactly.
    reproduction: dict[str, dict[str, Any]] = {"dense": {}, "hybrid": {}, "deltas": {}}
    for name, attribute, graded, _note in QUERY_SETS:
        runs = await measure(name, attribute, scope=whole_corpus(dataset), draws=1)
        reproduction["dense"][name] = _summarise(runs.dense, graded=graded)
        reproduction["hybrid"][name] = _summarise(runs.hybrid, graded=graded)
        reproduction["deltas"][name] = _deltas(runs.hybrid, runs.dense, graded=graded)
    # Through JSON, as the archive went: tuples become lists and nothing else moves.
    mismatches = compare_with_archive(
        json.loads(json.dumps(reproduction)),
        archived,
        dense=GATE_DENSE_BLOCK,
        hybrid=GATE_HYBRID_BLOCK,
    )

    # A failed precondition stops here (D168): the scoped runs would rest on
    # retrieval that does not reproduce M2, so none is measured, and the artifact
    # carries no scoped number that could be quoted without its verdict.
    sizes: dict[str, Any] = {}
    deciding: dict[str, Any] = {}
    if not mismatches:
        for size in SIZES:
            sizes[f"N={size}"] = {
                name: _gate_block(
                    await measure(name, attribute, scope=drawn(dataset, size=size), draws=DRAWS),
                    graded=graded,
                )
                for name, attribute, graded, _note in QUERY_SETS
            }
        deciding = {
            name: block["deltas_hybrid_vs_dense"]
            for name, block in sizes[f"N={DECIDING_SIZE}"].items()
        }

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "comparison": "D167's workspace-scale gate (D168): RRF hybrid vs dense-only, "
            "each query ranked against its own paper plus papers drawn by hash",
            "dataset_versions": dataset.versions,
            "grade_scale": dataset.grade_scale,
            "corpus_papers": len(dataset.papers),
            "corpus_chunks": len(dataset.chunks),
            "embedder_version": embedder.version,
            "collections": {"dense": dense_collection, "hybrid": hybrid_collection},
            "scope": {
                "sizes": list(SIZES),
                "deciding_size": DECIDING_SIZE,
                "draws_per_query": DRAWS,
                "draw": "uniform over every other corpus paper: rank each by "
                f'sha256("{GATE_KEY}:{{draw}}:{{query_id}}:{{paper_id}}"), take the '
                "first N - 1; sizes nested; a query's metric is the mean over its draws",
            },
            "depth": DEPTH,
            "fusion": "rrf",
            "rule": RULE,
            "reproduces": str(reproduces),
            "latency_measured": "not measured: the gate measures retrieval quality only",
            "does_not_establish": [
                "LaTeX-derived corpus with BIBREF tokens (T12); binary 2/0 judgments (D69)",
                "a random scope is not a real user's single-topic workspace",
                "recall@10 stands in for QA's 8 evidence passages",
                "retrieval quality, not answer quality",
            ],
        },
        blocks={
            "reproduction": {
                "reproduced": not mismatches,
                "mismatches": mismatches,
                "compared_against": [GATE_DENSE_BLOCK, GATE_HYBRID_BLOCK],
            },
            "scoped_measurement": {
                "run": not mismatches,
                "reason": None if not mismatches else "the precondition failed (D168)",
            },
            "sizes": sizes,
            "verdict": decide(not mismatches, deciding),
        },
    )


def _render_gate(report: EvalReport) -> str:
    """The verdict first, then each size with its deciding metrics marked."""
    cfg, blocks = report.config, report.blocks
    verdict, reproduction = blocks["verdict"], blocks["reproduction"]
    out = [
        f"D167 workspace-scale gate  ({report.generated_at})",
        f"  dataset    {cfg['dataset_versions']}",
        f"  grades     {cfg['grade_scale']}",
        f"  scope      {cfg['scope']['draw']}",
        f"  rule       {cfg['rule']}",
        "",
        f"  precondition: whole-corpus run vs {cfg['reproduces']}: "
        + ("REPRODUCED exactly" if reproduction["reproduced"] else "NOT REPRODUCED"),
    ]
    out.extend(f"    {line}" for line in reproduction["mismatches"][:20])
    if len(reproduction["mismatches"]) > 20:
        out.append(f"    ... and {len(reproduction['mismatches']) - 20} more")
    out.append(f"  VERDICT: {verdict['outcome']}")
    for key in ("significantly_worse", "significantly_better", "within_noise"):
        if verdict.get(key):
            out.append(f"    {key.replace('_', ' ')}: {', '.join(verdict[key])}")
    scoped = blocks.get("scoped_measurement", {"run": True})
    if not scoped["run"]:
        out.append(f"  scoped measurement: not run, {scoped['reason']}")
    out.append("")
    for size, by_set in blocks["sizes"].items():
        deciding = size == f"N={DECIDING_SIZE}"
        out.append(f"=== {size}" + ("  [DECIDES]" if deciding else "  (reported only)"))
        for name, block in by_set.items():
            out.append(f"  {name}  (n={block['dense']['n']})")
            for metric, delta in sorted(block["deltas_hybrid_vs_dense"].items()):
                low, high = delta["ci95"]
                dense = block["dense"]["metrics"][metric]["point"]
                hybrid = block["hybrid"]["metrics"][metric]["point"]
                flag = "within noise" if delta["within_noise"] else "SIGNIFICANT"
                out.append(
                    f"    {metric:<12}dense {dense:.3f}  hybrid {hybrid:.3f}   "
                    f"d {delta['delta']:+.3f} [{low:+.3f}, {high:+.3f}] {flag}"
                )
            for metric, spread in block["draw_spread_of_delta"].items():
                out.append(
                    f"    {metric} delta by draw: {spread['min']:+.3f} .. {spread['max']:+.3f}"
                )
        out.append("")
    return "\n".join(out)


async def run_abstention(
    *, reference: Path = GATE_REPRODUCES, progress_dir: Path = DEFAULT_REPORT_DIR
) -> EvalReport:
    """D177's scoring: three non-fused signals for every instance, then the fixed rule.

    **Read-only against Qdrant**, under the gate's checks (D177, clarified
    2026-10-28):

    1. The collection names and the embedder version are checked against the ones
       M2's archive records, before any Qdrant call.
    2. The hybrid collection, which serves all three signals as production's
       hybrid path does, must hold every D-RET chunk by `existing_count`, which
       only reads. A missing or partial collection is refused, never built.

    No LLM is called and no gated embedder is built (D74). Rows are written to a
    progress file as they are scored, one file per identity (sets, collection,
    models), so a standby costs one instance and a rerun resumes (D160).
    """
    archived = _load_reference(reference)
    sets = load_sets()
    sets_sha256 = hashlib.sha256(answerability.SETS.read_bytes()).hexdigest()
    dataset = build_dret(load_snapshot())
    verify_split(dataset)
    verify_core(dataset)
    base = next(s for s in COMPARED_EMBEDDERS if s.baseline)
    embedder, _dense_index, dense_collection = _build_dependencies(dataset, base)
    _, hybrid_index, hybrid_collection = _build_dependencies(dataset, base, sparse=True)
    wrong = _identity_mismatches(
        archived,
        collections={"dense": dense_collection, "hybrid": hybrid_collection},
        embedder_version=embedder.version,
    )
    if wrong:
        raise ValueError(
            "the abstention run would not read the collections M2's arms read (D177):\n  "
            + "\n  ".join(wrong)
        )
    present = await hybrid_index.existing_count()
    if present is None:
        raise ValueError(
            f"{hybrid_collection} does not exist. The abstention run reads the collection "
            "M2's hybrid arm measured and never creates one (D177)."
        )
    if present != len(dataset.chunks):
        raise ValueError(
            f"{hybrid_collection} holds {present} of {len(dataset.chunks)} chunks. The "
            "abstention run must not build it; a rebuilt collection is not M2's (D177)."
        )

    reranker = _build_reranker()
    todo = instances(sets, dataset)
    identity = {
        "sets_sha256": sets_sha256,
        "collection": hybrid_collection,
        "embedder_version": embedder.version,
        "reranker_version": reranker.version,
        "lexical_version": LEXICAL_VERSION,
        "depth": DEPTH,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    progress = progress_dir / f"abstention-progress-{digest}.jsonl"
    done = read_progress(progress)
    adapters: dict[str, Any] = {
        "dataset": dataset,
        "index": hybrid_index,
        "lexical": hybrid_index,
        "embedder": embedder,
        "reranker": reranker,
    }
    with progress_writer(progress) as sink:
        rows = await score_all(todo, done=done, sink=sink, **adapters)
    sample = reproduce_sample(todo)
    again = [await score_one(instance, vectors={}, **adapters) for instance in sample]
    by_key = {row.key: row for row in rows}
    analysis = analyse(rows)

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "comparison": "D177's abstention signals: the top dense cosine, top lexical "
            "score and top cross-encoder score in a scope, none fused, separating "
            "answerable from unanswerable questions",
            "decision": "D177 (clarified 2026-10-22 and 2026-10-28)",
            "set_version": sets["set_version"],
            "sets_sha256": sets_sha256,
            "sheet_sha256": sets["sheet_sha256"],
            "dataset_versions": dataset.versions,
            "corpus_papers": len(dataset.papers),
            "corpus_chunks": len(dataset.chunks),
            **identity,
            "abstention_version": ABSTENTION_VERSION,
            "signals": {
                "dense": "top cosine of the dense arm in the scope (production's, D174)",
                "lexical": "top IDF-weighted sparse score in the scope; indexes BIBREF tokens",
                "rerank": "top cross-encoder score over the RRF top 50; measured, not adopted",
            },
            "scope": {
                "sizes": list(SIZES),
                "deciding_size": DECIDING_SIZE,
                "draws_per_question": DRAWS,
                "draw": f"D168's scope_for, key {GATE_KEY!r}",
                "withheld_sizes": list(WITHHELD_SIZES),
                "withheld": "the hand-checked 10-paper scope, and its first 5 at size 5; "
                "not asked at 25, where 15 papers were never checked",
            },
            "weights": "each instance 1/k, k its question's instances at that size",
            "rule": ABSTENTION_RULE,
            "reference": str(reference),
            "progress_file": str(progress),
            "instances": len(rows),
            "resumed_rows": sum(1 for instance in todo if instance.key in done),
            "llm_calls": 0,
            "does_not_establish": [
                "QASPER 'unanswerable' often means the paper discusses the topic "
                "without answering it (D177)",
                "LaTeX-derived corpus with BIBREF tokens (T12), which the lexical signal indexes",
                "a random scope is not a real user's single-topic workspace",
                "withheld-paper labels are one labeller's, judged on titles and abstracts",
                "retrieval scores only: whether the model itself refuses is phase 2",
                "eval holds 9 QASPER-unanswerable questions, so their interval is wide",
            ],
        },
        blocks={
            "verdict": analysis["verdict"],
            "chosen": analysis["chosen"],
            "train_fits": analysis["train_fits"],
            "eval": analysis["eval"],
            "sizes": analysis["sizes"],
            "reproducibility": {
                "instances": len(sample),
                "of": REPRODUCE_SAMPLE,
                "largest_abs_difference": largest_differences(
                    [by_key[instance.key] for instance in sample], again
                ),
                "gated": False,
            },
            # Every instance's three scores, so any curve can be recomputed offline.
            "instances": {"rows": [row.to_json() for row in rows]},
        },
    )


def _render_abstention(report: EvalReport) -> str:
    """The verdict first, then the train fits, the eval rates and the AUCs."""
    cfg, blocks = report.config, report.blocks
    verdict, chosen = blocks["verdict"], blocks["chosen"]
    out = [
        f"D177 abstention signals  ({report.generated_at})",
        f"  sets       {cfg['set_version']}  sha256 {cfg['sets_sha256'][:16]}...",
        f"  collection {cfg['collection']}",
        f"  rule       {cfg['rule']}",
        f"  instances  {cfg['instances']} ({cfg['resumed_rows']} resumed)",
        "",
        f"  VERDICT: {verdict['outcome']}",
        f"    {verdict['reason']}",
        "",
        f"  train, N={DECIDING_SIZE}, at false-answer <= 0.10:",
    ]
    for signal, fit in blocks["train_fits"].items():
        tau = "+inf (answers nothing)" if fit["answers_nothing"] else f"{fit['tau']:.4f}"
        out.append(
            f"    {signal:<8} tau {tau:<12} recall {fit['answer_recall']:.3f}  "
            f"false-answer {fit['false_answer']:.3f}"
        )
    out += [
        "",
        f"  eval, N={DECIDING_SIZE}: {chosen['signal']} at its train tau, papers resampled:",
    ]
    for name, interval in blocks["eval"].items():
        if name == "papers":
            continue
        point = "undefined" if interval["point"] is None else f"{interval['point']:.3f}"
        ci = interval["ci95"]
        band = "no interval" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        out.append(f"    {name:<36} {point}  {band}  ({interval['undefined_resamples']} undefined)")
    out += ["", "  AUC, pooled / qasper / withheld:"]
    for size, block in blocks["sizes"].items():
        tag = "[DECIDES]" if block["deciding"] else "(reported only)"
        out.append(f"    {size} {tag}")
        for split in ("train", "eval"):
            for signal, numbers in block.get(split, {}).items():
                cells = [
                    numbers["auc"],
                    numbers["auc/unanswerable-qasper"],
                    numbers["auc/unanswerable-withheld"],
                ]
                shown = " / ".join("-" if v is None else f"{v:.3f}" for v in cells)
                out.append(f"      {split:<5} {signal:<8} {shown}")
    differences = blocks["reproducibility"]["largest_abs_difference"]
    shown_differences = ", ".join(
        f"{signal} " + ("one side empty" if value is None else f"{value:.2e}")
        for signal, value in differences.items()
    )
    out += [
        "",
        f"  reproducibility: {blocks['reproducibility']['instances']} instances re-scored, "
        f"largest |difference| {shown_differences}",
    ]
    return "\n".join(out)


def _render_pdf_check(report: EvalReport) -> str:
    """Two builds side by side, with what each one's evidence did."""
    cfg = report.config
    out = [
        f"Scholiast PDF-subset check  ({report.generated_at})",
        f"  measurement {cfg['measurement']}",
        f"  sample      {cfg['sample']}",
        f"  varied      {cfg['varied']}",
        f"  basis       {cfg['comparison_basis']}  {cfg['comparable']}",
        f"  scope       {cfg['retrieval_scope']}",
        "",
    ]
    for label, block in report.blocks.items():
        out.append(f"=== {label}  -- {block['note']}")
        out.append(
            f"    {block['chunks']} chunks   {block['judgments']} judgments   "
            f"{block['queries_scored']} queries scored   evidence {block['evidence']}"
        )
        out.append(f"    mapping {block['mapping_diagnosis']}")
        for metric, values in sorted(block["metrics"].items()):
            low, high = values["ci95"]
            line = f"    {metric:<12}{values['point']:>7.3f}  [{low:.3f}, {high:.3f}]"
            delta = (block.get("deltas_vs_control") or {}).get("paper-scoped", {}).get(metric)
            if delta:
                flag = "within noise" if delta["within_noise"] else "SIGNIFICANT"
                dl, dh = delta["ci95"]
                line += f"   d {delta['delta']:+.3f} [{dl:+.3f}, {dh:+.3f}] {flag}"
            out.append(line)
        out.append("")
    return "\n".join(out)


def _render_contention(report: EvalReport) -> str:
    """One row per condition: what the user waits, and what ingestion paid for it."""
    cfg = report.config
    out = [
        f"Scholiast GPU contention  ({report.generated_at})",
        f"  measurement {cfg['measurement']}",
        f"  load        {cfg['ingestion_load']}",
        f"  queries     {cfg['query_workload']}",
        f"  budget      {cfg['budget']}",
        "",
        f"  {'condition':<34}{'rerank p50':>11}{'p95':>8}{'p99':>8}"
        f"{'stage p95':>11}{'ingest/s':>10}{'gate Q/I':>12}",
    ]
    for label, block in report.blocks.items():
        rerank, stage = block["rerank_ms"], block["stage_total_ms"]
        acquisitions = block["gate_acquisitions"]
        gate = (
            f"{acquisitions.get('QUERY', 0)}/{acquisitions.get('INGESTION', 0)}"
            if acquisitions
            else "-"
        )
        throughput = block["ingestion"]["chunks_per_second"]
        # "-", not 0.0: the idle row has no ingestion to measure, and a zero there
        # reads as "ingestion stalled" rather than "not applicable".
        ingest = f"{throughput:.1f}" if throughput is not None else "-"
        out.append(
            f"  {label:<34}{rerank.get('p50', 0):>11.0f}{rerank.get('p95', 0):>8.0f}"
            f"{rerank.get('p99', 0):>8.0f}{stage.get('p95', 0):>11.0f}{ingest:>10}{gate:>12}"
        )
        if block["diagnostic"]:
            out.append(f"  {'':<34}^ diagnostic — {block['note']}")
    out.append("")
    out.append(f"  not measured: {cfg['not_measured']}")
    return "\n".join(out)


async def _recompute_dense() -> Latency:
    """D152's committed scores, rewritten with the encoder production runs."""
    from scholiast.application.ports.ml import Precision
    from scholiast.infrastructure.ml.embedder import LocalEmbedder

    settings = get_settings()
    embedder = LocalEmbedder(
        settings.embedding_model,
        device=settings.embedding_device,
        precision=Precision(settings.embedding_precision),
        batch_size=settings.embedding_batch_size,
    )
    data, latency = await compute_dense_scores(load_searches(), embedder)
    DENSE_SCORES.write_text(
        json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"rewrote {DENSE_SCORES} with {data['embedder_version']}")
    return latency


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval",
        description="Retrieval evaluation over D-RET (doc 9 §9.3).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help=f"where the JSON artifact is archived (default: {DEFAULT_REPORT_DIR})",
    )
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="print the report without writing it",
    )
    parser.add_argument(
        "--compare-embedders",
        action="store_true",
        help=(
            "run every encoder in COMPARED_EMBEDDERS and report paired deltas "
            "against the M1 baseline (ML-4, REV-7). Off by default: the plain "
            "command must keep measuring what production runs."
        ),
    )
    parser.add_argument(
        "--compare-arms",
        action="store_true",
        help=(
            "run dense / lexical / RRF-hybrid / hybrid+rerank against the M1 baseline "
            "encoder and report paired deltas (FR-Q3, doc 6 sections 6.5-6.7). Off by default: "
            "the plain command measures the production arm alone, hybrid RRF (D174)."
        ),
    )
    parser.add_argument(
        "--compare-depths",
        action="store_true",
        help=(
            f"sweep the candidate depth over {' / '.join(str(d) for d in DEPTH_SWEEP)}, "
            f"reporting pre-rerank recall (the reranking ceiling) and post-rerank "
            f"metrics at each, against depth {INCUMBENT_DEPTH} (REV-4, D77). "
            f"{DIAGNOSTIC_DEPTH} is a diagnostic, not a production candidate."
        ),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "FR-E6's regression gate: run a seeded 10-paper subset on CPU and compare "
            f"every metric with {BASELINE_PATH}. Exits non-zero if any metric drops more "
            f"than {TOLERANCE} below baseline; a rise is reported and passes (D82)."
        ),
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help=(
            "with --smoke, rewrite the committed baseline from this run instead of "
            "checking against it. Deliberate by design: a gate that regenerates its own "
            "reference gates nothing."
        ),
    )
    parser.add_argument(
        "--pdf-check",
        action="store_true",
        help=(
            f"D68's validity check: re-parse {SAMPLE_SIZE} sampled papers from real arXiv "
            "PDFs through GROBID and the same chunker, and report the Recall@k delta "
            "against the published JSON build (D81). Fetches PDFs over the network."
        ),
    )
    parser.add_argument(
        "--contention",
        action="store_true",
        help=(
            "measure rerank latency under concurrent ingestion with the GPU priority "
            "gate on and off (doc 8 deliverable 6, §5.7, D79). Reads the collection "
            "the retrieval runs built and writes nothing to it."
        ),
    )
    parser.add_argument(
        "--compare-fusion",
        action="store_true",
        help=(
            "fit the weighted-fusion weight on the D-RET train split, then report "
            "weighted min-max fusion against RRF on the evaluation sets (ML-7, D76). "
            "The evaluation queries are never used for fitting."
        ),
    )
    parser.add_argument(
        "--workspace-gate",
        action="store_true",
        help=(
            f"D167's gate (D168): RRF hybrid vs dense with each query ranked against its "
            f"own paper plus papers drawn by hash, at {' / '.join(str(s) for s in SIZES)} "
            f"papers ({DECIDING_SIZE} decides), {DRAWS} draws per query. First reproduces "
            f"{GATE_REPRODUCES} exactly, or stops with no verdict. Read-only: checks "
            "that the collections are the archive's and complete, and never creates, "
            "indexes or writes either."
        ),
    )
    parser.add_argument(
        "--dedup",
        action="store_true",
        help=(
            "FR-D3: fit the deduplication thresholds on the D146 fixture's train topics, "
            "score the shipped rule on its held-out topics with Wilson intervals, list "
            "every error, and time the grouping (D148). Offline: reads committed data only."
        ),
    )
    parser.add_argument(
        "--ranking",
        action="store_true",
        help=(
            "FR-D4: fit the ranking weights on the D149 judgments' train searches, score "
            "the shipped ranker on the held-out ones against today's order and relevance "
            "alone, and time the ranking step (D151). Offline unless --recompute-dense."
        ),
    )
    parser.add_argument(
        "--recompute-dense",
        action="store_true",
        help=(
            "with --ranking, re-embed every search with the production encoder, rewrite "
            "the committed dense scores (D152) and time dense ranking on this device."
        ),
    )
    parser.add_argument(
        "--withheld-sheet",
        action="store_true",
        help=(
            "write D177's withheld-paper candidates, each with its scope's titles and "
            "abstracts, as the sheet the hand-check is done on. Scores nothing and "
            "archives nothing."
        ),
    )
    parser.add_argument(
        "--sheet-dir",
        type=Path,
        default=DEFAULT_SHEET_DIR,
        help=f"where --withheld-sheet writes (default: {DEFAULT_SHEET_DIR})",
    )
    parser.add_argument(
        "--build-sets",
        action="store_true",
        help=(
            "build D177's versioned answerability sets from the hand-check's verdicts "
            "(--verdicts), refusing any row that does not match the sheet. Scores nothing."
        ),
    )
    parser.add_argument(
        "--verdicts", type=Path, help="with --build-sets, the labeller's completed CSV"
    )
    parser.add_argument("--labeller", help="with --build-sets, who labelled the candidates")
    parser.add_argument(
        "--labelled-on", help="with --build-sets, the date the labels were made (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--abstention",
        action="store_true",
        help=(
            "D177: score the top dense, lexical and cross-encoder scores for every "
            "answerable and unanswerable question in its scopes, then fit tau by the "
            "pre-registered rule. No LLM. Read-only: never creates, indexes or writes a "
            "collection. Resumes from its progress file after an interruption."
        ),
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help=(
            "measure the dense-only arm, the D56 ablation baseline, archived as "
            "`retrieval` as the plain command's runs were before D174. The plain "
            "command measures production's hybrid RRF arm, archived as `retrieval-hybrid`."
        ),
    )
    args = parser.parse_args(argv)

    if args.dense:
        # Refused rather than ignored, as below: `--dense` names the plain run's
        # arm, and beside any other mode it would either do nothing or change
        # what that mode compares.
        others = [
            flag
            for flag, selected in (
                ("--compare-embedders", args.compare_embedders),
                ("--compare-arms", args.compare_arms),
                ("--compare-depths", args.compare_depths),
                ("--compare-fusion", args.compare_fusion),
                ("--smoke", args.smoke),
                ("--pdf-check", args.pdf_check),
                ("--contention", args.contention),
                ("--workspace-gate", args.workspace_gate),
                ("--dedup", args.dedup),
                ("--ranking", args.ranking),
                ("--withheld-sheet", args.withheld_sheet),
                ("--build-sets", args.build_sets),
                ("--abstention", args.abstention),
            )
            if selected
        ]
        if others:
            parser.error(f"--dense only means something on its own, not with {others[0]}")

    if args.build_sets:
        if not (args.verdicts and args.labeller and args.labelled_on):
            parser.error("--build-sets needs --verdicts, --labeller and --labelled-on")
        sheet_path = args.sheet_dir / "d177-withheld-candidates.json"
        sheet = json.loads(sheet_path.read_text(encoding="utf-8"))
        raw = args.verdicts.read_bytes()
        verdicts = read_verdicts(raw.decode("utf-8"), sheet)
        rows = load_snapshot()
        sets = build_sets(rows, build_dret(rows), verdicts, sheet_sha256=sheet["sheet_sha256"])
        record = write_sets(
            sets,
            labeller=args.labeller,
            labelled_on=args.labelled_on,
            verdicts_sha256=hashlib.sha256(raw).hexdigest(),
        )
        print(f"{record['set_version']}: {record['counts']}, sha256 {record['sha256']}")
        return 0

    if args.withheld_sheet:
        # D177's hand-check input, not a result: nothing is scored, so nothing is
        # archived, and the sheet's own hash is what a label file must quote.
        rows = load_snapshot()
        entries = candidate_sheet(rows, build_dret(rows))
        body = sheet_json(entries)
        digest = json.loads(body)["sheet_sha256"]
        args.sheet_dir.mkdir(parents=True, exist_ok=True)
        (args.sheet_dir / "d177-withheld-candidates.json").write_text(body + "\n", encoding="utf-8")
        (args.sheet_dir / "d177-withheld-candidates.md").write_text(
            sheet_markdown(entries, sheet_sha256=digest) + "\n", encoding="utf-8"
        )
        print(f"{len(entries)} candidates, sheet sha256 {digest}, written to {args.sheet_dir}")
        return 0

    if args.recompute_dense and not args.ranking:
        parser.error("--recompute-dense only means something with --ranking")

    if args.update_baseline and not args.smoke:
        # Refused rather than ignored: a flag that quietly does nothing is how
        # someone comes to believe they regenerated a baseline they did not.
        parser.error("--update-baseline only means something with --smoke")

    if args.smoke:
        # A gate, not a report: it prints a verdict and sets an exit code, and it
        # archives nothing to var/eval — the committed baseline IS its artifact.
        smoke_report, measured = asyncio.run(run_smoke())
        if args.update_baseline:
            save_baseline(smoke_report)
            print(f"baseline written to {BASELINE_PATH}")
            return 0
        comparison = compare(load_baseline()["metrics"], measured, tolerance=TOLERANCE)
        print(comparison.render())
        return 0 if comparison.passed else 1

    # Both kinds archive the same way: a timestamp and a JSON body.
    report: EvalReport | DedupReport | RankingReport
    if args.ranking:
        dense_latency = asyncio.run(_recompute_dense()) if args.recompute_dense else None
        report = run_ranking(dense_latency)
        print(_render_ranking(report))
    elif args.dedup:
        # Discovery, not retrieval: no index, no encoder, its own fixture.
        report = run_dedup()
        print(_render_dedup(report))
    elif args.pdf_check:
        # Two corpora rather than two retrieval configurations, so it has its own
        # runner and its own view.
        report = asyncio.run(run_pdf_check())
        print(_render_pdf_check(report))
    elif args.contention:
        # A different question with a different shape: conditions rather than
        # retrieval configurations, so it has its own runner and its own view.
        report = asyncio.run(run_contention())
        print(_render_contention(report))
    elif args.abstention:
        # A verdict under a rule fixed in advance (D177), so it leads with it.
        report = asyncio.run(run_abstention())
        print(_render_abstention(report))
    elif args.workspace_gate:
        # A verdict under a rule fixed in advance (D168), so it leads with the
        # verdict and the precondition rather than with a table.
        report = asyncio.run(run_workspace_gate())
        print(_render_gate(report))
    else:
        report = asyncio.run(
            _run(
                compare=args.compare_embedders,
                compare_arms=args.compare_arms,
                compare_depths=args.compare_depths,
                compare_fusion=args.compare_fusion,
                dense=args.dense,
            )
        )
        print(_render(report))

    if not args.no_archive:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.generated_at.replace(":", "").replace("-", "")
        # One kind per comparison, so §9.13's date-stamped trend line does not mix
        # a depth sweep into the arms series.
        kind = next(
            (
                name
                for name, selected in (
                    ("ranking", args.ranking),
                    ("dedup", args.dedup),
                    ("pdf-check", args.pdf_check),
                    ("contention", args.contention),
                    ("gate", args.workspace_gate),
                    ("abstention", args.abstention),
                    ("depths", args.compare_depths),
                    ("fusion", args.compare_fusion),
                    ("arms", args.compare_arms),
                    ("compare", args.compare_embedders),
                    # The dense baseline keeps the kind the plain run had before
                    # D174, so its series continues unbroken.
                    ("retrieval", args.dense),
                )
                if selected
            ),
            "retrieval-hybrid",
        )
        path = args.report_dir / f"{kind}-{stamp}.json"
        path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"archived to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin argv shim
    sys.exit(main())
