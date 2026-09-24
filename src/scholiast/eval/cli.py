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
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholiast.config.settings import get_settings
from scholiast.eval.dret import DRet, Query, build_dret, verify_core, verify_split
from scholiast.eval.fusion import FusionFit, fit_dense_weight, training_queries
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
from scholiast.eval.snapshot import load_snapshot

#: Where a run lands. Date-stamped rather than overwritten (§9.13): the trend line
#: across milestones is itself a deliverable, and it cannot exist if each run
#: destroys the last.
DEFAULT_REPORT_DIR = Path("var/eval")


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
) -> tuple[EmbedderLike, Any, str]:
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
    return [Cell(base.label, base, is_m1_baseline=True)]


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
            "adoption waits on the phase 7 ablation, so production stays dense-only."
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
        "--compare-fusion",
        action="store_true",
        help=(
            "fit the weighted-fusion weight on the D-RET train split, then report "
            "weighted min-max fusion against RRF on the evaluation sets (ML-7, D76). "
            "The evaluation queries are never used for fitting."
        ),
    )
    args = parser.parse_args(argv)

    report = asyncio.run(
        _run(
            compare=args.compare_embedders,
            compare_arms=args.compare_arms,
            compare_depths=args.compare_depths,
            compare_fusion=args.compare_fusion,
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
                    ("depths", args.compare_depths),
                    ("fusion", args.compare_fusion),
                    ("arms", args.compare_arms),
                    ("compare", args.compare_embedders),
                )
                if selected
            ),
            "retrieval",
        )
        path = args.report_dir / f"{kind}-{stamp}.json"
        path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"archived to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin argv shim
    sys.exit(main())
