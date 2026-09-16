"""`uv run eval` — the one command that reproduces the report (doc 9 §9.12, FR-E4).

§9.12's requirement is literal: *"One command: `make eval` (or `uv run eval`)
reproduces the published report."* This is that command.

**What it emits, and why both.** A JSON artifact and a terminal table. The JSON is
the archive — §9.13 keeps every milestone's run date-stamped so the trend line
across milestones becomes an artifact in itself, and that needs a machine-readable
file rather than scrollback. The table is what a human reads while working.

**Every run records what produced it.** Dataset versions, embedder version,
collection name, retrieval scope, query counts and the grade scale go into the
report next to the numbers. §9.12 calls this a "full config snapshot"; the reason it
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
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholiast.config.settings import get_settings
from scholiast.eval.dret import DRet, Query, build_dret, verify_core, verify_split
from scholiast.eval.harness import (
    EmbedderLike,
    IndexReport,
    RunReport,
    build_index,
    collection_name,
    run_retrieval,
)
from scholiast.eval.metrics import (
    NDCG_CUTOFF,
    RECALL_CUTOFFS,
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
    for k in RECALL_CUTOFFS:
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


def _render(report: EvalReport) -> str:
    """The terminal view: one section per encoder, deltas under the challenger."""
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
    for label, arm in report.blocks.items():
        marker = "  [M1 DENSE BASELINE]" if arm.get("is_m1_baseline") else ""
        res = arm["resources"]
        out.append(f"=== {label}{marker}")
        out.append(f"    {arm['model']}  {res['dimensions']}d  ->  {arm['collection']}")
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
                delta = (arm.get("deltas_vs_baseline") or {}).get(name, {}).get(metric)
                if delta:
                    flag = "within noise" if delta["within_noise"] else "SIGNIFICANT"
                    dl, dh = delta["ci95"]
                    line += f"   d {delta['delta']:+.3f} [{dl:+.3f}, {dh:+.3f}] {flag}"
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
                f"   (embed p50 {lat['embed_p50_ms']}ms  search p50 {lat['search_p50_ms']}ms)"
            )
        out.append("")
    return "\n".join(out)


def _build_dependencies(dataset: DRet, spec: EmbedderSpec) -> tuple[EmbedderLike, Any, str]:
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
    collection = collection_name(dataset, embedder.version)
    index = QdrantVectorIndex(
        settings.qdrant_url,
        collection=collection,
        dimensions=spec.dimensions,
    )
    return embedder, index, collection


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
    dataset: DRet, spec: EmbedderSpec, scope: list[str]
) -> tuple[dict[str, Any], dict[str, RunReport], dict[str, Any]]:
    """Run one encoder over every query set. Returns blocks, raw runs, config."""
    embedder, index, collection = _build_dependencies(dataset, spec)
    index_report = await build_index(dataset, index, embedder, collection=collection)

    blocks: dict[str, Any] = {}
    runs: dict[str, RunReport] = {}
    for name, attribute, graded, note in QUERY_SETS:
        queries: list[Query] = getattr(dataset, attribute)
        run = await run_retrieval(dataset, queries, index, embedder, label=name, paper_ids=scope)
        runs[name] = run
        blocks[name] = {**_summarise(run, graded=graded), "note": note}
    return (
        blocks,
        runs,
        {
            "embedder_version": embedder.version,
            "collection": collection,
            "resources": _resources(spec, index_report, len(dataset.chunks)),
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
    pairs: list[tuple[str, list[float], list[float]]] = [
        (f"recall@{k}", treatment.metrics.recall[k], baseline.metrics.recall[k])
        for k in RECALL_CUTOFFS
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
    }


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(len(ordered) * q))], 2)


async def _run(*, compare: bool) -> EvalReport:
    dataset = build_dret(load_snapshot())
    verify_split(dataset)
    verify_core(dataset)
    # D70: the whole corpus, every query. Stated here, once, where it is passed.
    scope = sorted(dataset.papers)

    specs = COMPARED_EMBEDDERS if compare else tuple(s for s in COMPARED_EMBEDDERS if s.baseline)
    arms: dict[str, Any] = {}
    raw: dict[str, dict[str, RunReport]] = {}
    for spec in specs:
        blocks, runs, arm_config = await _measure(dataset, spec, scope)
        raw[spec.label] = runs
        arms[spec.label] = {
            **arm_config,
            "model": spec.model,
            "is_m1_baseline": spec.baseline,
            "blocks": blocks,
            "latency": {name: _latency(run) for name, run in runs.items()},
        }

    if compare:
        control = next(s for s in COMPARED_EMBEDDERS if s.baseline)
        for spec in specs:
            if spec.baseline:
                continue
            arms[spec.label]["deltas_vs_baseline"] = {
                name: _deltas(raw[spec.label][name], raw[control.label][name], graded=graded)
                for name, _attribute, graded, _note in QUERY_SETS
            }

    return EvalReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        config={
            "configuration": "dense-only (D56 ablation baseline)",
            "dataset_versions": dataset.versions,
            "retrieval_scope": f"whole corpus, {len(scope)} papers, "
            f"{len(dataset.chunks)} chunks (D70)",
            "grade_scale": dataset.grade_scale,
            "corpus_chunks": len(dataset.chunks),
            "comparison": "ML-4 embedding-model comparison" if compare else "single arm",
            # Scope of the latency figures, stated in the artifact rather than
            # only in doc 9 §9.8. A reader meets these numbers here, and a table
            # that says "latency" without saying WHICH stages invites the
            # assumption that it covers the pipeline. Rows for unbuilt stages
            # would be fabrication, so they are absent rather than zero.
            "latency_measured": (
                "retrieve only: query embed + dense search, idle system, "
                "CUDA init excluded by an untimed warm-up. "
                "lexical+fusion (phase 3), rerank (phase 4), context/TTFT/complete "
                "(generation path) and the two non-idle conditions are NOT measured yet"
            ),
        },
        blocks=arms,
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
    args = parser.parse_args(argv)

    report = asyncio.run(_run(compare=args.compare_embedders))
    print(_render(report))

    if not args.no_archive:
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.generated_at.replace(":", "").replace("-", "")
        kind = "compare" if args.compare_embedders else "retrieval"
        path = args.report_dir / f"{kind}-{stamp}.json"
        path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"archived to {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin argv shim
    sys.exit(main())
