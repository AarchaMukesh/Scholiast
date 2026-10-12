"""D167's workspace-scale gate: does hybrid still not lose to dense in a workspace?

M2 measured hybrid RRF against dense-only with every query ranked against the
whole 281-paper corpus (D70), and hybrid won where it won by attacking **paper
confusion** — the right chunk lost to chunks from other papers. Production QA does
not rank against a corpus: it filters to one workspace's papers (D36), where there
are far fewer other papers to be confused with. Whether hybrid's advantage
survives that is the one limit of M2's result that bears directly on adopting it,
so D167 makes production wait on this measurement.

**Everything here was fixed by D168 before any of it was written**, so no choice
below can have been made after seeing a result:

- **Scope**: the query's own paper plus N - 1 others. N = 10 decides; 5 and 25 are
  reported beside it and decide nothing.
- **Draw**: uniform over every other corpus paper, by hash rather than by RNG, in
  `assign_split`'s style (see `scope_for`). Not "the most similar papers": any
  similarity is measured by one arm's model, which would bias the draw toward or
  against that arm.
- **Five draws per query**, each query's metric the mean over its draws, dense and
  hybrid always on the identical scope. The query stays the unit the bootstrap
  resamples.
- **The rule** (`decide`): hybrid is adopted unless the upper bound of its paired
  delta against dense is below zero on CORE recall@10, CORE nDCG@10 or FULL
  recall@10. A null is adopted and reported as one.
- **The precondition**: with every query's scope set to the whole corpus, this
  code must reproduce M2's archived arms numbers exactly, or there is no verdict.

**Retrieval is M2's, called unchanged.** Each (query, draw) is one call to
`run_retrieval` and one to `run_hybrid`, with a one-query list and that draw's
scope; neither function is modified, and each arm reads the collection M2's arm
read. Only the scope differs from M2's run, which is what makes the precondition a
real check: given M2's scope, the only way to miss M2's numbers is a defect in the
scope plumbing this module adds.

**It measures quality, not latency.** The per-call warm-ups and the scope churn
make the clocks meaningless here, so no latency leaves this module.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from scholiast.application.ports.vector import VectorIndex
from scholiast.eval.dret import DRet, Query
from scholiast.eval.harness import (
    EmbedderLike,
    LexicalLike,
    MetricSet,
    RunReport,
    cutoffs_for,
    run_hybrid,
    run_retrieval,
)
from scholiast.eval.metrics import NDCG_CUTOFF

#: The hash key's prefix (D168). Part of every draw: changing it redraws every
#: scope, so it is a constant with a test that pins it, not a parameter.
GATE_KEY = "d167-gate"

#: The one size that decides (D168), and the sizes reported. Every scope of a
#: smaller size is contained in the larger ones (see `scope_for`).
DECIDING_SIZE = 10
SIZES: tuple[int, ...] = (5, 10, 25)

#: Independent scopes per query; a query's metric is the mean over them.
DRAWS = 5

#: M2's candidate budget, so recall@10 and nDCG@10 mean what they meant there.
DEPTH = 50

#: The (query set, metric) pairs the rule reads, at `DECIDING_SIZE`. FULL has no
#: nDCG because its judgments are binary (D69).
DECIDING: tuple[tuple[str, str], ...] = (
    ("D-RET-CORE", "recall@10"),
    ("D-RET-CORE", f"ndcg@{NDCG_CUTOFF}"),
    ("D-RET-FULL", "recall@10"),
)

RULE = (
    f"at N = {DECIDING_SIZE}, hybrid is adopted unless the upper bound of its paired "
    "delta against dense is below zero on "
    + ", ".join(f"{name.removeprefix('D-RET-')} {metric}" for name, metric in DECIDING)
    + "; a null is adopted and reported as a null (D167, D168)"
)

#: The metrics whose per-draw delta is reported, to show how much the result
#: depends on which papers a query happened to be given.
SPREAD_METRICS: tuple[str, ...] = ("recall@10", f"ndcg@{NDCG_CUTOFF}")

#: A scope for a query at one draw.
Scope = Callable[[Query, int], Sequence[str]]


def scope_for(
    query_id: str, own_paper: str, papers: Iterable[str], *, size: int, draw: int
) -> tuple[str, ...]:
    """The papers a query ranks against at one draw: its own plus `size - 1` others.

    Every other paper is ranked by `sha256("d167-gate:{draw}:{query_id}:{paper}")`
    and the first `size - 1` are taken (D168). Three properties follow, and each is
    why it is a hash rather than a seeded shuffle:

    - **Independent of corpus order and size**: a paper's position depends only on
      its own id, so listing the corpus differently draws the same scope.
    - **Nested**: the key omits `size`, so the four papers drawn into the 5-paper
      scope are the first four of the nine drawn into the 10-paper one, and a
      difference between sizes is a difference of size, not of which papers
      happened to be drawn.
    - **Keyed by query, not paper**: two queries on one paper get different
      scopes, so the draws stay as independent as the queries are.

    Returned sorted: a scope is a set, and a fixed order keeps a call reproducible.
    """
    pool = set(papers)
    if own_paper not in pool:
        raise ValueError(f"query {query_id!r} belongs to {own_paper!r}, which is not in the corpus")
    others = sorted(pool - {own_paper})
    if not 1 <= size <= len(others) + 1:
        raise ValueError(f"a scope of {size} papers cannot be drawn from {len(others) + 1}")
    ranked = sorted(
        others,
        key=lambda paper: hashlib.sha256(f"{GATE_KEY}:{draw}:{query_id}:{paper}".encode()).digest(),
    )
    return tuple(sorted((own_paper, *ranked[: size - 1])))


def whole_corpus(dataset: DRet) -> Scope:
    """M2's scope for every query and draw: the whole corpus, as `--compare-arms` passed it."""
    everything = sorted(dataset.papers)
    return lambda _query, _draw: everything


def drawn(dataset: DRet, *, size: int) -> Scope:
    """D168's scope at one size: each query's own paper plus `size - 1` drawn by hash."""
    papers = sorted(dataset.papers)
    return lambda query, draw: scope_for(query.id, query.paper_id, papers, size=size, draw=draw)


@dataclass(slots=True)
class ScopedRuns:
    """Dense and hybrid over one query set, each query averaged over its draws."""

    dense: RunReport
    hybrid: RunReport
    #: Metric -> the mean of (hybrid - dense) across queries, one value per draw.
    draw_deltas: dict[str, list[float]] = field(default_factory=dict)


def _empty(label: str, depth: int) -> RunReport:
    return RunReport(label=label, metrics=MetricSet(recall={k: [] for k in cutoffs_for(depth)}))


def _mean(values: Sequence[float]) -> float:
    # Over one draw this is the value itself, exactly: the precondition compares
    # floats with `==`, and x / 1 is x.
    return sum(values) / len(values)


def _single(run: RunReport, query: Query) -> dict[str, float]:
    """One query's values out of a one-query run, refusing anything else."""
    if run.metrics.n != 1:
        raise ValueError(f"expected one scored query for {query.id!r}, got {run.metrics.n}")
    values = {f"recall@{k}": v[0] for k, v in run.metrics.recall.items()}
    values[f"ndcg@{NDCG_CUTOFF}"] = run.metrics.ndcg[0]
    values["mrr"] = run.metrics.rr[0]
    values["ceiling"] = run.ceiling_recall[0]
    values["own_paper"] = run.own_paper_in_top10[0]
    return values


def _append_mean(into: RunReport, draws: Sequence[dict[str, float]]) -> None:
    for k in into.metrics.recall:
        into.metrics.recall[k].append(_mean([d[f"recall@{k}"] for d in draws]))
    into.metrics.ndcg.append(_mean([d[f"ndcg@{NDCG_CUTOFF}"] for d in draws]))
    into.metrics.rr.append(_mean([d["mrr"] for d in draws]))
    into.ceiling_recall.append(_mean([d["ceiling"] for d in draws]))
    into.own_paper_in_top10.append(_mean([d["own_paper"] for d in draws]))


async def measure_scoped(
    dataset: DRet,
    queries: Sequence[Query],
    dense_index: VectorIndex,
    hybrid_index: Any,
    embedder: EmbedderLike,
    *,
    label: str,
    scope: Scope,
    draws: int,
    depth: int = DEPTH,
) -> ScopedRuns:
    """Dense and RRF hybrid for every query at every draw, on the identical scope.

    `dense_index` and `hybrid_index` are the two collections M2's arms read:
    dense-only for `run_retrieval`, the sparse-capable one (which serves both of
    hybrid's arms, D73) for `run_hybrid`.
    """
    if draws < 1:
        raise ValueError(f"a query needs at least one draw, got {draws}")
    lexical: LexicalLike = hybrid_index
    dense, hybrid = _empty(label, depth), _empty(label, depth)
    per_draw: dict[str, list[list[float]]] = {m: [[] for _ in range(draws)] for m in SPREAD_METRICS}

    for query in queries:
        dense_draws: list[dict[str, float]] = []
        hybrid_draws: list[dict[str, float]] = []
        for draw in range(draws):
            papers = scope(query, draw)
            dense_draws.append(
                _single(
                    await run_retrieval(
                        dataset,
                        [query],
                        dense_index,
                        embedder,
                        label=label,
                        paper_ids=papers,
                        depth=depth,
                    ),
                    query,
                )
            )
            hybrid_draws.append(
                _single(
                    await run_hybrid(
                        dataset,
                        [query],
                        hybrid_index,
                        lexical,
                        embedder,
                        label=label,
                        paper_ids=papers,
                        arms="hybrid",
                        depth=depth,
                    ),
                    query,
                )
            )
            for metric in SPREAD_METRICS:
                per_draw[metric][draw].append(hybrid_draws[-1][metric] - dense_draws[-1][metric])
        _append_mean(dense, dense_draws)
        _append_mean(hybrid, hybrid_draws)

    return ScopedRuns(
        dense=dense,
        hybrid=hybrid,
        draw_deltas={
            metric: [_mean(values) for values in draws_of] if queries else []
            for metric, draws_of in per_draw.items()
        },
    )


# --- the precondition -------------------------------------------------------

#: The keys of a summary block the precondition compares. Latency is excluded by
#: name: it is not reproducible (M2 log §3) and the gate does not measure it.
COMPARED_KEYS: tuple[str, ...] = ("n", "metrics", "diagnostics")


def compare_with_archive(
    measured: Mapping[str, Any], archived: Mapping[str, Any], *, dense: str, hybrid: str
) -> list[str]:
    """Every place the gate's whole-corpus run differs from M2's arms artifact.

    `measured` is `{"dense": {set: summary}, "hybrid": {set: summary},
    "deltas": {set: deltas}}`; `dense` and `hybrid` name the archived blocks.
    Floats are compared with `==`: M2 reproduced itself exactly, 124 of 124, so
    anything less here is a difference, not a tolerance. An empty list is the
    precondition met.

    **Both directions, and never vacuously.** D168 reproduces dense and hybrid on
    CORE and FULL, so a query set the archive holds but this run did not measure
    is a difference, as is a set measured but not archived; and a comparison
    with nothing on either side is a difference too. Checking only what was
    measured would let an empty run "reproduce" an empty archive.
    """
    blocks = archived.get("blocks", {})
    differences: list[str] = []
    for side in ("dense", "hybrid", "deltas"):
        if not measured.get(side):
            differences.append(f"{side}: the whole-corpus run measured nothing to compare")
    for side, block_label in (("dense", dense), ("hybrid", hybrid)):
        if block_label not in blocks:
            differences.append(f"{block_label}: not in the archived artifact")
            continue
        archived_sets = blocks[block_label].get("blocks", {})
        for name in sorted(set(archived_sets) - set(measured[side])):
            differences.append(f"{block_label}/{name}: archived but not measured")
        for name, summary in measured[side].items():
            stored = blocks[block_label].get("blocks", {}).get(name)
            if stored is None:
                differences.append(f"{block_label}/{name}: not in the archived artifact")
                continue
            for key in COMPARED_KEYS:
                differences.extend(
                    _differences(f"{block_label}/{name}/{key}", summary.get(key), stored.get(key))
                )
    if hybrid in blocks:
        stored_deltas = blocks[hybrid].get("deltas_vs_control", {})
        for name in sorted(set(stored_deltas) - set(measured["deltas"])):
            differences.append(f"{hybrid}/deltas_vs_control/{name}: archived but not measured")
        for name, deltas in measured["deltas"].items():
            differences.extend(
                _differences(f"{hybrid}/deltas_vs_control/{name}", deltas, stored_deltas.get(name))
            )
    return differences


def _differences(path: str, ours: Any, theirs: Any) -> list[str]:
    if isinstance(ours, Mapping) and isinstance(theirs, Mapping):
        out: list[str] = []
        for key in sorted(set(ours) | set(theirs), key=str):
            if key not in theirs:
                out.append(f"{path}/{key}: not in the archived artifact")
            elif key not in ours:
                out.append(f"{path}/{key}: archived but not measured")
            else:
                out.extend(_differences(f"{path}/{key}", ours[key], theirs[key]))
        return out
    if isinstance(ours, list | tuple) and isinstance(theirs, list | tuple):
        if len(ours) != len(theirs):
            return [f"{path}: {len(ours)} values against {len(theirs)} archived"]
        out = []
        for i, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            out.extend(_differences(f"{path}[{i}]", a, b))
        return out
    # `type(...) is bool` keeps True from equalling 1 by accident.
    if ours != theirs or (type(ours) is bool) != (type(theirs) is bool):
        return [f"{path}: measured {ours!r}, archived {theirs!r}"]
    return []


# --- the rule ---------------------------------------------------------------


def decide(
    reproduced: bool, deltas: Mapping[str, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    """D167's rule over the deltas at `DECIDING_SIZE`. Only `DECIDING` is read.

    `deltas` is `{query set: {metric: {"delta", "ci95": [low, high], ...}}}`, as
    the arms comparison reports them. "Below zero" is strict: an upper bound of
    exactly zero is an interval that touches zero, and touching it is not
    excluding it.
    """
    if not reproduced:
        return {
            "outcome": "no verdict",
            "reason": "the whole-corpus run did not reproduce M2's arms numbers exactly (D168)",
            "rule": RULE,
        }
    worse: list[str] = []
    better: list[str] = []
    null: list[str] = []
    # The intervals the rule actually read, so the verdict carries its own
    # evidence and a reader need not find the deciding block to check it.
    read: dict[str, Any] = {}
    for name, metric in DECIDING:
        try:
            low, high = deltas[name][metric]["ci95"]
        except KeyError:
            raise ValueError(f"the rule reads {name} {metric}, which was not measured") from None
        label = f"{name} {metric}"
        read[label] = {"delta": deltas[name][metric].get("delta"), "ci95": [low, high]}
        if high < 0.0:
            worse.append(label)
        elif low > 0.0:
            better.append(label)
        else:
            null.append(label)
    return {
        "outcome": "stay dense-only" if worse else "adopt hybrid",
        "significantly_worse": worse,
        "significantly_better": better,
        "within_noise": null,
        "read": read,
        "rule": RULE,
    }
