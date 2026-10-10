"""FR-D4's ablation: fit ranking weights on train topics, score them on held-out ones (D151).

**The searches are rebuilt, not replayed.** Each of A3's 20 queries (10 topics x 2
phrasings) is rebuilt from its recorded responses exactly as production would build
it: both sources' records, interleaved (D122), merged into works by the matcher the
judgments were made with (D148). Each work then takes the grade D149 gave the judged
work it belongs to. Every work a search returns is judged, so no ranking can score
by burying an unjudged paper (`test_discovery_judgments`).

**The fit sees only the 10 training searches** (odd topics). Over a 0.1 simplex of
the three weights (66 points) for each relevance signal, it keeps the highest mean
nDCG@10, ties to the larger relevance weight, then the smaller recency weight, then
lexical (D151). The held-out searches are scored once, with paired bootstrap deltas
against today's order (D147) and against relevance alone.

**Dense scores come from a committed file** (D152): cosine between the query and
"title. abstract", computed once with the production embedder and stamped with its
version, so the fit runs offline. `--recompute-dense` rewrites it, and loading
refuses a file stamped by another embedder.

**What this does not establish.** 10 held-out searches give wide intervals. The pool
is two-thirds grade 2 (D149), so the room any ranker has is mostly 2s above 1s. The
judge also wrote the ranker. Latency here is the ranking step alone; phase 10's live
run measures the whole request.
"""

from __future__ import annotations

import gzip
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholiast.application.ports.ml import Embedder, TextRole
from scholiast.domain import dedup, ranking
from scholiast.domain.dedup import DiscoveredWork, MatchRule
from scholiast.domain.discovery import PaperCandidate, interleave
from scholiast.domain.ranking import RankingWeights, RelevanceSignal
from scholiast.eval.metrics import Interval, bootstrap_interval, ndcg_at_k, paired_bootstrap_delta
from scholiast.infrastructure.discovery.arxiv import parse_feed
from scholiast.infrastructure.discovery.openalex import parse_works

ROOT = Path(__file__).resolve().parents[3]
POOLS = ROOT / "data" / "eval" / "discovery-pools-a3.json.gz"
JUDGMENTS = ROOT / "data" / "eval" / "discovery-judgments.json"
DENSE_SCORES = ROOT / "data" / "eval" / "discovery-dense-scores.json"

#: The matcher the judgments were made with (D148), named rather than read from
#: SHIPPED_RULE, so a later matcher change cannot silently redefine the searches.
JUDGED_WITH = MatchRule(title_threshold=0.85, year_tolerance=2)
#: A3 was recorded on 2026-10-08; ages are counted from that year so the numbers
#: reproduce whatever year the eval is rerun in.
RANKING_YEAR = 2026
CUTOFF = 10
#: D151's grid: three weights in steps of 0.1, summing to one.
GRID: tuple[RankingWeights, ...] = tuple(
    RankingWeights(r / 10, c / 10, (10 - r - c) / 10) for r in range(11) for c in range(11 - r)
)
SIGNALS: tuple[RelevanceSignal, ...] = (RelevanceSignal.LEXICAL, RelevanceSignal.DENSE)
LATENCY_REPEATS = 30


def record_key(record: PaperCandidate) -> str:
    return f"{record.source}:{record.source_id}"


@dataclass(frozen=True)
class Search:
    query_id: str
    topic: str
    split: str
    text: str
    works: tuple[DiscoveredWork, ...]
    #: The judged work each of `works` belongs to, in the same order.
    judged: tuple[str, ...]
    #: Judged work -> grade, for the works this search returned.
    grades: Mapping[str, int]


def load_searches(pools: Path = POOLS, judgments: Path = JUDGMENTS) -> tuple[Search, ...]:
    data = json.loads(judgments.read_text(encoding="utf-8"))
    owner = {
        (t["topic"], f"{m[0]}:{m[1]}"): w["work"]
        for t in data["topics"]
        for w in t["works"]
        for m in w["members"]
    }
    grade = {
        w["work"]: w.get("owner_grade", w["grade"]) for t in data["topics"] for w in t["works"]
    }
    split = {t["topic"]: t["split"] for t in data["topics"]}
    text = {
        (t["topic"], ph): t[ph] for t in data["topics"] for ph in ("keyword", "natural_language")
    }

    pages: dict[tuple[str, str], dict[str, tuple[PaperCandidate, ...]]] = {}
    for entry in json.loads(gzip.decompress(pools.read_bytes()))["entries"]:
        if entry["status"] != 200:
            continue
        body = entry["body"]
        page = parse_works(json.loads(body)) if entry["source"] == "openalex" else parse_feed(body)
        pages.setdefault((entry["topic"], entry["phrasing"]), {})[entry["source"]] = page.candidates

    searches = []
    for (topic, phrasing), by_source in sorted(pages.items()):
        records = interleave([by_source.get("openalex", ()), by_source.get("arxiv", ())])
        works = dedup.works(records, JUDGED_WITH)
        judged = []
        for w in works:
            owners = {owner[(topic, record_key(m))] for m in w.members}
            if len(owners) != 1:
                raise ValueError(f"{topic}-{phrasing}: a work spans judged works {sorted(owners)}")
            judged.append(owners.pop())
        searches.append(
            Search(
                query_id=f"{topic}-{phrasing}",
                topic=topic,
                split=split[topic],
                text=text[(topic, phrasing)],
                works=works,
                judged=tuple(judged),
                grades={j: grade[j] for j in judged},
            )
        )
    return tuple(searches)


def load_dense(path: Path = DENSE_SCORES, *, expect_version: str | None = None) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if expect_version is not None and data["embedder_version"] != expect_version:
        raise ValueError(
            f"{path.name} was computed with {data['embedder_version']}, not {expect_version}; "
            "rerun with --recompute-dense rather than mixing embedders (D152)"
        )
    return data


def relevance(search: Search, signal: RelevanceSignal, dense: Mapping[str, Any]) -> list[float]:
    if signal is RelevanceSignal.LEXICAL:
        return ranking.lexical_relevance(
            search.text, [ranking.relevance_text(w) for w in search.works]
        )
    scores = dense["scores"][search.query_id]
    return [float(scores[record_key(w.add)]) for w in search.works]


def _ndcg(order: Sequence[str], grades: Mapping[str, int]) -> float:
    # A judged work two of a search's works both belong to counts once, where it
    # first appears: it is one paper the user sees twice, not two relevant papers.
    seen: list[str] = []
    for j in order:
        if j not in seen:
            seen.append(j)
    relevant = {j: g for j, g in grades.items() if g > 0}
    return ndcg_at_k(seen, relevant, CUTOFF)


def baseline_ndcg(search: Search) -> float:
    """Today's order (D147): works where their earliest record sat."""
    return _ndcg(search.judged, search.grades)


def ranked_ndcg(
    search: Search, signal: RelevanceSignal, weights: RankingWeights, dense: Mapping[str, Any]
) -> float:
    ranked = ranking.rank(
        search.works, relevance(search, signal, dense), weights, ranking_year=RANKING_YEAR
    )
    return _ndcg([search.judged[r.position] for r in ranked], search.grades)


@dataclass(frozen=True)
class GridRow:
    signal: str
    relevance: float
    citation: float
    recency: float
    mean_ndcg: float


def fit(
    searches: Sequence[Search], dense: Mapping[str, Any]
) -> tuple[RelevanceSignal, RankingWeights, tuple[GridRow, ...]]:
    """D151's fit. Call it with training searches only; it cannot tell them apart."""
    rows = []
    for signal in SIGNALS:
        for weights in GRID:
            scores = [ranked_ndcg(s, signal, weights, dense) for s in searches]
            rows.append(
                GridRow(
                    signal.value,
                    weights.relevance,
                    weights.citation,
                    weights.recency,
                    round(sum(scores) / len(scores), 12),
                )
            )
    best = max(
        rows,
        key=lambda r: (r.mean_ndcg, r.relevance, -r.recency, r.signal == RelevanceSignal.LEXICAL),
    )
    return (
        RelevanceSignal(best.signal),
        RankingWeights(best.relevance, best.citation, best.recency),
        tuple(rows),
    )


@dataclass(frozen=True)
class Arm:
    name: str
    per_query: tuple[float, ...]

    @property
    def mean(self) -> Interval:
        return bootstrap_interval(self.per_query)


def held_out_arms(
    searches: Sequence[Search],
    signal: RelevanceSignal,
    weights: RankingWeights,
    dense: Mapping[str, Any],
) -> dict[str, Arm]:
    only = RankingWeights(1.0, 0.0, 0.0)
    return {
        "today's order": Arm("today's order", tuple(baseline_ndcg(s) for s in searches)),
        "lexical only": Arm(
            "lexical only",
            tuple(ranked_ndcg(s, RelevanceSignal.LEXICAL, only, dense) for s in searches),
        ),
        "dense only": Arm(
            "dense only",
            tuple(ranked_ndcg(s, RelevanceSignal.DENSE, only, dense) for s in searches),
        ),
        "fitted": Arm("fitted", tuple(ranked_ndcg(s, signal, weights, dense) for s in searches)),
    }


@dataclass(frozen=True)
class Latency:
    signal: str
    works: int
    p50_ms: float
    p95_ms: float


def time_lexical(searches: Sequence[Search], weights: RankingWeights) -> Latency:
    """Lexical relevance plus ranking, per search, on this machine's CPU."""
    samples = []
    for s in searches:
        for _ in range(LATENCY_REPEATS):
            started = time.perf_counter()
            scores = ranking.lexical_relevance(s.text, [ranking.relevance_text(w) for w in s.works])
            ranking.rank(s.works, scores, weights, ranking_year=RANKING_YEAR)
            samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return Latency(
        "lexical",
        max(len(s.works) for s in searches),
        samples[len(samples) // 2],
        samples[min(len(samples) - 1, int(0.95 * len(samples)))],
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


async def dense_relevance(
    embedder: Embedder, query: str, works: Sequence[DiscoveredWork]
) -> list[float]:
    """Cosine between the query and each work's "title. abstract" (D151): what production runs."""
    [q] = await embedder.embed([query], role=TextRole.QUERY)
    passages = await embedder.embed(
        [ranking.relevance_text(w) for w in works], role=TextRole.PASSAGE
    )
    return [_cosine(q, p) for p in passages]


async def compute_dense_scores(
    searches: Sequence[Search], embedder: Embedder
) -> tuple[dict[str, Any], Latency]:
    """D152's committed scores, and dense ranking's latency on this machine's device.

    The first search warms the model, so it is scored but not timed: whichever
    condition touches a model first pays its load (CLAUDE.md's warm-up rule).
    """
    scores: dict[str, dict[str, float]] = {}
    samples = []
    for i, s in enumerate(searches):
        started = time.perf_counter()
        values = await dense_relevance(embedder, s.text, s.works)
        ranking.rank(s.works, values, ranking.SHIPPED_WEIGHTS, ranking_year=RANKING_YEAR)
        if i:
            samples.append((time.perf_counter() - started) * 1000.0)
        scores[s.query_id] = {
            record_key(w.add): round(v, 6) for w, v in zip(s.works, values, strict=True)
        }
    samples.sort()
    data = {
        "dataset": "scholiast-m4-discovery-dense-scores",
        "decision": "D152",
        "embedder_version": embedder.version,
        "text": "query as TextRole.QUERY; each work as TextRole.PASSAGE 'title. abstract' "
        "(title alone "
        "when there is no abstract); cosine; rounded to 6 places",
        "keys": "query id -> the work's add record (source:source_id) -> cosine",
        "scores": scores,
    }
    latency = Latency(
        "dense",
        max(len(s.works) for s in searches),
        samples[len(samples) // 2],
        samples[min(len(samples) - 1, int(0.95 * len(samples)))],
    )
    return data, latency


@dataclass(frozen=True)
class RankingReport:
    generated_at: str
    ranker_version: str
    dense_embedder: str
    fitted_signal: str
    fitted: RankingWeights
    shipped_signal: str
    shipped: RankingWeights
    grid: tuple[GridRow, ...]
    train_arms: dict[str, Arm]
    held_out_arms: dict[str, Arm]
    deltas: dict[str, Interval]
    latency: tuple[Latency, ...]
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        def interval(i: Interval) -> dict[str, float]:
            return {"point": i.point, "low": i.low, "high": i.high, "n": i.n}

        def arms(a: Mapping[str, Arm]) -> dict[str, Any]:
            return {
                k: {"mean": interval(v.mean), "per_query": list(v.per_query)} for k, v in a.items()
            }

        return json.dumps(
            {
                "kind": "ranking",
                "generated_at": self.generated_at,
                "ranker_version": self.ranker_version,
                "dense_embedder": self.dense_embedder,
                "fitted": {"signal": self.fitted_signal, **asdict(self.fitted)},
                "shipped": {"signal": self.shipped_signal, **asdict(self.shipped)},
                "grid": [asdict(r) for r in self.grid],
                "train": arms(self.train_arms),
                "held_out": arms(self.held_out_arms),
                "held_out_deltas": {k: interval(v) for k, v in self.deltas.items()},
                "latency": [asdict(x) for x in self.latency],
                "config": self.config,
            },
            indent=2,
        )


def run_ranking(dense_latency: Latency | None = None) -> RankingReport:
    searches = load_searches()
    dense = load_dense()
    train = [s for s in searches if s.split == "train"]
    held = [s for s in searches if s.split == "held_out"]
    signal, weights, grid = fit(train, dense)
    shipped_arms = held_out_arms(held, ranking.SHIPPED_SIGNAL, ranking.SHIPPED_WEIGHTS, dense)
    fitted_relevance_only = (
        "lexical only" if ranking.SHIPPED_SIGNAL is RelevanceSignal.LEXICAL else "dense only"
    )
    deltas = {
        "fitted - today's order": paired_bootstrap_delta(
            shipped_arms["fitted"].per_query, shipped_arms["today's order"].per_query
        ),
        f"fitted - {fitted_relevance_only}": paired_bootstrap_delta(
            shipped_arms["fitted"].per_query, shipped_arms[fitted_relevance_only].per_query
        ),
    }
    latency = [time_lexical(searches, ranking.SHIPPED_WEIGHTS)]
    if dense_latency is not None:
        latency.append(dense_latency)
    return RankingReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ranker_version=ranking.RANKER_VERSION,
        dense_embedder=str(dense["embedder_version"]),
        fitted_signal=signal.value,
        fitted=weights,
        shipped_signal=ranking.SHIPPED_SIGNAL.value,
        shipped=ranking.SHIPPED_WEIGHTS,
        grid=grid,
        train_arms=held_out_arms(train, signal, weights, dense),
        held_out_arms=shipped_arms,
        deltas=deltas,
        latency=tuple(latency),
        config={
            "searches": {"train": len(train), "held_out": len(held)},
            "cutoff": CUTOFF,
            "ranking_year": RANKING_YEAR,
            "objective": "max mean train nDCG@10; ties to larger relevance weight, smaller recency "
            "weight, then lexical",
            "intervals": "percentile bootstrap over searches, 1000 iterations, seeded",
        },
    )


def render(report: RankingReport) -> str:
    out = [
        f"Ranking (FR-D4, D151) - {report.ranker_version}; "
        f"dense scores from {report.dense_embedder}",
        f"  fitted on train: {report.fitted_signal} {report.fitted}",
        f"  shipped:         {report.shipped_signal} {report.shipped}"
        + (
            ""
            if (report.fitted_signal, report.fitted) == (report.shipped_signal, report.shipped)
            else "   <-- SHIPPED RANKER IS NOT THE FIT"
        ),
        "  nDCG@10, mean [95% bootstrap]:      train            held out",
    ]
    for name in report.held_out_arms:
        out.append(
            f"    {name:<16} {report.train_arms[name].mean!s:>26}"
            f"   {report.held_out_arms[name].mean!s:>26}"
        )
    out.append("  held-out paired deltas:")
    for name, delta in report.deltas.items():
        out.append(f"    {name:<30} {delta}")
    for lat in report.latency:
        out.append(
            f"  {lat.signal} ranking per search ({lat.works} works): "
            f"p50 {lat.p50_ms:.1f} ms, p95 {lat.p95_ms:.1f} ms"
        )
    out.append("  not established: 10 held-out searches; pool two-thirds grade 2 (D149)")
    return "\n".join(out)
