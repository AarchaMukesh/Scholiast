"""FR-D3's measurement: fit the matcher on train topics, score it on held-out ones (D146, D148).

**The fixture** (`data/eval/dedup-fixture.json`) is 379 candidate pairs from the
spike's frozen pools, labelled before any matcher existed and frozen by hash. A
pair names its two records by source and source id; the records themselves are
re-parsed here from the pools through the production adapters, so the fixture
cannot drift from what the adapters actually produce.

**What is scored is the grouping, not the pair rule.** Every topic's 100 records go
through `dedup.group` exactly as one search's would, and a pair counts as merged if
its two records land in one work. Transitive merges are therefore scored, including
the ones no single pair rule made.

**The fit sees only odd-numbered topics.** T and Y are chosen there (D148: the highest
recall with no false merge, ties to the larger T then the smaller Y), and the
held-out topics are scored once, with the fitted rule. A test holds
`dedup.SHIPPED_RULE` to this fit, so the shipped thresholds cannot drift from the
measurement without a test failing.

**What this does not establish.** Recall is recall within the candidate set: a
version retitled past 0.55 overlap with no shared identifier is invisible to the
fixture and the matcher alike (D146). Pairs share records, so the Wilson intervals
are narrower than the truth. The labeller saw the held-out pairs while labelling, so
only the thresholds, not the rule's form, are fitted blind to them (D148).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scholiast.domain import dedup
from scholiast.domain.dedup import MatchRule, name_words, title_overlap
from scholiast.domain.discovery import PaperCandidate, interleave, normalise_title, paper_identity
from scholiast.eval.metrics import Interval, wilson_interval
from scholiast.infrastructure.discovery.arxiv import parse_feed
from scholiast.infrastructure.discovery.openalex import parse_works

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "data" / "eval" / "dedup-fixture.json"
MANIFEST = ROOT / "data" / "eval" / "dedup-fixture.manifest.json"
POOLS = ROOT / "data" / "eval" / "discovery-pools.json.gz"

#: D148's grid. Thresholds start at the fixture's own candidate floor (0.55): below
#: it the fixture cannot see a merge, so it could not score one.
THRESHOLDS: tuple[float, ...] = tuple(round(0.55 + 0.05 * i, 2) for i in range(10))
YEAR_TOLERANCES: tuple[int, ...] = (0, 1, 2, 3)
#: Production's interleave order: OpenAlex is the primary (D122).
SOURCE_ORDER: tuple[str, ...] = ("openalex", "arxiv")
#: How often each topic's grouping is timed for the latency figure.
LATENCY_REPEATS = 50

RecordKey = tuple[str, str]


@dataclass(frozen=True, slots=True)
class LabelledPair:
    pair: str
    topic: str
    split: str
    a: RecordKey
    b: RecordKey
    #: The scored label: the owner's where the owner gave one (D146).
    same: bool


def load_pairs(path: Path = FIXTURE) -> tuple[LabelledPair, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    pairs = []
    for p in data["pairs"]:
        label = p.get("owner_label", p["label"])
        if label not in ("same", "different"):
            raise ValueError(f"{p['pair']} is still {label!r}; resolve it before scoring")
        pairs.append(
            LabelledPair(
                pair=p["pair"],
                topic=p["topic"],
                split=p["split"],
                a=(p["a"]["source"], p["a"]["source_id"]),
                b=(p["b"]["source"], p["b"]["source_id"]),
                same=label == "same",
            )
        )
    return tuple(pairs)


def topic_records(pools: Path = POOLS) -> dict[str, tuple[PaperCandidate, ...]]:
    """Each topic's P1 keyword search, both sources, interleaved as production does."""
    pages: dict[str, dict[str, tuple[PaperCandidate, ...]]] = {}
    entries = json.loads(gzip.decompress(pools.read_bytes()))["entries"]
    for entry in entries:
        if (
            entry.get("pass") != "P1"
            or entry.get("kind") != "search"
            or entry.get("phrasing") not in ("keyword", None)
        ):
            continue
        body = entry["body"]
        if entry["source"] == "openalex":
            page = parse_works(json.loads(body) if isinstance(body, str) else body)
        else:
            page = parse_feed(body)
        pages.setdefault(entry["topic"], {})[entry["source"]] = page.candidates
    return {
        topic: interleave([by_source.get(s, ()) for s in SOURCE_ORDER])
        for topic, by_source in sorted(pages.items())
    }


def _work_of(records: Sequence[PaperCandidate], rule: MatchRule) -> dict[RecordKey, int]:
    work: dict[RecordKey, int] = {}
    for index, members in enumerate(dedup.group(records, rule)):
        for i in members:
            work[(records[i].source, records[i].source_id)] = index
    return work


@dataclass(frozen=True, slots=True)
class PairError:
    pair: str
    #: `false_merge` or `missed`.
    kind: str
    #: For a false merge, why the rule merged; for a miss, the first condition the
    #: direct pair failed (`title`, `year`, `author`, `arxiv_veto`), or
    #: `transitive` when the pair passes but the grouping still split them.
    cause: str
    a_title: str
    b_title: str
    years: tuple[int | None, int | None]


@dataclass(frozen=True, slots=True)
class Score:
    tp: int
    fp: int
    fn: int
    tn: int
    errors: tuple[PairError, ...] = ()

    @property
    def precision(self) -> Interval | None:
        return wilson_interval(self.tp, self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> Interval:
        return wilson_interval(self.tp, self.tp + self.fn)


def _miss_cause(a: PaperCandidate, b: PaperCandidate, rule: MatchRule) -> str:
    _, xa = paper_identity(a.doi, a.arxiv_id)
    _, xb = paper_identity(b.doi, b.arxiv_id)
    if xa is not None and xb is not None and xa != xb:
        return "arxiv_veto"
    words_a, words_b = (frozenset(normalise_title(c.title).split()) for c in (a, b))
    if title_overlap(words_a, words_b) < rule.title_threshold:
        return "title"
    if a.year is not None and b.year is not None and abs(a.year - b.year) > rule.year_tolerance:
        return "year"
    if a.authors and b.authors and not name_words(a.authors[0]) & name_words(b.authors[0]):
        return "author"
    return "transitive"


def score(
    rule: MatchRule,
    pairs: Sequence[LabelledPair],
    records: Mapping[str, Sequence[PaperCandidate]],
) -> Score:
    """The grouped decision for every pair, against its label."""
    tp = fp = fn = tn = 0
    errors: list[PairError] = []
    by_topic = {topic: _work_of(records[topic], rule) for topic in {p.topic for p in pairs}}
    lookup = {(topic, (c.source, c.source_id)): c for topic, recs in records.items() for c in recs}
    for p in pairs:
        work = by_topic[p.topic]
        merged = work[p.a] == work[p.b]
        if merged and p.same:
            tp += 1
        elif not merged and not p.same:
            tn += 1
        else:
            a, b = lookup[(p.topic, p.a)], lookup[(p.topic, p.b)]
            if merged:
                fp += 1
                reason = dedup.same_work(a, b, rule)
                cause = reason.value if reason is not None else "transitive"
            else:
                fn += 1
                cause = _miss_cause(a, b, rule)
            errors.append(
                PairError(
                    p.pair,
                    "false_merge" if merged else "missed",
                    cause,
                    a.title,
                    b.title,
                    (a.year, b.year),
                )
            )
    return Score(tp, fp, fn, tn, tuple(errors))


@dataclass(frozen=True, slots=True)
class GridRow:
    title_threshold: float
    year_tolerance: int
    tp: int
    fp: int
    fn: int


def fit(
    pairs: Sequence[LabelledPair], records: Mapping[str, Sequence[PaperCandidate]]
) -> tuple[MatchRule, tuple[GridRow, ...]]:
    """D148's fit: the highest recall with no false merge, ties to larger T, then smaller Y.

    Call it with training pairs only. It cannot check that itself, because a split
    is a property of how the caller filtered, not of a pair.
    """
    rows = []
    for t in THRESHOLDS:
        for y in YEAR_TOLERANCES:
            s = score(MatchRule(t, y), pairs, records)
            rows.append(GridRow(t, y, s.tp, s.fp, s.fn))
    clean = [r for r in rows if r.fp == 0]
    if not clean:
        raise ValueError("no rule on the grid merges without a false merge on the training pairs")
    best = max(clean, key=lambda r: (r.tp, r.title_threshold, -r.year_tolerance))
    return MatchRule(best.title_threshold, best.year_tolerance), tuple(rows)


@dataclass(frozen=True, slots=True)
class Latency:
    records: int
    p50_ms: float
    p95_ms: float
    max_ms: float


def time_grouping(records: Mapping[str, Sequence[PaperCandidate]], rule: MatchRule) -> Latency:
    """Grouping and merging one search's records, timed per search (inside NFR-P1's 3.0 s)."""
    samples = []
    for recs in records.values():
        dedup.works(recs, rule)  # warm: the first call pays for imports and caches
        for _ in range(LATENCY_REPEATS):
            started = time.perf_counter()
            dedup.works(recs, rule)
            samples.append((time.perf_counter() - started) * 1000.0)
    samples.sort()
    return Latency(
        records=max(len(r) for r in records.values()),
        p50_ms=samples[len(samples) // 2],
        p95_ms=samples[min(len(samples) - 1, int(0.95 * len(samples)))],
        max_ms=samples[-1],
    )


@dataclass(frozen=True, slots=True)
class DedupReport:
    generated_at: str
    fixture_sha256: str
    dedup_version: str
    fitted: MatchRule
    shipped: MatchRule
    grid: tuple[GridRow, ...]
    train: Score
    held_out: Score
    #: The identifier-only rule on the held-out pairs: what the title rule adds.
    held_out_ids_only: Score
    latency: Latency
    config: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        def interval(i: Interval | None) -> dict[str, float] | None:
            return None if i is None else {"point": i.point, "low": i.low, "high": i.high, "n": i.n}

        def scored(s: Score) -> dict[str, Any]:
            return {
                **{k: v for k, v in asdict(s).items() if k != "errors"},
                "precision": interval(s.precision),
                "recall": interval(s.recall),
                "errors": [asdict(e) for e in s.errors],
            }

        payload = {
            "kind": "dedup",
            "generated_at": self.generated_at,
            "fixture_sha256": self.fixture_sha256,
            "dedup_version": self.dedup_version,
            "fitted": asdict(self.fitted),
            "shipped": asdict(self.shipped),
            "grid": [asdict(r) for r in self.grid],
            "train": scored(self.train),
            "held_out": scored(self.held_out),
            "held_out_ids_only": scored(self.held_out_ids_only),
            "latency": asdict(self.latency),
            "config": self.config,
        }
        return json.dumps(payload, indent=2)


#: A rule no title can satisfy, so only identifiers merge: the baseline the title
#: rule has to beat.
IDS_ONLY = MatchRule(title_threshold=1.0, year_tolerance=0)


def _ids_only_score(
    pairs: Sequence[LabelledPair], records: Mapping[str, Sequence[PaperCandidate]]
) -> Score:
    # Jaccard 1.0 still merges identical titles, so the baseline is computed with
    # titles blanked rather than with a threshold.
    blanked = {topic: tuple(_without_title(c) for c in recs) for topic, recs in records.items()}
    return score(IDS_ONLY, pairs, blanked)


def _without_title(c: PaperCandidate) -> PaperCandidate:
    # A unique non-word per record: its title overlaps no other record's.
    return replace(c, title=f"{c.source}{c.source_id}".replace(".", "").replace("/", ""))


def run_dedup() -> DedupReport:
    pairs = load_pairs()
    records = topic_records()
    train = [p for p in pairs if p.split == "train"]
    held_out = [p for p in pairs if p.split == "held_out"]
    fitted, grid = fit(train, records)
    return DedupReport(
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        dedup_version=dedup.DEDUP_VERSION,
        fitted=fitted,
        shipped=dedup.SHIPPED_RULE,
        grid=grid,
        train=score(fitted, train, records),
        held_out=score(dedup.SHIPPED_RULE, held_out, records),
        held_out_ids_only=_ids_only_score(held_out, records),
        latency=time_grouping(records, dedup.SHIPPED_RULE),
        config={
            "fixture": str(FIXTURE.relative_to(ROOT)),
            "pools": str(POOLS.relative_to(ROOT)),
            "train_pairs": len(train),
            "held_out_pairs": len(held_out),
            "objective": (
                "max train recall with zero train false merges; ties to larger T, then smaller Y"
            ),
            "intervals": "Wilson 95%; pairs share records, so these are narrower than the truth",
        },
    )


def render(report: DedupReport) -> str:
    def line(name: str, s: Score) -> str:
        precision = str(s.precision) if s.precision is not None else "n/a (nothing merged)"
        return (
            f"  {name:<18} precision {precision}   recall {s.recall}   "
            f"(tp {s.tp}, fp {s.fp}, fn {s.fn}, tn {s.tn})"
        )

    out = [
        f"Deduplication (FR-D3, D146, D148) — {report.dedup_version}",
        f"  fixture sha256 {report.fixture_sha256}",
        f"  fitted on train: T={report.fitted.title_threshold} Y={report.fitted.year_tolerance}"
        f"   shipped: T={report.shipped.title_threshold} Y={report.shipped.year_tolerance}"
        + ("" if report.fitted == report.shipped else "   <-- SHIPPED RULE IS NOT THE FIT"),
        line("train (fitted)", report.train),
        line("held out", report.held_out),
        line("held out, ids only", report.held_out_ids_only),
        f"  grouping latency per search ({report.latency.records} records): "
        f"p50 {report.latency.p50_ms:.2f} ms, p95 {report.latency.p95_ms:.2f} ms, "
        f"max {report.latency.max_ms:.2f} ms",
        "  held-out errors:",
    ]
    for e in report.held_out.errors:
        out.append(
            f"    {e.pair} {e.kind} ({e.cause}) {e.years}: {e.a_title[:60]!r} / {e.b_title[:60]!r}"
        )
    if not report.held_out.errors:
        out.append("    none")
    out.append(
        "  not established: recall within the candidate set only;"
        " intervals assume independent pairs"
    )
    return "\n".join(out)
