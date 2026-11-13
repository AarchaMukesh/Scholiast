"""D177's abstention signals, scored: which non-fused score says "no evidence", and where.

M5 phase 1 fits tau, the score below which a question is refused rather than
answered (FR-Q4). Everything here was fixed by D177 and its two clarifications
before any signal was scored, so no choice below can have been made after seeing
one:

- **What is asked where.** Answerable and QASPER-unanswerable questions in D168's
  scopes (`scope_for`: own paper plus papers drawn by hash, 5 draws, sizes 5 / 10
  / 25, 10 decides). Withheld-paper questions in the 10-paper scope the owner
  checked by hand, and at size 5 in its first five papers, a subset of it; never
  at 25, where 15 papers were never checked.
- **Three signals, none fused (D167):** the top dense cosine, the top lexical
  score, and the top cross-encoder score over the RRF top 50, measured only.
- **Every question counts once.** An instance weighs 1/k, k being its question's
  instances at that size, so five draws of one question do not outvote one
  withheld question.
- **The rule** (`fit_threshold`, `choose`, `decide`): on train, each signal's tau
  is the lowest observed score whose false-answer rate is at most 0.10; the
  signal with the highest answer recall there is chosen, the dense cosine within
  0.01. Eval applies it unchanged.

**The rule's comparisons are exact.** Weights are integers (1/k scaled by the
least common multiple of every k) and rates are `Fraction`s, so a false-answer
rate of exactly 0.10 is 0.10 and not 0.10000000000000002 - the difference between
a threshold that is allowed and one that is not.

**A missing score is no evidence.** A scope in which a search returns nothing has
no top score; it is recorded as `None` and ranks below every threshold, which is
what refusing on it would do.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import VectorIndex
from scholiast.domain.lexical import query_weights, tokenize
from scholiast.domain.retrieval import ScoredChunk, reciprocal_rank_fusion
from scholiast.eval.answerability import WITHHELD_SCOPE_SIZE, Category, withheld_scope
from scholiast.eval.dret import DRet
from scholiast.eval.gate import DECIDING_SIZE, DEPTH, DRAWS, SIZES, scope_for
from scholiast.eval.harness import EmbedderLike, LexicalLike, RerankerLike, _rerank
from scholiast.eval.metrics import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED

#: tau's version stamp (NFR-M5), carried on every message once tau is wired in.
ABSTENTION_VERSION = "abstention/1.0.0"

#: The signals, in D177's order; the order also breaks an exact tie in recall.
SIGNALS: tuple[str, ...] = ("dense", "lexical", "rerank")
#: The signal that wins within `TIE_MARGIN`: it costs production nothing extra.
PREFERRED = "dense"

FALSE_ANSWER_LIMIT = Fraction(1, 10)
TIE_MARGIN = Fraction(1, 100)
#: Below this train recall at the limit, no tau is adopted (D177).
RECALL_FLOOR = Fraction(1, 2)
#: Above this eval false-answer rate, nothing is wired in and the owner decides.
EVAL_FALSE_ANSWER_CEILING = Fraction(1, 5)

#: Withheld-paper questions are asked only where their scope was checked by hand
#: (D177's 2026-10-28 clarification): the checked 10, and its first 5.
WITHHELD_SIZES: tuple[int, ...] = (5, WITHHELD_SCOPE_SIZE)

#: How many instances are re-scored at the end, and the key that picks them.
REPRODUCE_SAMPLE = 20
REPRODUCE_KEY = "d177-reproduce"

#: The rule as the archive states it, beside every number it produced.
RULE = (
    "on train at N=10, each signal's tau is the lowest observed score whose weighted "
    "false-answer rate (both unanswerable categories pooled) is at most 0.10; the signal "
    "with the highest answer recall there is chosen, the dense cosine within 0.01; no "
    "signal at recall 0.50 -> no tau; eval false-answer rate above 0.20 -> owner decides"
)

UNANSWERABLE: tuple[Category, ...] = (
    Category.UNANSWERABLE_QASPER,
    Category.UNANSWERABLE_WITHHELD,
)


# --- what is asked where ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Instance:
    """One question asked in one scope."""

    question_id: str
    #: The question's own paper; for a withheld question, the one left out. The
    #: bootstrap resamples these, so a withheld question moves with its paper.
    paper_id: str
    split: str
    category: Category
    size: int
    draw: int
    question: str
    scope: tuple[str, ...]

    @property
    def key(self) -> str:
        return f"{self.category.value}:{self.question_id}:{self.size}:{self.draw}"


def instances(sets: Mapping[str, Any], dataset: DRet) -> list[Instance]:
    """Every (question, scope) D177 scores, in the sets' order.

    A withheld question's scope is recomputed and compared with the one the sets
    recorded, which is the one the owner checked: a corpus that changed since
    would draw a different scope, and scoring it would score a question nobody
    checked.
    """
    papers = sorted(dataset.papers)
    built: list[Instance] = []
    for entry in sets["questions"]:
        category = Category(entry["category"])
        question_id, paper_id = entry["question_id"], entry["paper_id"]
        if category is Category.UNANSWERABLE_WITHHELD:
            checked = tuple(sorted(entry["scope"]))
            if withheld_scope(question_id, paper_id, papers) != checked:
                raise ValueError(
                    f"{question_id}: the sets record a withheld scope this corpus does "
                    "not draw, so it is not the scope the hand-check judged"
                )
            scopes = [
                (size, 0, withheld_scope(question_id, paper_id, papers, size=size))
                for size in WITHHELD_SIZES
            ]
        else:
            scopes = [
                (size, draw, scope_for(question_id, paper_id, papers, size=size, draw=draw))
                for size in SIZES
                for draw in range(DRAWS)
            ]
        built.extend(
            Instance(
                question_id=question_id,
                paper_id=paper_id,
                split=entry["split"],
                category=category,
                size=size,
                draw=draw,
                question=entry["question"],
                scope=scope,
            )
            for size, draw, scope in scopes
        )
    return built


# --- the scores -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Scored:
    """One instance's three top scores. `None`: the scope returned nothing."""

    key: str
    question_id: str
    paper_id: str
    split: str
    category: Category
    size: int
    draw: int
    scores: Mapping[str, float | None]

    @classmethod
    def of(cls, instance: Instance, scores: Mapping[str, float | None]) -> Scored:
        return cls(
            key=instance.key,
            question_id=instance.question_id,
            paper_id=instance.paper_id,
            split=instance.split,
            category=instance.category,
            size=instance.size,
            draw=instance.draw,
            scores=dict(scores),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "split": self.split,
            "category": self.category.value,
            "size": self.size,
            "draw": self.draw,
            "scores": dict(self.scores),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Scored:
        return cls(
            key=data["key"],
            question_id=data["question_id"],
            paper_id=data["paper_id"],
            split=data["split"],
            category=Category(data["category"]),
            size=data["size"],
            draw=data["draw"],
            scores=dict(data["scores"]),
        )


def _top(hits: Sequence[ScoredChunk]) -> float | None:
    return max((hit.score for hit in hits), default=None)


async def score_one(
    instance: Instance,
    *,
    dataset: DRet,
    index: VectorIndex,
    lexical: LexicalLike,
    embedder: EmbedderLike,
    reranker: RerankerLike,
    vectors: dict[str, Sequence[float]],
) -> Scored:
    """The three signals for one instance, by the calls production's hybrid path makes.

    Dense and lexical search the scope exactly as `run_hybrid` does; the
    cross-encoder reorders their RRF top 50 as the `hybrid_rerank` arm does, with
    the same contextual header. Each question is embedded once (`vectors`) and one
    at a time, as production embeds a question: batching would perturb fp16
    vectors (D74).
    """
    vector = vectors.get(instance.question)
    if vector is None:
        vector = (await embedder.embed([instance.question], role=TextRole.QUERY))[0]
        vectors[instance.question] = vector
    dense = await index.search(vector, paper_ids=instance.scope, limit=DEPTH)
    sparse = await lexical.search_lexical(
        query_weights(tokenize(instance.question)), paper_ids=instance.scope, limit=DEPTH
    )
    fused = reciprocal_rank_fusion([dense, sparse], limit=DEPTH)
    reranked = await _rerank(dataset, instance.question, fused, reranker) if fused else []
    return Scored.of(
        instance, {"dense": _top(dense), "lexical": _top(sparse), "rerank": _top(reranked)}
    )


async def score_all(
    todo: Sequence[Instance],
    *,
    done: Mapping[str, Scored],
    sink: Callable[[Scored], None],
    **adapters: Any,
) -> list[Scored]:
    """Every instance scored, reusing `done` and handing each new row to `sink` at once.

    The sink writes to disk as each row is made, so a standby or a crash costs the
    instance in flight and nothing else (D160); a rerun passes what was written
    back as `done`.
    """
    vectors: dict[str, Sequence[float]] = {}
    rows: list[Scored] = []
    for instance in todo:
        row = done.get(instance.key)
        if row is None:
            row = await score_one(instance, vectors=vectors, **adapters)
            sink(row)
        rows.append(row)
    return rows


def read_progress(path: Path) -> dict[str, Scored]:
    """Rows a previous, interrupted run wrote. Only a torn LAST line is forgiven."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[str, Scored] = {}
    for number, line in enumerate(lines, start=1):
        try:
            row = Scored.from_json(json.loads(line))
        except (json.JSONDecodeError, KeyError, ValueError):
            if number == len(lines):
                break
            raise
        rows[row.key] = row
    return rows


class ProgressRow(Protocol):
    """What a progress file holds a line of: phase 1's scores, phase 2's answers."""

    def to_json(self) -> dict[str, Any]: ...


@contextmanager
def progress_writer(path: Path) -> Iterator[Callable[[ProgressRow], None]]:
    """A sink that appends each row to `path` and flushes it before returning.

    Synchronous on purpose: the run scores one instance at a time, so nothing else
    is waiting on the event loop while a line is written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:

        def sink(row: ProgressRow) -> None:
            handle.write(json.dumps(row.to_json(), sort_keys=True) + "\n")
            handle.flush()

        yield sink


def reproduce_sample(todo: Sequence[Instance]) -> list[Instance]:
    """A fixed sample of deciding-size instances to re-score, chosen by hash of the key."""
    deciding = [i for i in todo if i.size == DECIDING_SIZE]
    ranked = sorted(
        deciding, key=lambda i: hashlib.sha256(f"{REPRODUCE_KEY}:{i.key}".encode()).digest()
    )
    return ranked[:REPRODUCE_SAMPLE]


def largest_differences(
    first: Sequence[Scored], second: Sequence[Scored]
) -> dict[str, float | None]:
    """Per signal, the largest absolute difference between two scorings of the same rows.

    `None` when one scoring found nothing where the other found something: no
    finite number describes that difference, and reporting 0 or infinity would
    each misstate it.
    """
    largest: dict[str, float | None] = {}
    for signal in SIGNALS:
        pairs = [(a.scores[signal], b.scores[signal]) for a, b in zip(first, second, strict=True)]
        if any((x is None) != (y is None) for x, y in pairs):
            largest[signal] = None
        else:
            largest[signal] = max(
                (abs(x - y) for x, y in pairs if x is not None and y is not None), default=0.0
            )
    return largest


# --- weights, rates, thresholds ---------------------------------------------------


Weighted = Sequence[tuple[Scored, int]]


def weighted(rows: Sequence[Scored]) -> list[tuple[Scored, int]]:
    """Each row with an integer weight proportional to 1/k, k its question's instance count.

    Computed once, on the rows as measured, and carried with them: a bootstrap
    that drew one paper twice must count its questions twice, which recomputing
    the weights on the resample would silently undo.
    """
    counts = Counter((row.category, row.question_id, row.size) for row in rows)
    scale = math.lcm(*counts.values()) if counts else 1
    return [(row, scale // counts[(row.category, row.question_id, row.size)]) for row in rows]


def _value(row: Scored, signal: str) -> float:
    score = row.scores[signal]
    return -math.inf if score is None else score


@dataclass(frozen=True, slots=True)
class Rates:
    """At one threshold: answer recall, and the false-answer rate pooled and per category."""

    answer_recall: Fraction | None
    false_answer: Fraction | None
    by_category: Mapping[Category, Fraction | None]


def _ratio(part: int, whole: int) -> Fraction | None:
    return Fraction(part, whole) if whole else None


def rates(rows: Weighted, signal: str, tau: float) -> Rates:
    """Who is answered at `tau`: everything at or above it (D177)."""
    answered: Counter[Category] = Counter()
    total: Counter[Category] = Counter()
    for row, weight in rows:
        total[row.category] += weight
        if _value(row, signal) >= tau:
            answered[row.category] += weight
    return Rates(
        answer_recall=_ratio(answered[Category.ANSWERABLE], total[Category.ANSWERABLE]),
        false_answer=_ratio(
            sum(answered[c] for c in UNANSWERABLE), sum(total[c] for c in UNANSWERABLE)
        ),
        by_category={c: _ratio(answered[c], total[c]) for c in UNANSWERABLE},
    )


def _sweep(rows: Weighted, signal: str) -> Iterator[tuple[float, Counter[Category]]]:
    """Every observed score, highest first, with the weight answered at or above it.

    A missing score is never a threshold: nothing can be "at or above" no evidence.
    """
    ordered = sorted(rows, key=lambda pair: _value(pair[0], signal), reverse=True)
    answered: Counter[Category] = Counter()
    position = 0
    while position < len(ordered):
        tau = _value(ordered[position][0], signal)
        if tau == -math.inf:
            return
        while position < len(ordered) and _value(ordered[position][0], signal) == tau:
            row, weight = ordered[position]
            answered[row.category] += weight
            position += 1
        yield tau, answered


@dataclass(frozen=True, slots=True)
class Fit:
    signal: str
    #: `math.inf` when no observed score keeps the false-answer rate in bounds:
    #: then nothing is answered, and recall is 0.
    tau: float
    answer_recall: Fraction
    false_answer: Fraction


def fit_threshold(rows: Weighted, signal: str, *, limit: Fraction = FALSE_ANSWER_LIMIT) -> Fit:
    """The lowest observed score whose false-answer rate is at most `limit` (D177).

    Lowering tau only ever answers more, so the false-answer rate never falls as
    the sweep descends: the first threshold that exceeds the limit ends it, and
    the last one inside it is the highest recall the limit allows.
    """
    totals: Counter[Category] = Counter()
    for row, weight in rows:
        totals[row.category] += weight
    answerable = totals[Category.ANSWERABLE]
    unanswerable = sum(totals[c] for c in UNANSWERABLE)
    if not answerable or not unanswerable:
        raise ValueError("a threshold needs answerable and unanswerable instances to fit on")
    best = Fit(signal, math.inf, Fraction(0), Fraction(0))
    for tau, answered in _sweep(rows, signal):
        false_answer = Fraction(sum(answered[c] for c in UNANSWERABLE), unanswerable)
        if false_answer > limit:
            break
        best = Fit(signal, tau, Fraction(answered[Category.ANSWERABLE], answerable), false_answer)
    return best


def choose(fits: Mapping[str, Fit]) -> str:
    """The signal with the highest train recall; the dense cosine within `TIE_MARGIN`."""
    best = max(fits.values(), key=lambda fit: (fit.answer_recall, -SIGNALS.index(fit.signal)))
    preferred = fits.get(PREFERRED)
    if preferred is not None and preferred.answer_recall >= best.answer_recall - TIE_MARGIN:
        return PREFERRED
    return best.signal


def decide(fits: Mapping[str, Fit], chosen: str, held_out: Rates) -> dict[str, Any]:
    """D177's outcomes, fixed in advance, read from the train fits and the eval rates.

    One case D177 did not foresee is sent to the owner rather than resolved here:
    the dense cosine chosen by the tie-break while below the recall floor that
    another signal reaches.
    """
    reaching = [
        signal
        for signal in SIGNALS
        if signal in fits and fits[signal].answer_recall >= RECALL_FLOOR
    ]
    if not reaching:
        return {
            "outcome": "no tau adopted",
            "reason": f"no signal reaches answer recall {float(RECALL_FLOOR)} at a false-answer "
            f"rate of at most {float(FALSE_ANSWER_LIMIT)} on train; production keeps D60's "
            "structural refusal only, and abstention moves to the model's own refusals (phase 2)",
        }
    if chosen not in reaching:
        return {
            "outcome": "owner decides",
            "reason": f"the tie-break chose {chosen}, below answer recall {float(RECALL_FLOOR)}, "
            f"while {', '.join(reaching)} reach it; D177 does not say which applies",
        }
    false_answer = held_out.false_answer
    if false_answer is None or false_answer > EVAL_FALSE_ANSWER_CEILING:
        return {
            "outcome": "owner decides",
            "reason": f"the eval false-answer rate {_show(false_answer)} exceeds "
            f"{float(EVAL_FALSE_ANSWER_CEILING)}; nothing is wired in (D177)",
        }
    return {
        "outcome": "adopt",
        "reason": f"{chosen} at tau {fits[chosen].tau!r}: train recall "
        f"{_show(fits[chosen].answer_recall)}, eval false-answer rate {_show(false_answer)}",
    }


def _show(value: Fraction | None) -> str:
    return "undefined" if value is None else f"{float(value):.3f}"


# --- what the report shows --------------------------------------------------------


def auc(rows: Weighted, signal: str, negatives: Sequence[Category]) -> float | None:
    """Weighted Mann-Whitney: the chance an answerable instance outscores an unanswerable one.

    Ties count half; counted in halves so the arithmetic stays in integers. `None`
    when either side is empty.
    """
    positives = [(_value(r, signal), w) for r, w in rows if r.category is Category.ANSWERABLE]
    against: Counter[float] = Counter()
    for row, weight in rows:
        if row.category in negatives:
            against[_value(row, signal)] += weight
    if not positives or not against:
        return None
    values = sorted(against)
    below = [0]
    for value in values:
        below.append(below[-1] + against[value])
    half_wins = sum(
        weight * (2 * below[bisect.bisect_left(values, value)] + against.get(value, 0))
        for value, weight in positives
    )
    return float(Fraction(half_wins, 2 * sum(w for _, w in positives) * below[-1]))


def curve(rows: Weighted, signal: str) -> dict[str, list[float | None]]:
    """The full trade-off: at every observed score, recall and the false-answer rates."""
    totals: Counter[Category] = Counter()
    for row, weight in rows:
        totals[row.category] += weight
    unanswerable = sum(totals[c] for c in UNANSWERABLE)
    points: dict[str, list[float | None]] = {
        "tau": [],
        "answer_recall": [],
        "false_answer": [],
        **{f"false_answer/{c.value}": [] for c in UNANSWERABLE},
    }
    for tau, answered in _sweep(rows, signal):
        points["tau"].append(tau)
        points["answer_recall"].append(
            _float(_ratio(answered[Category.ANSWERABLE], totals[Category.ANSWERABLE]))
        )
        points["false_answer"].append(
            _float(_ratio(sum(answered[c] for c in UNANSWERABLE), unanswerable))
        )
        for c in UNANSWERABLE:
            points[f"false_answer/{c.value}"].append(_float(_ratio(answered[c], totals[c])))
    return points


def _float(value: Fraction | None) -> float | None:
    return None if value is None else float(value)


def paper_intervals(
    rows: Weighted,
    signal: str,
    tau: float,
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, dict[str, Any]]:
    """Every rate at `tau` with a 95% interval that resamples PAPERS (D177).

    A paper's questions, their draws and a withheld question about it move
    together: they share the paper's text, so they are not independent, and
    resampling instances would narrow every interval for no reason. The weights
    travel with the rows. A resample with no instance of some category leaves that
    category's rate undefined; such resamples are counted, not imputed.
    """
    by_paper: dict[str, list[tuple[Scored, int]]] = defaultdict(list)
    for pair in rows:
        by_paper[pair[0].paper_id].append(pair)
    papers = sorted(by_paper)
    names = ["answer_recall", "false_answer", *(f"false_answer/{c.value}" for c in UNANSWERABLE)]

    def flat(measured: Rates) -> dict[str, Fraction | None]:
        return {
            "answer_recall": measured.answer_recall,
            "false_answer": measured.false_answer,
            **{f"false_answer/{c.value}": measured.by_category[c] for c in UNANSWERABLE},
        }

    point = flat(rates(rows, signal, tau))
    drawn: dict[str, list[float]] = {name: [] for name in names}
    undefined: Counter[str] = Counter()
    rng = random.Random(seed)  # noqa: S311 - a bootstrap must be seedable (metrics.py)
    for _ in range(iterations):
        sample = [pair for _ in papers for pair in by_paper[papers[rng.randrange(len(papers))]]]
        for name, value in flat(rates(sample, signal, tau)).items():
            if value is None:
                undefined[name] += 1
            else:
                drawn[name].append(float(value))
    out: dict[str, dict[str, Any]] = {}
    for name in names:
        values = sorted(drawn[name])
        out[name] = {
            "point": _float(point[name]),
            "ci95": _percentiles(values),
            "undefined_resamples": undefined[name],
        }
    out["papers"] = {"resampled": len(papers), "iterations": iterations, "seed": seed}
    return out


def _percentiles(values: Sequence[float]) -> list[float] | None:
    """2.5th and 97.5th, by `bootstrap_interval`'s own convention."""
    if not values:
        return None
    n = len(values)
    return [values[int(0.025 * n)], values[min(n - 1, int(0.975 * n))]]


def analyse(rows: Sequence[Scored]) -> dict[str, Any]:
    """Every number D177 asks for, from the scored rows alone - no Qdrant, no model."""
    by: dict[tuple[str, int], list[Scored]] = defaultdict(list)
    for row in rows:
        by[(row.split, row.size)].append(row)
    train = weighted(by[("train", DECIDING_SIZE)])
    held_out = weighted(by[("eval", DECIDING_SIZE)])

    fits = {signal: fit_threshold(train, signal) for signal in SIGNALS}
    chosen = choose(fits)
    tau = fits[chosen].tau
    applied = rates(held_out, chosen, tau)

    def fit_json(fit: Fit) -> dict[str, Any]:
        return {
            "tau": None if fit.tau == math.inf else fit.tau,
            "answers_nothing": fit.tau == math.inf,
            "answer_recall": float(fit.answer_recall),
            "false_answer": float(fit.false_answer),
        }

    sizes: dict[str, Any] = {}
    for size in SIZES:
        per_split: dict[str, Any] = {}
        for split in ("train", "eval"):
            these = weighted(by[(split, size)])
            if not these:
                continue
            per_split[split] = {
                signal: {
                    "auc": auc(these, signal, UNANSWERABLE),
                    **{f"auc/{c.value}": auc(these, signal, (c,)) for c in UNANSWERABLE},
                    "at_deciding_tau": _rates_json(rates(these, signal, fits[signal].tau)),
                    "curve": curve(these, signal),
                }
                for signal in SIGNALS
            }
        sizes[f"N={size}"] = {
            "deciding": size == DECIDING_SIZE,
            "withheld_reported": size in WITHHELD_SIZES,
            **per_split,
        }

    return {
        "train_fits": {signal: fit_json(fit) for signal, fit in fits.items()},
        "chosen": {"signal": chosen, **fit_json(fits[chosen])},
        "eval": paper_intervals(held_out, chosen, tau),
        "verdict": decide(fits, chosen, applied),
        "sizes": sizes,
    }


def _rates_json(measured: Rates) -> dict[str, float | None]:
    return {
        "answer_recall": _float(measured.answer_recall),
        "false_answer": _float(measured.false_answer),
        **{f"false_answer/{c.value}": _float(measured.by_category[c]) for c in UNANSWERABLE},
    }
