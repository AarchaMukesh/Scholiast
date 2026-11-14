"""Retrieval metrics (doc 9 §9.3, FR-E2).

Four numbers, each answering a different question, none a substitute for another:

- **Recall@k** — of the chunks that *are* relevant, how many did we surface? The
  primary metric, because nothing can be grounded that was never retrieved.
- **nDCG@k** — rank-sensitive, graded. The headline *ranking-quality* number.
- **MRR** — how high is the first relevant hit? Matters for single-fact lookup,
  where the top result is the whole answer.
- **Recall@50** doubles as §9.3's **reranking ceiling**: a reranker cannot recover
  what the first stage never surfaced, so this bounds the whole two-stage design.

Deliberately pure. No I/O, no index, no embedder — a ranking and a qrel mapping in,
a float out. That is what lets the tests below check them against values computed by
hand rather than against the pipeline's own output, which would be circular.

**nDCG is computed on the grades that exist, and they are currently binary.** Doc 9
§9.2 specifies a 3-point scale, but grade 1 arrives with the CORE hand-review (D69),
so today every judgment is grade 2. The formula is the graded one and needs no
change when grade 1 lands; what changes is how much it can discriminate. A report
that prints `DRet.grade_scale` alongside these numbers says so without anyone having
to remember to.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Doc 9 §9.3's cutoffs. 50 is also the reranking ceiling the two-stage design is
#: bounded by, which is why it survives even though the QA path returns 8.
RECALL_CUTOFFS: tuple[int, ...] = (10, 20, 50)
NDCG_CUTOFF = 10


def recall_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of relevant chunks appearing in the top `k`.

    Undefined with no relevant chunks — and that is raised rather than returned as
    0.0 or 1.0, because either default silently shifts the mean over a query set.
    D-RET drops such queries at build time (`DRet.dropped_queries`), so reaching
    here with an empty qrel set means something upstream changed.
    """
    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant chunks")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    found = sum(1 for chunk_id in ranking[:k] if chunk_id in relevant)
    return found / len(relevant)


def reciprocal_rank(ranking: Sequence[str], relevant: Mapping[str, int]) -> float:
    """1 / rank of the first relevant hit, or 0.0 if there is none.

    0.0 for a miss is the standard convention and is *not* the same judgement call
    as recall's above: a miss is a real, meaningful outcome here, whereas a query
    with nothing relevant is a malformed input.
    """
    if not relevant:
        raise ValueError("reciprocal rank is undefined for a query with no relevant chunks")
    for position, chunk_id in enumerate(ranking, start=1):
        if chunk_id in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Discounted cumulative gain with the exponential gain `2^rel - 1`.

    The exponential form rather than linear `rel`: it is the one that makes a
    grade-2 hit worth meaningfully more than a grade-1 hit, which is the entire
    reason doc 9 §9.2 specifies graded judgments. Under today's binary judgments
    the two forms coincide up to a constant factor, so this choice costs nothing
    now and starts mattering the moment the CORE hand-review populates grade 1.
    """
    total = 0.0
    for position, chunk_id in enumerate(ranking[:k], start=1):
        grade = relevant.get(chunk_id, 0)
        if grade:
            total += (2**grade - 1) / math.log2(position + 1)
    return total


def ndcg_at_k(ranking: Sequence[str], relevant: Mapping[str, int], k: int = NDCG_CUTOFF) -> float:
    """DCG normalised against the best achievable ordering of the same judgments.

    The ideal ranking is the relevant chunks sorted by grade descending — *not*
    the ranking we produced sorted by grade, which would normalise away exactly
    the ordering error the metric exists to detect.

    Capped at 1.0 rather than left to float drift: with equal grades the ideal and
    actual DCG are the same sum in a different order, and floating-point addition
    is not associative, so a perfect ranking can compute to 1.0000000000000002 and
    make a "better than ideal" row appear in a report.
    """
    if not relevant:
        raise ValueError("nDCG is undefined for a query with no relevant chunks")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ideal_order = sorted(relevant, key=lambda cid: relevant[cid], reverse=True)
    ideal = dcg_at_k(ideal_order, relevant, k)
    if ideal == 0.0:
        raise ValueError("every judgment for this query is grade 0, which is not a judgment")
    return min(1.0, dcg_at_k(ranking, relevant, k) / ideal)


#: Doc 9 section 9.4: resample queries with replacement, 1000 iterations, 95%.
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_SEED = 20260915


@dataclass(frozen=True, slots=True)
class Interval:
    """A point estimate and the interval it lives in. Never one without the other.

    Doc 9's guiding standard is that a number without a confidence interval is, at
    this n, frequently noise -- so the two are one type rather than two fields a
    reporter might print separately. Printing `Interval` prints both.
    """

    point: float
    low: float
    high: float
    n: int

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2.0

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_interval(
    values: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Percentile bootstrap over per-query values (doc 9 section 9.4).

    **Resamples queries, not judgments.** The query is the unit of independence:
    two judgments for one query are correlated, so resampling judgments would
    understate the interval and make every difference look more significant than
    it is -- the precise failure the section exists to prevent.

    Seeded, because FR-E4 requires two runs to agree and an unseeded bootstrap
    would move the interval on every invocation for reasons unrelated to retrieval.
    The seed is a constant rather than a parameter with a random default, so
    forgetting to pass one cannot silently produce an irreproducible report.

    For comparing two configurations use `paired_bootstrap_delta`, not two of
    these: overlapping intervals are not a significance test, and treating runs
    over the *same* queries as independent throws away the pairing that makes a
    small difference detectable at this n.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sample")
    point = sum(values) / len(values)
    if len(values) == 1:
        return Interval(point=point, low=point, high=point, n=1)

    # S311 flags non-cryptographic RNG. That is exactly what a bootstrap wants: it
    # must be seedable and reproducible, which a cryptographic source is designed
    # not to be. Suppressed here with the reason rather than relaxed in the config,
    # so the next genuine use of `random` in a security context still fails.
    rng = random.Random(seed)  # noqa: S311
    size = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return Interval(
        point=point,
        low=means[int(tail * iterations)],
        high=means[min(iterations - 1, int((1.0 - tail) * iterations))],
        n=size,
    )


#: The normal quantile for a two-sided 95% interval.
_Z95 = 1.959963984540054


def wilson_interval(successes: int, trials: int) -> Interval:
    """A proportion with its Wilson 95% score interval (D148).

    For counts, not per-query means: deduplication's precision and recall are
    proportions of pairs, where a bootstrap over a few dozen pairs is coarse and
    the normal approximation misbehaves near 1.0, which is exactly where a good
    matcher's precision sits. Wilson stays inside [0, 1] and is honest at the edge.

    It assumes independent trials. Pairs that share a record are not, so the
    interval is narrower than the truth; the caller says so where it prints one.
    """
    if trials <= 0:
        raise ValueError("a proportion of zero trials is undefined")
    if not 0 <= successes <= trials:
        raise ValueError(f"successes must lie in [0, {trials}], got {successes}")
    p = successes / trials
    z2 = _Z95 * _Z95
    centre = (p + z2 / (2 * trials)) / (1 + z2 / trials)
    spread = _Z95 * math.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / (1 + z2 / trials)
    return Interval(
        point=p, low=max(0.0, centre - spread), high=min(1.0, centre + spread), n=trials
    )


def cohen_kappa(first: Sequence[str], second: Sequence[str]) -> float | None:
    """Agreement between two labellers beyond what chance alone would give (D179).

    Raw agreement flatters a skewed label set: two labellers who both say
    "answered" 90% of the time agree 82% by chance. Kappa subtracts that. It is
    undefined, and returned as None rather than as 0 or 1, when chance agreement
    is total: both labellers used one and the same label for every item.
    """
    if len(first) != len(second):
        raise ValueError(f"{len(first)} labels against {len(second)}")
    if not first:
        raise ValueError("agreement over zero items is undefined")
    n = len(first)
    observed = sum(1 for a, b in zip(first, second, strict=True) if a == b) / n
    labels = set(first) | set(second)
    expected = sum((first.count(label) / n) * (second.count(label) / n) for label in labels)
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def paired_bootstrap_delta(
    treatment: Sequence[float],
    baseline: Sequence[float],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    confidence: float = 0.95,
    seed: int = BOOTSTRAP_SEED,
) -> Interval:
    """Interval on the mean per-query difference (doc 9 section 9.4, point 2).

    **Paired, because the configurations were evaluated on identical queries.**
    Two independent intervals would be the wrong test twice over: overlapping
    intervals do not mean "no difference", and treating paired runs as
    independent discards the correlation that makes a small effect visible at
    n around 50. Resampling query INDICES and differencing within each resample
    keeps the pairing intact.

    Returns the delta as `treatment - baseline`. An interval straddling zero is
    the honest way to report "within noise" -- which doc 9 insists is as valuable
    a result as a win, and is what phase 5's ablation table marks with an X.
    """
    if len(treatment) != len(baseline):
        raise ValueError(
            f"paired comparison needs the same queries in both arms, got "
            f"{len(treatment)} and {len(baseline)}"
        )
    if not treatment:
        raise ValueError("cannot bootstrap an empty sample")
    differences = [t - b for t, b in zip(treatment, baseline, strict=True)]
    point = sum(differences) / len(differences)
    if len(differences) == 1:
        return Interval(point=point, low=point, high=point, n=1)

    rng = random.Random(seed)  # noqa: S311 - see bootstrap_interval
    size = len(differences)
    means = []
    for _ in range(iterations):
        means.append(sum(differences[rng.randrange(size)] for _ in range(size)) / size)
    means.sort()
    tail = (1.0 - confidence) / 2.0
    return Interval(
        point=point,
        low=means[int(tail * iterations)],
        high=means[min(iterations - 1, int((1.0 - tail) * iterations))],
        n=size,
    )
