"""Fitting the weighted-fusion weight, on the train split only (D76, doc 6 §6.6).

Doc 6 keeps weighted score fusion as "a fair ablation" against RRF because it
"needs per-query normalization and a weight fitted on a labeled set — which does
exist here". **Which labelled set is the entire question.** CORE and FULL are the
report; a weight chosen on them would put a tuned configuration against RRF, whose
one constant (k=60) is fitted on nothing, and the delta would then measure the
tuning rather than the method.

D-RET's train split exists for this: 629 queries over 207 papers, disjoint by paper
from the 69 evaluation papers — the paper-level separation `verify_split` enforces
and doc 9 §9.2 calls the first thing a careful interviewer probes.

**The objective is Recall@10** (doc 9 §9.3's primary metric), not nDCG@10: train
judgments are binary (D69), and a graded metric computed on binary judgments is the
weaker form doc 9 deliberately keeps out of the FULL block. MRR is scored beside it
at every weight, so a disagreement between the two objectives is visible in the
report rather than hidden by whichever one was chosen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scholiast.domain.retrieval import weighted_score_fusion
from scholiast.eval.dret import DRet, Query, Split
from scholiast.eval.harness import ArmHits
from scholiast.eval.metrics import RECALL_CUTOFFS, recall_at_k, reciprocal_rank

#: Weights swept on the dense arm; the lexical arm takes the remainder. A 0.1 grid
#: is as fine as 629 queries can resolve — at this n the difference between 0.5 and
#: 0.55 is far inside the noise the bootstrap reports on the evaluation sets.
DEFAULT_WEIGHT_GRID: tuple[float, ...] = tuple(round(step / 10, 1) for step in range(11))

#: The fitting objective, named in the artifact so a reader never has to infer it.
OBJECTIVE = f"recall@{min(RECALL_CUTOFFS)}"


def training_queries(dataset: DRet) -> list[Query]:
    """Train-split queries that carry judgments — the only queries fitting may see.

    Returned as a list rather than filtered at the call site so there is one place
    that decides what "fitting data" means, and one place for a test to assert that
    no evaluation query can reach it.
    """
    queries = [q for q in dataset.queries if q.split is Split.TRAIN and dataset.qrels(q.id)]
    if not queries:
        raise ValueError(
            "no train-split queries with judgments — a weight fitted on the "
            "evaluation queries would report its own tuning as a result (D76)"
        )
    return queries


@dataclass(frozen=True, slots=True)
class WeightScore:
    """What one candidate weight achieved on the fitting set."""

    weight: float
    recall: float
    mrr: float


@dataclass(frozen=True, slots=True)
class FusionFit:
    """The fitted weight, the curve behind it, and what the runner-up objective said."""

    weight: float
    curve: tuple[WeightScore, ...]
    objective: str
    mrr_weight: float
    n_queries: int

    @property
    def objectives_agree(self) -> bool:
        """Whether MRR would have chosen the same weight as Recall@10.

        Reported rather than resolved: two objectives picking different weights is
        evidence that the curve is flat and the choice weakly determined, which is
        worth more to a reader than a single number that looks decisive.
        """
        return self.weight == self.mrr_weight


def score_weight(hits: Sequence[ArmHits], weight: float, *, depth: int) -> WeightScore:
    """Mean Recall@10 and MRR over the fitting set at one weight. Pure arithmetic."""
    if not hits:
        raise ValueError("cannot score a weight over an empty fitting set")
    cutoff = min(RECALL_CUTOFFS)
    recalls: list[float] = []
    rrs: list[float] = []
    for arm_hits in hits:
        fused = weighted_score_fusion(
            [arm_hits.dense, arm_hits.lexical],
            weights=(weight, 1.0 - weight),
            limit=depth,
        )
        ranking = [hit.chunk_id for hit in fused]
        recalls.append(recall_at_k(ranking, arm_hits.qrels, cutoff))
        rrs.append(reciprocal_rank(ranking, arm_hits.qrels))
    return WeightScore(
        weight=weight,
        recall=sum(recalls) / len(recalls),
        mrr=sum(rrs) / len(rrs),
    )


def fit_dense_weight(
    hits: Sequence[ArmHits],
    *,
    grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
    depth: int = max(RECALL_CUTOFFS),
) -> FusionFit:
    """Choose the dense weight that maximises Recall@10 on the fitting set.

    Ties keep the first weight on the grid, so the choice is reproducible and never
    depends on iteration order (FR-E4). Weight 0.0 and 1.0 stay in the grid on
    purpose: they are the degenerate single-arm configurations, and a fit that
    lands on one of them is itself a finding — it says the fusion has nothing to
    add at this normalisation.
    """
    if not grid:
        raise ValueError("the weight grid is empty, so nothing can be fitted")
    curve = tuple(score_weight(hits, weight, depth=depth) for weight in grid)
    best = max(curve, key=lambda score: score.recall)
    best_mrr = max(curve, key=lambda score: score.mrr)
    return FusionFit(
        weight=best.weight,
        curve=curve,
        objective=OBJECTIVE,
        mrr_weight=best_mrr.weight,
        n_queries=len(hits),
    )
