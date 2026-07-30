"""Retrieval, fusion, evidence assembly, and citations.

The pure logic lives here because it is the part most worth unit-testing and the
part least worth mocking: rank fusion is arithmetic, per-paper capping is a
counting rule, and citation verification is a set membership check. None of it
needs a database, a GPU, or a network.

`reciprocal_rank_fusion` deserves the comment it gets — the choice of ranks over
scores is the reason hybrid retrieval works at all here (doc 6 §6.6).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import SectionKind

#: Conventional RRF constant. Larger k flattens the contribution of top ranks,
#: making fusion more forgiving of one arm being confidently wrong.
RRF_K = 60

#: No single paper may occupy more than this share of the evidence budget.
#: Without it, one verbose paper turns a multi-paper workspace into a
#: single-paper answer — the QA-path echo of the comparison coverage rule
#: (FR-C2, doc 6 §6.9 step 3).
DEFAULT_PER_PAPER_BUDGET_SHARE = 0.5


@unique
class RetrievalArm(StrEnum):
    """Which retriever produced a result.

    Kept on every scored chunk so the eval harness can attribute a win to the
    lexical or the dense arm rather than to "hybrid" as an opaque whole.
    """

    LEXICAL = "lexical"
    DENSE = "dense"
    FUSED = "fused"
    RERANKED = "reranked"


@unique
class QueryIntent(StrEnum):
    """Routing classes (doc 6 §6.8).

    Routing exists to keep expensive paths off cheap queries — and, for
    OUT_OF_SCOPE, to avoid spending a generation call at all.
    """

    LOOKUP = "lookup"
    COMPARATIVE = "comparative"
    MULTI_HOP = "multi_hop"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True, slots=True)
class Query:
    """A retrieval request, already resolved against conversation history.

    `text` is the *rewritten*, self-contained form — "what about their ablations?"
    is not retrievable, so rewriting happens before a Query exists (FR-Q8).
    `original_text` is retained for display and for the rewriting ablation.
    """

    text: str
    workspace_id: str
    paper_ids: tuple[str, ...]
    intent: QueryIntent = QueryIntent.LOOKUP
    original_text: str | None = None
    section_priors: tuple[SectionKind, ...] = ()

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise InvariantViolation("query text cannot be blank")
        if not self.paper_ids:
            # Guards the failure mode where an empty scope silently becomes a
            # global search across tenants.
            raise InvariantViolation("query must be scoped to at least one paper")

    @property
    def was_rewritten(self) -> bool:
        return self.original_text is not None and self.original_text != self.text


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk with the score that retrieved it."""

    chunk_id: str
    paper_id: str
    score: float
    arm: RetrievalArm
    rank: int | None = None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[ScoredChunk]],
    *,
    k: int = RRF_K,
    limit: int | None = None,
) -> list[ScoredChunk]:
    """Fuse ranked lists by reciprocal rank (doc 6 §6.6, ML-7).

    Uses **ranks, not scores**, and that is the whole point. Cosine similarity
    clusters tightly around 0.6-0.9 while BM25 is unbounded and query-dependent,
    so their score distributions are not comparable and shift per query. Weighted
    score fusion therefore needs per-query normalisation that is itself fragile.
    Ranks are immune: position 1 means the same thing in both lists.

    Input order within each ranking is authoritative — callers pass lists already
    sorted best-first. Ties are broken deterministically by chunk_id so eval runs
    are reproducible (FR-E4).
    """
    if k <= 0:
        raise InvariantViolation(f"RRF k must be positive, got {k}")

    fused: dict[str, float] = {}
    paper_of: dict[str, str] = {}

    for ranking in rankings:
        for position, scored in enumerate(ranking, start=1):
            fused[scored.chunk_id] = fused.get(scored.chunk_id, 0.0) + 1.0 / (k + position)
            paper_of[scored.chunk_id] = scored.paper_id

    ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    if limit is not None:
        ordered = ordered[:limit]

    return [
        ScoredChunk(
            chunk_id=chunk_id,
            paper_id=paper_of[chunk_id],
            score=score,
            arm=RetrievalArm.FUSED,
            rank=rank,
        )
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]


@unique
class CitationStatus(StrEnum):
    """Outcome of the three-layer citation check (doc 6 §6.11).

    UNVERIFIED is surfaced in the UI rather than hidden: a claim whose evidence
    does not entail it is exactly what the user needs to know about (NFR-U4).
    """

    VERIFIED = "verified"  # label resolves AND evidence entails the claim
    UNVERIFIED = "unverified"  # label resolves, entailment not established
    FABRICATED = "fabricated"  # label does not exist in the context set


@dataclass(frozen=True, slots=True)
class Citation:
    """A link from an answer span to the chunk that supports it."""

    label: str
    chunk_id: str
    claim_text: str
    status: CitationStatus
    entailment_score: float | None = None

    @property
    def is_trustworthy(self) -> bool:
        return self.status is CitationStatus.VERIFIED


@dataclass(slots=True)
class EvidenceSet:
    """The assembled context for one generation call.

    Holds the ordered chunks, their stable `[Cn]` labels, and the disclosure
    fields that every answer must carry. Labels are assigned here so a citation
    is a lookup into a known set rather than a string to be trusted.
    """

    chunks: list[ScoredChunk] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)  # chunk_id -> "C1"
    excluded_paper_ids: tuple[str, ...] = ()
    degraded: tuple[str, ...] = ()

    @classmethod
    def assemble(
        cls,
        reranked: Sequence[ScoredChunk],
        *,
        limit: int,
        per_paper_share: float = DEFAULT_PER_PAPER_BUDGET_SHARE,
        excluded_paper_ids: Sequence[str] = (),
        degraded: Sequence[str] = (),
    ) -> EvidenceSet:
        """Select and label evidence, enforcing the per-paper cap.

        Preserves rerank order among the survivors: relevance ordering is decided
        upstream, and this step only removes what the cap forbids.
        """
        if limit <= 0:
            raise InvariantViolation(f"evidence limit must be positive, got {limit}")

        distinct_papers = len({c.paper_id for c in reranked})
        # With a single paper in scope the cap is meaningless — applying it would
        # discard evidence for no reason.
        cap = limit if distinct_papers <= 1 else max(1, int(limit * per_paper_share))

        selected: list[ScoredChunk] = []
        per_paper: dict[str, int] = {}
        for chunk in reranked:
            if len(selected) >= limit:
                break
            if per_paper.get(chunk.paper_id, 0) >= cap:
                continue
            selected.append(chunk)
            per_paper[chunk.paper_id] = per_paper.get(chunk.paper_id, 0) + 1

        labels = {c.chunk_id: f"C{i}" for i, c in enumerate(selected, start=1)}
        return cls(
            chunks=selected,
            labels=labels,
            excluded_paper_ids=tuple(excluded_paper_ids),
            degraded=tuple(degraded),
        )

    @property
    def is_empty(self) -> bool:
        return not self.chunks

    @property
    def top_score(self) -> float | None:
        """Highest relevance score — the abstention signal (ML-13)."""
        return max((c.score for c in self.chunks), default=None)

    @property
    def paper_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for c in self.chunks:
            seen.setdefault(c.paper_id, None)
        return tuple(seen)

    def label_for(self, chunk_id: str) -> str | None:
        return self.labels.get(chunk_id)

    def resolve(self, label: str) -> str | None:
        """Map a `[Cn]` label back to its chunk id.

        Layer 2 of citation integrity: a label that does not resolve was
        fabricated, and this is a deterministic, free check.
        """
        for chunk_id, assigned in self.labels.items():
            if assigned == label:
                return chunk_id
        return None

    def should_abstain(self, threshold: float) -> bool:
        """Whether evidence is too weak to answer from (FR-Q4).

        `threshold` is fitted on labelled data and reported as a
        precision/recall curve, never hand-tuned until the demo looks good
        (doc 9 §9.6).
        """
        top = self.top_score
        return top is None or top < threshold
