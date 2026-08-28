"""Retrieval, fusion, evidence assembly, and citations.

The pure logic lives here because it is the part most worth unit-testing and the
part least worth mocking: rank fusion is arithmetic, per-paper capping is a
counting rule, and citation verification is a set membership check. None of it
needs a database, a GPU, or a network.

`reciprocal_rank_fusion` deserves the comment it gets — the choice of ranks over
scores is the reason hybrid retrieval works at all here (doc 6 §6.6).
"""

from __future__ import annotations

import re
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
    """A link from an answer span to the chunk that supports it.

    `chunk_id` is None exactly when the label was fabricated — there is no chunk
    to point at. Tying the two together as an invariant means a fabricated
    citation cannot be mistaken for a resolved one by a caller that forgot to
    check `status`.
    """

    label: str
    chunk_id: str | None
    claim_text: str
    status: CitationStatus
    entailment_score: float | None = None

    def __post_init__(self) -> None:
        fabricated = self.status is CitationStatus.FABRICATED
        if fabricated != (self.chunk_id is None):
            raise InvariantViolation(
                f"citation {self.label!r}: chunk_id must be None iff the label was "
                f"fabricated (status={self.status}, chunk_id={self.chunk_id!r})"
            )

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


# ---------------------------------------------------------------------------
# Citation integrity, layer 2 (doc 6 §6.11, FR-Q2)
#
# Layer 1 constrains the surface: the model may only cite [C1]…[Cn] labels that
# were put in front of it. Layer 2 — here — checks that what came back is
# actually from that set. It is deterministic, costs nothing, and catches the
# crudest failure with certainty. Layer 3 (does the chunk *support* the claim?)
# is entailment and arrives with the NLI adapter.
#
# The whole point is that a citation becomes a database lookup rather than a
# string to be trusted.
# ---------------------------------------------------------------------------

#: A citation marker. Deliberately liberal about `[C1, C2]`: models emit it, it
#: is unambiguous, and *dropping* it would be the worse failure — a real
#: citation would silently vanish from the answer and inflate nothing, so the
#: error would never surface.
_CITATION_MARKER = re.compile(r"\[(C\d+(?:\s*,\s*C\d+)*)\]")
_LABEL = re.compile(r"C\d+")

#: Sentence boundaries, used only to attribute a claim to a citation. Kept
#: separate from the chunker's splitter, which works on positioned blocks rather
#: than on a plain string.
_ANSWER_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    cursor = 0
    for match in _ANSWER_SENTENCE_END.finditer(text):
        bounds.append((cursor, match.start()))
        cursor = match.end()
    bounds.append((cursor, len(text)))
    return bounds


def _claim_containing(text: str, position: int, bounds: Sequence[tuple[int, int]]) -> str:
    for start, end in bounds:
        if start <= position < end:
            return text[start:end].strip()
    return text.strip()


@dataclass(frozen=True, slots=True)
class CitationAudit:
    """The outcome of checking an answer's citations against its evidence."""

    citations: tuple[Citation, ...]

    @property
    def fabricated(self) -> tuple[Citation, ...]:
        return tuple(c for c in self.citations if c.status is CitationStatus.FABRICATED)

    @property
    def resolved(self) -> tuple[Citation, ...]:
        return tuple(c for c in self.citations if c.status is not CitationStatus.FABRICATED)

    @property
    def is_clean(self) -> bool:
        """True when every emitted label resolved to real retrieved context."""
        return not self.fabricated

    @property
    def fabricated_labels(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for citation in self.fabricated:
            seen.setdefault(citation.label, None)
        return tuple(seen)


def audit_citations(answer: str, evidence: EvidenceSet) -> CitationAudit:
    """Check every emitted `[Cn]` label against the evidence set (FR-Q2).

    Returns one Citation per label *occurrence*, not per distinct label: the same
    chunk cited for two different claims is two citations, because layer 3 will
    have to judge each claim separately.

    Resolved labels are UNVERIFIED rather than VERIFIED — passing layer 2 only
    proves the label exists, not that the chunk supports the claim. Promoting
    them here would launder a syntactic check as a semantic one, which is the
    exact failure the three-layer design exists to avoid.
    """
    bounds = _sentence_bounds(answer)
    citations: list[Citation] = []

    for marker in _CITATION_MARKER.finditer(answer):
        claim = _claim_containing(answer, marker.start(), bounds)
        for label in _LABEL.findall(marker.group(1)):
            chunk_id = evidence.resolve(label)
            citations.append(
                Citation(
                    label=label,
                    chunk_id=chunk_id,
                    claim_text=claim,
                    status=(
                        CitationStatus.FABRICATED if chunk_id is None else CitationStatus.UNVERIFIED
                    ),
                )
            )

    return CitationAudit(citations=tuple(citations))


def strip_labels(answer: str, labels: Sequence[str]) -> str:
    """Remove specific citation labels from an answer, leaving the prose intact.

    Used on fabricated labels: showing `[C7]` when no C7 exists invites the user
    to click a citation that cannot resolve. A marker emptied of all its labels
    is removed whole, and the spacing it leaves behind is tidied — otherwise the
    answer acquires " ." artifacts that read as a rendering bug.

    Surviving markers come back in canonical one-label-per-bracket form, so
    `[C2, C1]` becomes `[C2][C1]`. That is deliberate: the UI has one shape to
    parse when it makes citations clickable.
    """
    if not labels:
        return answer
    doomed = set(labels)

    def replace(marker: re.Match[str]) -> str:
        kept = [x for x in _LABEL.findall(marker.group(1)) if x not in doomed]
        return "".join(f"[{label}]" for label in kept)

    stripped = _CITATION_MARKER.sub(replace, answer)
    return re.sub(r" +([.,;:])", r"\1", re.sub(r"[ \t]{2,}", " ", stripped)).strip()
