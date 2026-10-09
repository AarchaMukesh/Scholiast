"""Ranking discovered works, with every score explainable from its parts (FR-D4, FR-D5, D151, D152).

**Three parts, each in [0, 1], and a weighted sum.**

- *Relevance*: how well the work's title and abstract match the query. Computed
  outside this module (lexically here, or densely by an encoder the application
  owns) and min-max scaled within the search.
- *Citation impact*: `log(1 + citations / (age + 1))`, where age is years since the
  work first appeared (phase 7's earliest year). Raw counts favour old papers by
  construction; dividing by age compares a 2024 paper with a 2014 one on what each
  has earned per year, and the logarithm stops one landmark paper flattening the
  rest. Min-max scaled over the works whose count is known.
- *Recency*: `0.5 ** (age / 3)`, a three-year half-life fixed in D151, not fitted.

**Unknown is neither zero nor dropped** (D151, D152). arXiv reports no citation
counts, so an arXiv-only work's citation part is unknown. It takes the median of
the search's known values and is flagged `imputed`, so it is neither punished as
uncited nor rewarded by reweighting. The flagged value is the one used, so the
score is always exactly `sum(weight * part)` of what is shown.

**Parts are scaled over everything the search found, then filters remove works**
(D152), so a filter never changes a remaining work's explanation.

Pure and synchronous: dozens of works per search, inside the request.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique

from scholiast.domain.dedup import DiscoveredWork
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.lexical import BM25_B, BM25_K1, tokenize

#: NFR-M5: which ranker ordered a result. Bump on any change to a part's formula.
RANKER_VERSION = "rank-weighted-1"

#: D151: recency halves every three years. Fixed, not fitted.
HALF_LIFE_YEARS = 3.0


@unique
class RelevanceSignal(StrEnum):
    """How the relevance part is computed (D151 compares the two)."""

    LEXICAL = "lexical"
    DENSE = "dense"


@dataclass(frozen=True, slots=True)
class RankingWeights:
    """How much each part counts. Non-negative and summing to one."""

    relevance: float
    citation: float
    recency: float

    def __post_init__(self) -> None:
        parts = (self.relevance, self.citation, self.recency)
        if any(w < 0 for w in parts):
            raise InvariantViolation(f"ranking weights cannot be negative, got {parts}")
        if not math.isclose(sum(parts), 1.0, abs_tol=1e-9):
            raise InvariantViolation(f"ranking weights must sum to 1, got {sum(parts)}")


#: Fitted on the D149 judgments' training topics by `scholiast.eval.ranking.fit`
#: (D151). Change them only by re-running that fit; a test holds the two equal.
SHIPPED_SIGNAL = RelevanceSignal.DENSE
SHIPPED_WEIGHTS = RankingWeights(relevance=0.8, citation=0.0, recency=0.2)


@dataclass(frozen=True, slots=True)
class ScoreParts:
    """A work's three parts, as used in its score (FR-D5)."""

    relevance: float
    citation: float
    recency: float
    #: True when the part is the search's median, not a measurement (D152).
    citation_imputed: bool = False
    recency_imputed: bool = False


@dataclass(frozen=True, slots=True)
class RankedWork:
    work: DiscoveredWork
    score: float
    parts: ScoreParts
    #: Where the work sat before ranking (D147's interleaved position), 0-based.
    #: Breaks score ties, so equal scores keep the sources' order.
    position: int


def lexical_relevance(query: str, texts: Sequence[str]) -> list[float]:
    """BM25 of `query` against each text, with IDF from these texts alone.

    The search's own works are the corpus: a term every result shares (the topic's
    own name, usually) carries almost no weight, and the terms that separate one
    result from another carry the most. Uses the project's identifier-preserving
    tokenizer and BM25 constants (ML-6), so `LLaMA-2` and `nDCG@10` survive.
    """
    documents = [tokenize(text) for text in texts]
    if not documents:
        return []
    count = len(documents)
    average = sum(len(d) for d in documents) / count or 1.0
    frequency: Counter[str] = Counter()
    for doc in documents:
        frequency.update(set(doc))
    terms = list(dict.fromkeys(tokenize(query)))
    idf = {t: math.log(1.0 + (count - frequency[t] + 0.5) / (frequency[t] + 0.5)) for t in terms}
    scores = []
    for doc in documents:
        tf = Counter(doc)
        norm = BM25_K1 * (1.0 - BM25_B + BM25_B * len(doc) / average)
        scores.append(
            sum(idf[t] * tf[t] * (BM25_K1 + 1.0) / (tf[t] + norm) for t in terms if tf[t])
        )
    return scores


def relevance_text(work: DiscoveredWork) -> str:
    """What relevance is computed over: the title, then the abstract if any (D151)."""
    return f"{work.title}. {work.abstract}" if work.abstract else work.title


def _age(year: int | None, ranking_year: int) -> int | None:
    return None if year is None else max(0, ranking_year - year)


def citation_impact(citations: int | None, year: int | None, ranking_year: int) -> float | None:
    """`log(1 + citations / (age + 1))`, or None when either input is unknown."""
    age = _age(year, ranking_year)
    if citations is None or age is None:
        return None
    return math.log1p(citations / (age + 1))


def recency(year: int | None, ranking_year: int) -> float | None:
    """`0.5 ** (age / 3)`: 1.0 this year, 0.5 three years on. None for an unknown year."""
    age = _age(year, ranking_year)
    return None if age is None else 0.5 ** (age / HALF_LIFE_YEARS)


def scale(values: Sequence[float]) -> list[float]:
    """Min-max into [0, 1]. No spread scales to 1.0, not 0.0: equal is not irrelevant (D152)."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high == low:
        return [1.0 if high > 0 else 0.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def _fill(values: Sequence[float | None]) -> tuple[list[float], list[bool]]:
    """Unknowns take the median of the known (D151); with none known, 0.5 for all."""
    known = [v for v in values if v is not None]
    median = statistics.median(known) if known else 0.5
    return [median if v is None else v for v in values], [v is None for v in values]


def rank(
    works: Sequence[DiscoveredWork],
    relevance: Sequence[float],
    weights: RankingWeights,
    *,
    ranking_year: int,
) -> tuple[RankedWork, ...]:
    """`works` (in D147's order) scored and sorted best first, ties kept in that order."""
    if len(relevance) != len(works):
        raise InvariantViolation(
            f"one relevance score per work: {len(relevance)} scores for {len(works)} works"
        )
    rel = scale(relevance)
    raw_cit = [citation_impact(w.citation_count, w.year, ranking_year) for w in works]
    known = [v for v in raw_cit if v is not None]
    scaled_known = iter(scale(known))
    cit, cit_imputed = _fill([next(scaled_known) if v is not None else None for v in raw_cit])
    rec, rec_imputed = _fill([recency(w.year, ranking_year) for w in works])
    ranked = [
        RankedWork(
            work=w,
            score=weights.relevance * rel[i] + weights.citation * cit[i] + weights.recency * rec[i],
            parts=ScoreParts(rel[i], cit[i], rec[i], cit_imputed[i], rec_imputed[i]),
            position=i,
        )
        for i, w in enumerate(works)
    ]
    return tuple(sorted(ranked, key=lambda r: (-r.score, r.position)))


@unique
class SortOrder(StrEnum):
    """FR-D8's sorts. Unknowns sort last whatever the order."""

    SCORE = "score"
    CITATIONS = "citations"
    YEAR = "year"


@dataclass(frozen=True, slots=True)
class SearchFilters:
    """FR-D8's filters, applied after ranking (D124, D152). None means unfiltered."""

    year_from: int | None = None
    year_to: int | None = None
    min_citations: int | None = None
    #: Case-insensitive substring of the venue.
    venue: str | None = None
    sort: SortOrder = SortOrder.SCORE

    def __post_init__(self) -> None:
        if (
            self.year_from is not None
            and self.year_to is not None
            and self.year_from > self.year_to
        ):
            raise InvariantViolation(
                f"year range is empty: {self.year_from} is after {self.year_to}"
            )
        if self.min_citations is not None and self.min_citations < 0:
            raise InvariantViolation(
                f"minimum citations cannot be negative, got {self.min_citations}"
            )


@dataclass(frozen=True, slots=True)
class Filtered:
    works: tuple[RankedWork, ...]
    #: How many works each filter removed, counted against the first filter that
    #: removed each work, in the order year, citations, venue.
    excluded: dict[str, int] = field(default_factory=dict)
    #: Of the citation exclusions, how many had no count at all: "unknown" cannot
    #: satisfy "at least N", and a user should see that it was unknown, not low.
    excluded_unknown_citations: int = 0


def apply_filters(ranked: Sequence[RankedWork], filters: SearchFilters) -> Filtered:
    """Keep the works every filter admits, in the requested order (FR-D8)."""
    kept: list[RankedWork] = []
    excluded: Counter[str] = Counter()
    unknown_citations = 0
    needle = filters.venue.casefold().strip() if filters.venue else None
    for r in ranked:
        year, count, venue = r.work.year, r.work.citation_count, r.work.venue
        if (filters.year_from is not None or filters.year_to is not None) and (
            year is None
            or (filters.year_from is not None and year < filters.year_from)
            or (filters.year_to is not None and year > filters.year_to)
        ):
            excluded["year"] += 1
        elif filters.min_citations is not None and (count is None or count < filters.min_citations):
            excluded["citations"] += 1
            unknown_citations += count is None
        elif needle and (venue is None or needle not in venue.casefold()):
            excluded["venue"] += 1
        else:
            kept.append(r)
    if filters.sort is SortOrder.CITATIONS:
        kept.sort(key=lambda r: (r.work.citation_count is None, -(r.work.citation_count or 0)))
    elif filters.sort is SortOrder.YEAR:
        kept.sort(key=lambda r: (r.work.year is None, -(r.work.year or 0)))
    return Filtered(tuple(kept), dict(excluded), unknown_citations)
