"""One paper, one entry: matching search results across sources into works (FR-D3, D145-D148).

A search asks OpenAlex and arXiv separately, so a paper both hold comes back twice,
and OpenAlex alone sometimes holds one paper twice (a preprint record beside its
published version). This module decides which records are one work and merges them.

**Rules first, measured** (D145). Two records are one work when they share an arXiv
id or a publisher DOI, or when their titles are near-identical, their years are
compatible and their first authors share a name word (D148). No paper encoder: the
spike's misses were titles and years, not meaning, and an encoder would cost a model
download, VRAM and latency inside D122's 1.0 s. The thresholds are not chosen here;
they are fitted on the fixture's training topics and scored on its held-out topics
(`scholiast.eval.dedup`), and a test holds `SHIPPED_RULE` to what the fit produces.

**An identifier is decisive, a title is evidence.** A shared identifier merges two
records whatever their titles say: the spike found OpenAlex records carrying a famous
paper's arXiv id and authors under an unrelated title, which the owner labelled the
same work (D146). The reverse does not hold. Two *different* arXiv ids are two
submissions, so they veto a title match: "A Survey of Large Language Models" is a
title more than one paper has had.

**A work is the transitive closure of matched pairs.** If a preprint matches its
camera-ready by title and the camera-ready matches a third record by DOI, all three
are one work. The fit scores this grouped decision, because that is what users see.

Pure and synchronous: it runs on at most a few dozen records per search, inside the
request (NFR-P1), and is timed by the eval rather than assumed cheap.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum, unique

from scholiast.domain.discovery import (
    OpenAccessLink,
    PaperCandidate,
    normalise_title,
    paper_identity,
)
from scholiast.domain.errors import InvariantViolation

#: NFR-M5: which matcher grouped a result, so a change in what users see can be
#: traced to a change in the rule rather than in the sources.
DEDUP_VERSION = "dedup-rules-1"


@unique
class MatchReason(StrEnum):
    """Why two records were judged one work, so an error can be counted by kind."""

    ARXIV_ID = "arxiv_id"
    DOI = "doi"
    TITLE = "title"


@dataclass(frozen=True, slots=True)
class MatchRule:
    """The tunable part of D148's rule. Everything else about it is fixed."""

    #: Minimum Jaccard overlap of the two titles' normalised word sets.
    title_threshold: float
    #: Largest year difference still compatible. A journal extension can follow its
    #: preprint by a year or two (D146's labelling found both).
    year_tolerance: int

    def __post_init__(self) -> None:
        if not 0.0 < self.title_threshold <= 1.0:
            raise InvariantViolation(
                f"a title threshold is a Jaccard overlap in (0, 1], got {self.title_threshold}"
            )
        if self.year_tolerance < 0:
            raise InvariantViolation(
                f"year tolerance cannot be negative, got {self.year_tolerance}"
            )


#: Fitted on the D146 fixture's training topics by `scholiast.eval.dedup.fit`
#: (D148). Change it only by re-running that fit; a test holds the two equal.
SHIPPED_RULE = MatchRule(title_threshold=0.85, year_tolerance=2)


def _fold(text: str) -> str:
    """Accents dropped and case folded, so `Büyükakyüz` and `Buyukakyuz` are one name."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch)).casefold()


def name_words(name: str) -> frozenset[str]:
    """The words of a person's name that can identify them, as a set.

    A set because sources disagree on order: `Ge Zheng` and `Zheng Ge` are one
    author in D146's fixture, and `Ho, Jonathan` is `Jonathan Ho` once the comma is
    gone. Single letters are initials and identify nobody.
    """
    words = "".join(ch if ch.isalnum() else " " for ch in _fold(name)).split()
    return frozenset(word for word in words if len(word) > 1)


@dataclass(frozen=True, slots=True)
class _Features:
    """What matching reads from one record, computed once per record, not per pair."""

    doi: str | None
    arxiv_id: str | None
    title_words: frozenset[str]
    year: int | None
    first_author: frozenset[str] | None

    @classmethod
    def of(cls, candidate: PaperCandidate) -> _Features:
        doi, arxiv_id = paper_identity(candidate.doi, candidate.arxiv_id)
        first = name_words(candidate.authors[0]) if candidate.authors else None
        return cls(
            doi=doi,
            arxiv_id=arxiv_id,
            title_words=frozenset(normalise_title(candidate.title).split()),
            year=candidate.year,
            # A first author with no usable word is as unknown as no author at all.
            first_author=first or None,
        )


def title_overlap(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard overlap of two word sets; 0.0 when both are empty."""
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _match(a: _Features, b: _Features, rule: MatchRule) -> MatchReason | None:
    if a.arxiv_id is not None and a.arxiv_id == b.arxiv_id:
        return MatchReason.ARXIV_ID
    if a.doi is not None and a.doi == b.doi:
        return MatchReason.DOI
    if a.arxiv_id is not None and b.arxiv_id is not None:
        # Two submissions (D148's veto), whatever their titles say.
        return None
    if title_overlap(a.title_words, b.title_words) < rule.title_threshold:
        return None
    if a.year is not None and b.year is not None and abs(a.year - b.year) > rule.year_tolerance:
        return None
    if (
        a.first_author is not None
        and b.first_author is not None
        and not a.first_author & b.first_author
    ):
        return None
    # Unknown is compatible: a missing year or author list is no evidence of a
    # different paper, the rule Paper applies to absent values (PaperCandidate).
    return MatchReason.TITLE


def same_work(
    a: PaperCandidate, b: PaperCandidate, rule: MatchRule = SHIPPED_RULE
) -> MatchReason | None:
    """Why `a` and `b` are one work, or None if this rule cannot say they are."""
    return _match(_Features.of(a), _Features.of(b), rule)


def group(
    candidates: Sequence[PaperCandidate], rule: MatchRule = SHIPPED_RULE
) -> tuple[tuple[int, ...], ...]:
    """Indices into `candidates`, one tuple per work, transitively closed.

    Each work's indices ascend, and works are ordered by their first index: a work
    sits where its earliest record sat (D147).
    """
    features = [_Features.of(c) for c in candidates]
    parent = list(range(len(candidates)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if _match(features[i], features[j], rule) is not None:
                low, high = sorted((root(i), root(j)))
                parent[high] = low
    groups: dict[int, list[int]] = {}
    for i in range(len(candidates)):
        groups.setdefault(root(i), []).append(i)
    return tuple(tuple(members) for _, members in sorted(groups.items()))


@dataclass(frozen=True, slots=True)
class DiscoveredWork:
    """One paper as search found it: every record of it, merged (D147).

    Display fields (title, authors, venue, published date) are the `add` member's,
    so what the user sees is what they add (D148). The rest is merged across
    members: identifiers from the earliest member that has one, the earliest year
    (when the work first appeared, which FR-D4's citations per year divides by),
    the longest abstract, the largest citation count, every open-access link once.
    """

    #: Every record of the work, in the search's interleaved order.
    members: tuple[PaperCandidate, ...]
    #: The member the add endpoint should receive (D147).
    add: PaperCandidate
    doi: str | None
    arxiv_id: str | None
    year: int | None
    abstract: str | None
    #: None when no member reports one. Never 0 for unknown.
    citation_count: int | None
    oa_links: tuple[OpenAccessLink, ...]

    def __post_init__(self) -> None:
        if not self.members:
            raise InvariantViolation("a work has at least one record")
        if self.add not in self.members:
            raise InvariantViolation("a work's add reference must be one of its records")

    @property
    def title(self) -> str:
        return self.add.title

    @property
    def authors(self) -> tuple[str, ...]:
        return self.add.authors

    @property
    def venue(self) -> str | None:
        return self.add.venue

    @property
    def published(self) -> date | None:
        return self.add.published


def _choose_add(
    members: Sequence[PaperCandidate], fetchable: Callable[[PaperCandidate], bool]
) -> PaperCandidate:
    """D147: a member with an arXiv id and a link the fetcher may follow, else any
    member with such a link, else the first.

    arXiv first because its record resolves to the arXiv PDF, the one host the
    spike found reliably fetchable, and because add resolves identity from the
    record it receives (D133).
    """
    usable = [m for m in members if fetchable(m)]
    with_arxiv = [m for m in usable if paper_identity(m.doi, m.arxiv_id)[1] is not None]
    return (with_arxiv or usable or list(members))[0]


def merge(
    members: Sequence[PaperCandidate],
    fetchable: Callable[[PaperCandidate], bool] = lambda _: False,
) -> DiscoveredWork:
    """The work these records are, merged by D147's rules."""
    identities = [paper_identity(m.doi, m.arxiv_id) for m in members]
    years = [m.year for m in members if m.year is not None]
    abstracts = [m.abstract for m in members if m.abstract]
    counts = [m.citation_count for m in members if m.citation_count is not None]
    links: dict[str, OpenAccessLink] = {}
    for member in members:
        for link in member.oa_links:
            links.setdefault(link.url, link)
    return DiscoveredWork(
        members=tuple(members),
        add=_choose_add(members, fetchable),
        doi=next((doi for doi, _ in identities if doi is not None), None),
        arxiv_id=next((arxiv for _, arxiv in identities if arxiv is not None), None),
        year=min(years, default=None),
        # max() keeps the first of equally long abstracts: the earliest member's.
        abstract=max(abstracts, key=len, default=None),
        # The largest, not the sum: one work held twice by OpenAlex would have its
        # shared citations counted twice (D148).
        citation_count=max(counts, default=None),
        oa_links=tuple(links.values()),
    )


def works(
    candidates: Sequence[PaperCandidate],
    rule: MatchRule = SHIPPED_RULE,
    fetchable: Callable[[PaperCandidate], bool] = lambda _: False,
) -> tuple[DiscoveredWork, ...]:
    """`candidates`, in interleaved order, as works in order of first appearance."""
    return tuple(
        merge([candidates[i] for i in members], fetchable) for members in group(candidates, rule)
    )
