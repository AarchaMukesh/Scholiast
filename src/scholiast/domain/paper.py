"""Paper, its structure, and its chunks.

Two things here are load-bearing beyond ordinary data modelling:

`SectionKind` — the canonical section taxonomy. `RELATED_WORK` is the reason this
enum exists at all: a claim in Related Work belongs to *another* paper, and a
chunker that cannot tell the difference will confidently attribute a baseline's
result to the paper being read. That failure mode is the one doc 2 §2.3 opens with.

`Provenance` — every chunk carries enough information to reconstruct the exact
source span (FR-I5). Without it a citation is a gesture rather than a reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum, unique
from typing import Self

from scholiast.domain.discovery import ARXIV_DOI_PREFIX, normalise_arxiv_id, normalise_doi
from scholiast.domain.errors import FailureClass, InvariantViolation


@unique
class SectionKind(StrEnum):
    """Canonical section taxonomy (doc 6 §6.2).

    Real papers name sections inconsistently ("Experiments" / "Empirical
    Evaluation" / "Results and Analysis"), so raw headings are normalised onto
    this fixed set. Downstream logic — attribution, dimension-targeted retrieval
    for comparison, section priors — depends on the normalised value, never the
    raw string.
    """

    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    METHOD = "method"
    EXPERIMENTS = "experiments"
    RESULTS = "results"
    DISCUSSION = "discussion"
    LIMITATIONS = "limitations"
    CONCLUSION = "conclusion"
    REFERENCES = "references"
    APPENDIX = "appendix"
    OTHER = "other"

    @property
    def describes_other_work(self) -> bool:
        """True where claims are typically about *someone else's* contribution.

        Used to mark chunks so generation does not misattribute a cited
        baseline's result to the paper under discussion.
        """
        return self in (SectionKind.RELATED_WORK, SectionKind.REFERENCES)

    @property
    def is_body(self) -> bool:
        """Excludes back-matter that is rarely worth retrieving as evidence."""
        return self not in (SectionKind.REFERENCES, SectionKind.APPENDIX)


# Ordered longest-phrase-first so "related work" wins over "work", and
# "experimental setup" over "setup".
_SECTION_PATTERNS: tuple[tuple[SectionKind, tuple[str, ...]], ...] = (
    (SectionKind.RELATED_WORK, ("related work", "prior work", "related research", "background")),
    (SectionKind.LIMITATIONS, ("limitation", "broader impact", "ethical consideration")),
    (
        SectionKind.EXPERIMENTS,
        (
            "experimental setup",
            "experiment",
            "implementation detail",
            "evaluation setup",
            "empirical",
        ),
    ),
    (SectionKind.CONCLUSION, ("conclusion", "concluding remark", "future work")),
    (SectionKind.INTRODUCTION, ("introduction",)),
    (SectionKind.ABSTRACT, ("abstract",)),
    (
        SectionKind.METHOD,
        ("methodology", "method", "our approach", "proposed", "architecture", "model"),
    ),
    (SectionKind.RESULTS, ("main result", "result", "finding")),
    (SectionKind.DISCUSSION, ("discussion", "ablation", "analysis")),
    (SectionKind.REFERENCES, ("reference", "bibliography")),
    (SectionKind.APPENDIX, ("appendix", "supplementary")),
)

_LEADING_NUMBER = re.compile(r"^\s*(?:\d+(?:\.\d+)*\.?|[IVXLC]+\.|[A-Z]\.)\s*")


def normalise_section(raw_title: str) -> SectionKind:
    """Map a raw heading onto the canonical taxonomy.

    Falls back to OTHER rather than guessing — an unrecognised section is
    retrievable but carries no section prior, which is the safe default.
    """
    cleaned = _LEADING_NUMBER.sub("", raw_title).strip().lower()
    for kind, needles in _SECTION_PATTERNS:
        if any(n in cleaned for n in needles):
            return kind
    return SectionKind.OTHER


@unique
class PaperVisibility(StrEnum):
    """Whether a paper is globally shared or tenant-private (doc 4 §4.4.2).

    The privacy carve-out: open-access papers are deduplicated globally so they
    are parsed and embedded once, but a *user upload* is never shared even on a
    content-hash collision. An unpublished manuscript must not become a registry
    entry reachable through another tenant's membership.
    """

    PUBLIC = "public"
    PRIVATE = "private"


@unique
class PaperOrigin(StrEnum):
    """How a paper entered the system, which decides whose metadata wins (D135).

    A discovered paper keeps the title and authors its source reported, the ones
    the user chose it by; GROBID only fills gaps. An upload has nothing but a
    filename, so GROBID's header is the best title it will get.
    """

    UPLOAD = "upload"
    DISCOVERY = "discovery"


@unique
class ParseStatus(StrEnum):
    """Whether this paper's structure has been extracted (D44).

    Parse state lives on the *paper*, not the membership, because parsing
    happens once per paper — so a second workspace adding an already-failed
    paper can see the outcome without re-parsing it.

    The outcome is separate from the reason: `FAILED` is always accompanied by a
    `FailureClass`, reusing the §4.4.1 taxonomy rather than duplicating three of
    its values into a second enum that could drift out of step.
    """

    PENDING = "pending"
    PARSING = "parsing"
    PARSED = "parsed"
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (ParseStatus.PARSED, ParseStatus.FAILED)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Exactly where a chunk came from (FR-I5).

    Character offsets index the parsed **section** text (`sections.text`, D42),
    not the PDF byte stream and not the whole document — so they survive
    re-extraction with the same parser version, and `parser_version` is part of
    the contract rather than metadata.

    Section-relative is total rather than merely convenient: chunking rule 1
    guarantees no chunk ever crosses a section boundary, so no span can straddle
    two sections and become ambiguous. Resolving a citation is one join —
    chunk → its section row → slice.
    """

    paper_id: str
    section_path: str
    section_kind: SectionKind
    page: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise InvariantViolation(
                f"char_end ({self.char_end}) must exceed char_start ({self.char_start})"
            )
        if self.page < 1:
            raise InvariantViolation(f"page must be 1-indexed, got {self.page}")

    @property
    def length(self) -> int:
        return self.char_end - self.char_start


@dataclass(frozen=True, slots=True)
class Section:
    """A node in the document's section hierarchy, with its canonical text.

    `text` is what every chunk offset in this section indexes (D42) — the
    entity owns it because `sections.text` is where a citation is resolved, and
    a Section that could be persisted without its text would make the round trip
    unsatisfiable one row at a time (D52).

    `path` is unique within a paper (D51). That is not decoration: chunks are
    mapped back onto section rows by path, so two sections sharing one would
    silently attribute a chunk to the wrong text and highlight the wrong span.
    """

    id: str
    paper_id: str
    kind: SectionKind
    raw_title: str
    path: str
    order_index: int
    page_start: int
    page_end: int
    text: str
    parent_id: str | None = None

    @property
    def describes_other_work(self) -> bool:
        return self.kind.describes_other_work


@unique
class ChunkRole(StrEnum):
    """Small-to-big retrieval (doc 6 §6.3).

    CHILD chunks are embedded and searched — small enough to match a query
    precisely. PARENT windows are returned as generation context — large enough
    to answer completely. Decoupling the two resolves a tension a single chunk
    size cannot.
    """

    CHILD = "child"
    PARENT = "parent"
    TABLE = "table"


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrieval unit with full provenance and version stamps.

    `chunker_version` and `embedder_version` (NFR-M5) are what make it possible
    to answer "did quality change because of the reranker or because I
    re-chunked?" — and to re-index only what actually changed.
    """

    id: str
    text: str
    role: ChunkRole
    ordinal: int
    provenance: Provenance
    chunker_version: str
    token_count: int
    embedder_version: str | None = None
    parent_id: str | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise InvariantViolation("chunk text cannot be blank")
        if self.role is ChunkRole.CHILD and self.parent_id is None:
            raise InvariantViolation("child chunks must reference a parent window")

    @property
    def is_indexed(self) -> bool:
        return self.embedder_version is not None

    def contextual_header(self, paper_title: str, year: int | None) -> str:
        """The prefix prepended before embedding (doc 6 §6.3, ML-3).

        Costs ~20 tokens and no LLM calls, and is the cheapest quality win in the
        design: it makes a Related Work chunk *lexically and semantically* marked
        as describing other work, which is the direct defence against the
        misattribution failure in doc 2 §2.3.
        """
        title = f"[Paper: {paper_title}" + (f" ({year})]" if year else "]")
        section = f"[Section: {self.provenance.section_path}]"
        if self.provenance.section_kind.describes_other_work:
            section += "\n[Note: describes other work, not this paper's contribution]"
        return f"{title}\n{section}"

    def embedding_input(self, paper_title: str, year: int | None) -> str:
        return f"{self.contextual_header(paper_title, year)}\n\n{self.text}"

    def with_embedding_version(self, version: str) -> Self:
        """Return a copy stamped as embedded. Chunks are immutable."""
        return type(self)(
            id=self.id,
            text=self.text,
            role=self.role,
            ordinal=self.ordinal,
            provenance=self.provenance,
            chunker_version=self.chunker_version,
            token_count=self.token_count,
            embedder_version=version,
            parent_id=self.parent_id,
        )


@dataclass(slots=True)
class Paper:
    """A work, its metadata, and its parsed structure.

    Identity is content-addressed: `content_hash` makes ingestion idempotent
    (FR-I7) and drives the visibility-scoped deduplication in doc 4 §4.4.2.

    **A discovered paper exists before its PDF does** (D132): it is added by its
    DOI or arXiv id, and its hash is written when the FETCHING stage has the bytes
    (D134). Until then `content_hash` is None, which is allowed only while nothing
    has been parsed and the paper has an identifier to deduplicate on.
    """

    id: str
    content_hash: str | None
    visibility: PaperVisibility
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    external_ids: dict[str, str] = field(default_factory=dict)
    owner_id: str | None = None
    parse_status: ParseStatus = ParseStatus.PENDING
    #: Set exactly when parse_status is FAILED. Reuses the §4.4.1 taxonomy so
    #: the retry decision and the user message come from one place (D44).
    parse_failure_class: FailureClass | None = None
    parser_version: str | None = None
    #: Object-store key for the retained TEI (D43, D49). Persisted identity in
    #: the same sense as `content_hash` and `parser_version`: it records which
    #: parse output is actually on disk, so a re-chunk ablation reads the bytes
    #: that produced today's chunks rather than re-running a GROBID that may
    #: have moved underneath it.
    tei_key: str | None = None
    #: Identity by external id (FR-I7, D132), one spelling each: a publisher DOI,
    #: never arXiv's DataCite one, and an unversioned, lowercase arXiv id.
    doi: str | None = None
    arxiv_id: str | None = None
    origin: PaperOrigin = PaperOrigin.UPLOAD
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        # Encodes the privacy carve-out as an invariant rather than a convention.
        # The database mirrors it with partial unique indexes (doc 5 §5.6.3) so a
        # bug in one layer cannot violate it alone.
        if self.visibility is PaperVisibility.PRIVATE and self.owner_id is None:
            raise InvariantViolation("private papers must have an owner")
        if self.visibility is PaperVisibility.PUBLIC and self.owner_id is not None:
            raise InvariantViolation("public papers must not be owner-scoped")
        if not self.title.strip():
            raise InvariantViolation("paper title cannot be blank")
        # Mirrors the database CHECK. A failure with no class cannot produce a
        # retry decision or a user message, and a class without a failure claims
        # a problem that did not happen.
        if (self.parse_status is ParseStatus.FAILED) != (self.parse_failure_class is not None):
            raise InvariantViolation(
                f"parse_failure_class must be set iff parse_status is FAILED "
                f"(status={self.parse_status}, class={self.parse_failure_class})"
            )
        # D132, mirrored by ck_papers_hash_or_awaiting: no bytes means nothing has
        # been parsed, and the paper must be findable by an identifier meanwhile.
        if self.content_hash is None:
            if self.parse_status is not ParseStatus.PENDING:
                raise InvariantViolation("a paper without its PDF cannot have been parsed")
            if self.doi is None and self.arxiv_id is None:
                raise InvariantViolation("a paper without its PDF needs a DOI or an arXiv id")
        if self.doi is not None and (
            normalise_doi(self.doi) != self.doi or self.doi.startswith(ARXIV_DOI_PREFIX)
        ):
            raise InvariantViolation(f"paper DOI must be a normalised publisher DOI: {self.doi!r}")
        if self.arxiv_id is not None and normalise_arxiv_id(self.arxiv_id) != self.arxiv_id:
            raise InvariantViolation(
                f"paper arXiv id must be normalised and unversioned: {self.arxiv_id!r}"
            )

    @property
    def is_globally_shared(self) -> bool:
        """Whether this paper may be reused across tenants.

        Only open-access papers are; uploads never are, regardless of content
        hash. See PaperVisibility.
        """
        return self.visibility is PaperVisibility.PUBLIC

    @property
    def citation_impact_per_year(self) -> float | None:
        """Age-normalised citation impact for discovery ranking (FR-D4).

        Raw citation count is confounded by age — without normalising, 2015
        papers permanently bury strong 2025 work. Returns None when either the
        count or the year is unknown; the ranker must handle absent signals
        rather than defaulting them to zero, which would be a silent penalty.
        """
        if self.citation_count is None or self.year is None:
            return None
        # Current-year papers get a half-year of credit rather than dividing by
        # zero and reporting infinite impact.
        age = max(datetime.now().astimezone().year - self.year, 0) + 0.5
        return self.citation_count / age
