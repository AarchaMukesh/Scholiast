"""What an academic source returns, and the normalisation every discovery phase shares (D119).

A `PaperCandidate` is one search result from one source, before deduplication and
ranking. It is not a `Paper`: nothing has been fetched or parsed, and the only
identity it has is the identifiers the source reported, which may be wrong,
partial or absent. Phase 7 merges candidates that describe the same work, and phase
6 turns a chosen one into a `Paper`.

The normalisation functions live here rather than in an adapter because three
phases depend on them agreeing exactly: the adapters (phase 2), the cache key
(phase 4, `normalise_query`) and deduplication (phase 7, DOI, arXiv id and title).
Written once per adapter they would drift, and a DOI that normalises differently
in two places is two works.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from urllib.parse import urlsplit

from scholiast.domain.errors import DomainError, InvariantViolation

_WHITESPACE = re.compile(r"\s+")
_DOI_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)
#: arXiv's DataCite DOIs are another spelling of an arXiv id, not a publisher DOI.
ARXIV_DOI_PREFIX = "10.48550/arxiv."
_ARXIV_NEW = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"^([a-z\-]+(?:\.[a-z]{2})?/\d{7})(v\d+)?$", re.IGNORECASE)
_ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})


def normalise_query(text: str) -> str:
    """Unicode NFKC, casefold, whitespace collapsed: the form a query is cached under.

    Deliberately nothing more. Stemming or stopword removal here would make two
    queries the user typed differently share a cache entry and a result list, which
    is a relevance decision, not a normalisation one.
    """
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", text).casefold()).strip()


class UnsearchableQuery(DomainError):
    """A query with no letter or digit in it: nothing any source could search for.

    The caller's mistake, so a 400 and not a source failure (the discovery port's
    rule): reporting every source as failed would tell the user the sources are
    down when the query was empty.
    """


def is_searchable(query: str) -> bool:
    """Whether a query has at least one letter or digit, in any script, to search for."""
    return any(ch.isalnum() for ch in normalise_query(query))


def is_web_url(url: str) -> bool:
    """An absolute http or https URL with a host: the only kind a source's link may be.

    A link from a third-party record is shown to a browser (D122), and `javascript:`
    or `data:` in a poisoned record would be clickable there. This decides what may
    be shown, not what may be fetched; fetching has its own, stricter guard (D115).
    """
    try:
        parts = urlsplit(url.strip())
        host = parts.hostname
    except ValueError:  # a malformed netloc, such as an unclosed IPv6 bracket
        return False
    return parts.scheme.lower() in ("http", "https") and bool(host)


def _letters_and_digits(text: str) -> str:
    """Every character that is not a letter or digit (in any script) becomes a space."""
    return "".join(ch if ch.isalnum() else " " for ch in text)


def normalise_title(text: str | None) -> str:
    """The title form deduplication compares: punctuation and case erased."""
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", _letters_and_digits(folded)).strip()


def normalise_doi(value: str | None) -> str | None:
    """A DOI lowercased and stripped of resolver prefixes, or None if it is not one.

    DOIs are case-insensitive by specification, so case carries no identity. Every
    DOI starts with the `10.` directory indicator; anything else a source puts in its
    DOI field is not a DOI and must not become an identity a work can be merged on.
    """
    if not value:
        return None
    doi = value.strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
            break
    return doi if doi.startswith("10.") and "/" in doi else None


def normalise_arxiv_id(value: str | None) -> str | None:
    """An arXiv id with no prefix, `.pdf` or version (`2106.09685`), or None.

    The version is dropped because identity is the work, not the revision: v1 and v3
    are the same paper for discovery. The versioned form is still kept where it
    matters, in the PDF link the source supplied (see `OpenAccessLink`).
    """
    if not value:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered.startswith(ARXIV_DOI_PREFIX):
        text = text[len(ARXIV_DOI_PREFIX) :]
    elif lowered.startswith("arxiv:"):
        text = text[len("arxiv:") :]
    elif lowered.startswith(("http://", "https://")):
        return arxiv_id_from_url(text)
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    match = _ARXIV_NEW.match(text.strip("/")) or _ARXIV_OLD.match(text.strip("/"))
    # Lowercased so an old-style id has one spelling (`hep-th/9901001`, never
    # `HEP-TH/9901001`): identity is unique per owner on this value (D132), and a
    # unique index over two spellings of one id enforces nothing. New-style ids
    # are digits and unaffected.
    return match.group(1).lower() if match else None


def paper_identity(doi: str | None, arxiv_id: str | None) -> tuple[str | None, str | None]:
    """A work's (DOI, arXiv id) with one spelling each (D132).

    arXiv's DataCite DOI (`10.48550/arxiv.X`) is another spelling of the arXiv id
    X, not a publisher DOI, so it becomes the arXiv id and the DOI slot is left for
    a publisher's. Without this, OpenAlex (which keeps the DataCite DOI as the DOI)
    and arXiv (which reports the journal DOI or none) would give one work two
    identities, and per-owner uniqueness on each would let a user hold it twice.
    """
    normal_doi = normalise_doi(doi)
    normal_arxiv = normalise_arxiv_id(arxiv_id)
    if normal_doi is not None and normal_doi.startswith(ARXIV_DOI_PREFIX):
        normal_arxiv = normal_arxiv or normalise_arxiv_id(normal_doi)
        normal_doi = None
    return normal_doi, normal_arxiv


def arxiv_id_from_url(url: str | None) -> str | None:
    """The arXiv id in an arxiv.org `/abs/` or `/pdf/` URL, or None for any other URL."""
    if not url:
        return None
    parts = urlsplit(url.strip())
    if (parts.hostname or "").lower() not in _ARXIV_HOSTS:
        return None
    for marker in ("/abs/", "/pdf/"):
        if parts.path.startswith(marker):
            return normalise_arxiv_id(parts.path[len(marker) :])
    return None


class LinkKind(StrEnum):
    """What a source claims an open-access URL points at.

    The claim is the source's, and the spike found it wrong in both directions: 4 of
    6 HTML pages it fetched came from URLs a source called a PDF. Phase 5's fetcher
    decides by the bytes it receives, never by this field; the field records what
    was claimed, so a wrong claim can be counted.
    """

    PDF = "pdf"
    LANDING = "landing"


@dataclass(frozen=True, slots=True)
class OpenAccessLink:
    """An open-access URL as a source supplied it, with the licence it reported.

    The URL is kept exactly as given. For arXiv that means the versioned link from
    the API (`.../pdf/2507.20128v2`), never one rebuilt from the bare id: the spike's
    one arXiv 404 was a rebuilt link to a submission the API itself marked as a
    superseded duplicate.
    """

    url: str
    kind: LinkKind
    licence: str | None = None

    def __post_init__(self) -> None:
        if not is_web_url(self.url):
            # Adapters drop such links before building one (D122), so reaching this
            # means an adapter forgot to, and a poisoned record must not ride on it.
            raise InvariantViolation(
                f"an open-access link must be an http(s) URL, got {self.url!r}"
            )


@dataclass(frozen=True, slots=True)
class PaperCandidate:
    """One search result from one source (FR-D1), before deduplication and ranking.

    Absent is not zero. A source that reports no citation count gives None, and so
    does an unknown year, venue or abstract; arXiv reports no citation counts at all.
    Defaulting any of them to zero or the empty string would read as a measured
    value and silently penalise the paper in ranking (the rule
    `Paper.citation_impact_per_year` states for the same reason).
    """

    #: The source's name, as `AcademicSource.name` reports it.
    source: str
    #: The source's own id for the record (an OpenAlex W-id, a versioned arXiv id).
    source_id: str
    #: 1-based position in the source's own ranking.
    rank: int
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    published: date | None = None
    venue: str | None = None
    abstract: str | None = None
    citation_count: int | None = None
    #: Normalised (`normalise_doi`). May be an arXiv DataCite DOI.
    doi: str | None = None
    #: Unversioned (`normalise_arxiv_id`).
    arxiv_id: str | None = None
    oa_links: tuple[OpenAccessLink, ...] = ()
    #: The source's own relevance score, if it gives one. Not comparable across
    #: sources (OpenAlex's carries a citation weighting it capped in 2026-10, D116,
    #: D126), so ranking must not mix it with another source's.
    source_score: float | None = None
    #: Which adapter build produced this (NFR-M5), so a change in candidates can be
    #: traced to a change in parsing rather than in the source.
    adapter_version: str = ""

    def __post_init__(self) -> None:
        if not self.source.strip() or not self.source_id.strip():
            raise InvariantViolation("a candidate must name its source and the source's id for it")
        if not self.title.strip():
            raise InvariantViolation("a candidate title cannot be blank")
        if self.rank < 1:
            raise InvariantViolation(f"source rank is 1-based, got {self.rank}")
        if self.citation_count is not None and self.citation_count < 0:
            raise InvariantViolation(
                f"citation count cannot be negative, got {self.citation_count}"
            )
        if self.doi is not None and normalise_doi(self.doi) != self.doi:
            raise InvariantViolation(f"candidate DOI must be normalised, got {self.doi!r}")
        if self.arxiv_id is not None and normalise_arxiv_id(self.arxiv_id) != self.arxiv_id:
            raise InvariantViolation(
                f"candidate arXiv id must be normalised and unversioned, got {self.arxiv_id!r}"
            )
        if not self.adapter_version.strip():
            raise InvariantViolation("a candidate must record the adapter version that produced it")


def interleave(pages: Sequence[Sequence[PaperCandidate]]) -> tuple[PaperCandidate, ...]:
    """Each source's first result, then each source's second, and so on (D122).

    The order until phases 7 and 8 replace it: it favours no source and makes no
    relevance claim of its own, and a source with fewer results simply runs out.
    Duplicates across sources are kept; merging them is phase 7's job.
    """
    out: list[PaperCandidate] = []
    longest = max((len(page) for page in pages), default=0)
    for position in range(longest):
        for page in pages:
            if position < len(page):
                out.append(page[position])
    return tuple(out)
