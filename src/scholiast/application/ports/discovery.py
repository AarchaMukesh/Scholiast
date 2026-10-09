"""Discovery port: one interface over "an academic source that answers a topic query" (D119).

FR-D2 asks for candidates from at least one external source, and FR-D6 for partial
results when one fails. Both are only implementable if every source looks the same
at the call site: phase 3's fan-out asks each source the same question and treats
each failure the same way, whichever source it was.

Two things are in the contract rather than left to adapters:

  failure is typed   a source that cannot answer raises `SourceUnavailable` with a
                     kind from a fixed set, never an empty page. An empty page means
                     "the source searched and found nothing"; an error rendered as
                     one would tell the user there are no papers when the truth is
                     that the source did not answer (FR-D6 needs the difference).

  retry_after        a number, not prose (D103). A 429 or a 503 that names a wait is
                     the source stating when it will accept work again, and phase 3's
                     per-source breaker opens for exactly that window (D104).

  sent               whether a request reached the source at all. Our own arXiv
                     pacing (D123) and an open breaker refuse without asking, and a
                     breaker must not count those against the source's health, or a
                     burst of searches would open arXiv's breaker for nothing arXiv did.

  lookup by id       phase 6's add resolves `{source, source_id}` pairs exactly as
                     search returned them (D133). The client never sends a URL or
                     metadata, so the record a paper is created from is the source's
                     own, fetched fresh, and the id is checked against the source's
                     syntax before anything is sent.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from scholiast.domain.discovery import PaperCandidate


class SourceFailure(StrEnum):
    """Why a source did not answer: the per-source statuses phase 3 reports (FR-D6, D122)."""

    TIMEOUT = "timeout"
    #: The source's own limit, or our pacing of it (D123): in both, its limit and
    #: not its health is why nothing came back.
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    #: Not asked: its breaker is open (D104). Distinct from ERROR because nothing
    #: was sent, so it says nothing new about the source.
    CIRCUIT_OPEN = "circuit_open"


class SourceUnavailable(Exception):
    """This source cannot answer this search right now.

    Distinct from a bad query: it means "report this source as failed and carry on
    with the others", so the fan-out catches it and the caller sees a per-source
    status rather than an exception.
    """

    def __init__(
        self,
        source: str,
        kind: SourceFailure,
        detail: str,
        *,
        retry_after: float | None = None,
        sent: bool = True,
    ) -> None:
        super().__init__(f"{source}: {kind}: {detail}")
        self.source = source
        self.kind = kind
        self.detail = detail
        #: Seconds the source asked us to wait, when it said so at all. None means it
        #: said nothing, not "retry immediately".
        self.retry_after = retry_after
        #: False when the refusal was ours and no request reached the source.
        self.sent = sent


@dataclass(frozen=True, slots=True)
class SourcePage:
    """One source's answer to one search."""

    source: str
    #: In the source's own order; `PaperCandidate.rank` restates it.
    candidates: tuple[PaperCandidate, ...]
    #: How many works the source says matched, when it says (arXiv and OpenAlex do).
    total_hits: int | None = None
    #: Records the source returned that could not become candidates (no title, say).
    #: Counted rather than dropped silently, so a parser that starts discarding half a
    #: page shows up as a number instead of as quietly worse results.
    skipped: int = 0


@runtime_checkable
class AcademicSource(Protocol):
    """An external academic search service (FR-D2)."""

    @property
    def name(self) -> str:
        """Stable identifier, used in per-source status and on every candidate."""
        ...

    async def search(self, query: str, *, limit: int) -> SourcePage:
        """Up to `limit` candidates for a free-text topic query.

        Raises `SourceUnavailable` when the source does not answer, and `ValueError`
        for a query with nothing searchable in it, which is the caller's mistake and
        must not be reported as the source failing.
        """
        ...

    def is_valid_id(self, source_id: str) -> bool:
        """Whether `source_id` has this source's id syntax (D133). Sends nothing.

        Strict on purpose: an id the source itself would reject fails a whole
        batched request, and that failure would read as the source being down.
        """
        ...

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        """The source's records for `source_ids`, in one request (D133).

        Keyed by the id exactly as given. An id the source does not know is absent,
        which is how both sources answer one: there is no error to tell apart from
        an outage. Raises `SourceUnavailable` when the source does not answer, and
        `ValueError`, before sending anything, for an id `is_valid_id` refuses or
        for more ids than one request may carry (`MAX_LOOKUP_IDS`).
        """
        ...


#: The most ids one lookup carries: the add endpoint's batch (D133). Both sources
#: allow far more in one request, so this bounds a caller, not a source.
MAX_LOOKUP_IDS = 10


# -- the cache (D127) -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CacheKey:
    """What makes two requests to one source the same request, for one user (D127).

    Per user, so one user's searches never affect another's. The adapter version
    and page size are in it so a change to either misses the old entries rather
    than serving pages a different request produced.
    """

    #: The user the entry belongs to.
    scope: str
    source: str
    #: The adapter build that produced the page (NFR-M5).
    version: str
    page_size: int
    #: `normalise_query`'s form: Unicode form, case and whitespace only.
    query: str


@dataclass(frozen=True, slots=True)
class CachedPage:
    page: SourcePage
    #: When the source answered; how old the page is, which the response reports.
    fetched_at: datetime


@runtime_checkable
class DiscoveryCache(Protocol):
    """Each source's page, kept for a while so a repeated search asks nobody (FR-D7).

    **A cache may turn a source call into a miss, never into a failure.** An
    implementation that cannot reach its store returns None and stores nothing,
    and must not raise; discovery has to work with the cache gone.
    """

    async def get(self, key: CacheKey) -> CachedPage | None:
        """The page stored under `key`, or None on a miss or any failure."""
        ...

    async def put(self, key: CacheKey, page: SourcePage, fetched_at: datetime) -> None:
        """Store a page a source answered with. Only answers are stored, never failures."""
        ...


@dataclass(frozen=True, slots=True)
class VectorKey:
    """What makes two embeddings of a work the same embedding, for one user (D153).

    The text's hash is in it because a new adapter build can change a work's
    abstract, and a vector of the old text would rank the new one.
    """

    #: The user the entry belongs to, as D127's pages are.
    scope: str
    #: The embedder build that produced the vector (NFR-M5).
    embedder: str
    #: The work's add record, `source:source_id`.
    record: str
    #: SHA-256 of the exact text embedded.
    text_sha256: str


@runtime_checkable
class VectorCache(Protocol):
    """Each work's dense vector, kept so a repeated search embeds only its query (D153).

    The page cache's rule holds here too: a failure is a miss, never an error.
    """

    async def get_vectors(self, keys: Sequence[VectorKey]) -> list[list[float] | None]:
        """One entry per key, in order: the vector, or None on a miss or any failure."""
        ...

    async def put_vectors(self, entries: Sequence[tuple[VectorKey, Sequence[float]]]) -> None:
        """Store vectors just computed. Never raises."""
        ...
