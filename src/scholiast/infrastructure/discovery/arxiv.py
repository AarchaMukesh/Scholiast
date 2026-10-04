"""arXiv as an `AcademicSource` (D118, D119, D120).

**Secondary source, and the one whose recall depends on how we ask** (D120). In the
spike it was fast (p95 1.6 s) and complete, and missed ASM-1's relevance bar because
an AND of every content word excluded canonical papers that lack one of the words.
The join is AND, which addendum A1 kept: joined with OR, the same queries put fewer
anchors in the top 50 (4 of 20 against 6) and matched up to two million papers each.

**Terms, not text.** The query is cut into letter-and-digit words, stopwords and
one-character words dropped, each searched as `all:<word>`. Nothing but a word ever
reaches arXiv's query syntax, so a user cannot inject a field prefix, a parenthesis
or an operator: `ti:`, `(` and `ANDNOT` are split, dropped or casefolded away.

**Errors arrive as a 200.** arXiv reports a bad query as an Atom feed whose single
entry is the error message, with HTTP 200. Read naively that is one odd result; read
as zero results it tells the user there are no papers. It raises instead (D119).

**The PDF link is the one arXiv gives, versioned** (D119). The spike rebuilt links
from bare ids and its one arXiv 404 was a rebuilt link to a submission the API itself
marked as a superseded duplicate.

**XML from an outside service.** Parsed with ElementTree, which never resolves
external entities, over expat 2.6, which refuses entity-amplification bombs; a test
feeds it one. The GROBID adapter's reason ("our own service, our own network") does
not apply here, so this is the reason instead.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from typing import Final

import httpx

from scholiast.application.ports.discovery import SourceFailure, SourcePage, SourceUnavailable
from scholiast.domain.discovery import (
    LinkKind,
    OpenAccessLink,
    PaperCandidate,
    is_web_url,
    normalise_arxiv_id,
    normalise_doi,
    normalise_query,
)
from scholiast.infrastructure.discovery import USER_AGENT, tls_context
from scholiast.infrastructure.retry_after import retry_after_seconds

SOURCE_NAME = "arxiv"
#: 2: links that are not http(s) URLs are dropped (D122).
ADAPTER_VERSION = "arxiv-2"
DEFAULT_BASE_URL = "https://export.arxiv.org/api/query"
#: How the words of a query are joined. AND, kept by addendum A1 (D120); OR is
#: accepted by the constructor so a later measurement can try it again.
DEFAULT_JOIN: Final = "AND"
#: arXiv's ceiling for one request.
MAX_RESULTS = 2000
#: The spike's stopword list (D116, query_construction.arxiv). A test pins it to the
#: protocol, so the construction measured and the construction shipped cannot drift.
STOPWORDS: Final = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "by",
        "for",
        "from",
        "with",
        "over",
        "into",
        "via",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "may",
        "might",
        "how",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "when",
        "where",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "their",
        "there",
    ]
)

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"
_OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
_WORD = re.compile(r"[^\W_]+")
_WHITESPACE = re.compile(r"\s+")
_ABS_ID = re.compile(r"/abs/(.+)$")


def arxiv_terms(query: str) -> list[str]:
    """The words searched: letters and digits only, no stopwords, no repeats, first-seen order."""
    seen: set[str] = set()
    terms = []
    for word in _WORD.findall(normalise_query(query)):
        if len(word) < 2 or word in STOPWORDS or word in seen:
            continue
        seen.add(word)
        terms.append(word)
    return terms


class ArxivSource:
    """The arXiv `AcademicSource`."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        join: str = DEFAULT_JOIN,
        timeout: httpx.Timeout | float = httpx.Timeout(20.0, connect=5.0),
    ) -> None:
        if join not in ("AND", "OR"):
            raise ValueError(f"join must be AND or OR, got {join!r}")
        self._base_url = base_url
        self._join = join
        self._timeout = timeout

    @property
    def name(self) -> str:
        return SOURCE_NAME

    def request(self, query: str, *, limit: int) -> httpx.Request:
        """The request `search` sends, built without sending it."""
        terms = arxiv_terms(query)
        if not terms:
            raise ValueError("the query has no word arXiv can search for")
        if not 1 <= limit <= MAX_RESULTS:
            raise ValueError(f"limit must be 1..{MAX_RESULTS}, got {limit}")
        return httpx.Request(
            "GET",
            self._base_url,
            params={
                "search_query": f" {self._join} ".join(f"all:{t}" for t in terms),
                "start": 0,
                "max_results": limit,
                "sortBy": "relevance",
                "sortOrder": "descending",
            },
            headers={"User-Agent": USER_AGENT},
        )

    async def search(self, query: str, *, limit: int) -> SourcePage:
        request = self.request(query, limit=limit)
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False, verify=tls_context()
            ) as client:
                response = await client.send(request)
        except httpx.TimeoutException as exc:
            raise SourceUnavailable(SOURCE_NAME, SourceFailure.TIMEOUT, type(exc).__name__) from exc
        except httpx.HTTPError as exc:
            raise SourceUnavailable(
                SOURCE_NAME, SourceFailure.ERROR, f"{type(exc).__name__}: {exc}"[:200]
            ) from exc
        # arXiv throttles with 503 as well as 429, and says when to come back.
        if response.status_code in (429, 503):
            raise SourceUnavailable(
                SOURCE_NAME,
                SourceFailure.RATE_LIMITED,
                f"HTTP {response.status_code}",
                retry_after=retry_after_seconds(response),
            )
        if response.status_code != 200:
            raise SourceUnavailable(
                SOURCE_NAME,
                SourceFailure.ERROR,
                f"HTTP {response.status_code}: {response.text[:200]}",
            )
        return parse_feed(response.content)


def parse_feed(body: bytes | str) -> SourcePage:
    """An Atom search response to a page of candidates; an error feed raises."""
    try:
        root = ET.fromstring(body)  # noqa: S314 - see the module docstring: expat 2.6, no external entities
    except ET.ParseError as exc:
        raise SourceUnavailable(
            SOURCE_NAME, SourceFailure.ERROR, f"unparseable feed: {exc}"
        ) from exc
    entries = root.findall(f"{_ATOM}entry")
    for entry in entries:
        if "/api/errors" in (entry.findtext(f"{_ATOM}id") or ""):
            message = _clean(entry.findtext(f"{_ATOM}summary")) or "arXiv reported an error"
            raise SourceUnavailable(SOURCE_NAME, SourceFailure.ERROR, message[:200])
    candidates = []
    skipped = 0
    for index, entry in enumerate(entries):
        candidate = _candidate(entry, index + 1)
        if candidate is None:
            skipped += 1
        else:
            candidates.append(candidate)
    total = (root.findtext(f"{_OPENSEARCH}totalResults") or "").strip()
    return SourcePage(
        SOURCE_NAME, tuple(candidates), int(total) if total.isdigit() else None, skipped
    )


def _candidate(entry: ET.Element, rank: int) -> PaperCandidate | None:
    match = _ABS_ID.search(entry.findtext(f"{_ATOM}id") or "")
    versioned = match.group(1).strip("/") if match else ""
    arxiv_id = normalise_arxiv_id(versioned)
    title = _clean(entry.findtext(f"{_ATOM}title"))
    if not arxiv_id or not title:
        return None
    published = _date(entry.findtext(f"{_ATOM}published"))
    pdf = next(
        (
            link.get("href")
            for link in entry.findall(f"{_ATOM}link")
            # Only an http(s) link may reach a browser (D122); anything else is dropped.
            if link.get("title") == "pdf" and is_web_url(link.get("href") or "")
        ),
        None,
    )
    return PaperCandidate(
        source=SOURCE_NAME,
        source_id=versioned,
        rank=rank,
        title=title,
        authors=tuple(
            name
            for author in entry.findall(f"{_ATOM}author")
            if (name := _clean(author.findtext(f"{_ATOM}name")))
        ),
        year=published.year if published else None,
        published=published,
        venue=_clean(entry.findtext(f"{_ARXIV}journal_ref")) or None,
        abstract=_clean(entry.findtext(f"{_ATOM}summary")) or None,
        # arXiv reports no citation counts. None, never 0 (PaperCandidate).
        citation_count=None,
        doi=normalise_doi(entry.findtext(f"{_ARXIV}doi")),
        arxiv_id=arxiv_id,
        oa_links=(OpenAccessLink(pdf, LinkKind.PDF),) if pdf else (),
        adapter_version=ADAPTER_VERSION,
    )


def _clean(text: str | None) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def _date(value: str | None) -> date | None:
    try:
        return datetime.fromisoformat((value or "").strip().replace("Z", "+00:00")).date()
    except ValueError:
        return None
