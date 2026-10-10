"""OpenAlex as an `AcademicSource` (D118, D119).

**The primary source, on measured grounds** (D116, D118): in the spike it met three
of ASM-1's four bars, returning full metadata (title, year and citations on every
record, abstracts on 92%) with zero 429s on a key. It missed only latency, which
addendum A2 settled (D125). Its ranking changed between the spike and A2, finding
6 of 20 anchors where it had found 14 (D126); that is phase 8's to answer.

**The key is required and travels only in a header.** OpenAlex has required an API
key since 13 February 2026, beyond a test allowance. A URL ends up in access logs,
proxies and tracebacks; an Authorization header does not, which is why the spike
sent it that way and why a test here pins it.

**`?` and `*` never reach OpenAlex** (D117). Its default search is stemmed and
refuses both as wildcards with HTTP 400, so every question a user types as a
question would fail. They become spaces, as the spike's amended protocol did.

**The field list is the spike's, and the page size is the caller's** (D125). Author
lists are half of every page and `select` cannot trim inside a field, and dropping
`locations` would lose the arXiv id phase 7 links sources by on 18% of records, so
the fields stay. The caller asks for 25 results by default: at that size A2 measured
p95 1.58 s against D122's 2.0 s deadline, and a recorded URL and the one built here
can still be compared.

**A lookup is the same request with a filter instead of a search** (D133):
`filter=openalex:W1|W2`, one request for a whole batch. Measured on 2026-10-07: an
id OpenAlex does not know is simply absent from the results (`W12`, `W9999999999`),
while one it considers malformed (`W1`) fails the WHOLE request with a 400, which
is why `is_valid_id` insists on two digits or more before anything is sent.
`relevance_score` is left out of the lookup's fields, because there is no search
to score against.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import httpx

from scholiast.application.ports.discovery import (
    MAX_LOOKUP_IDS,
    SourceFailure,
    SourcePage,
    SourceUnavailable,
)
from scholiast.domain.discovery import (
    ARXIV_DOI_PREFIX,
    LinkKind,
    OpenAccessLink,
    PaperCandidate,
    arxiv_id_from_url,
    is_web_url,
    normalise_arxiv_id,
    normalise_doi,
    normalise_query,
)
from scholiast.infrastructure.discovery import USER_AGENT, tls_context
from scholiast.infrastructure.retry_after import retry_after_seconds

SOURCE_NAME = "openalex"
#: 2: links that are not http(s) URLs are dropped (D122).
ADAPTER_VERSION = "openalex-2"
DEFAULT_BASE_URL = "https://api.openalex.org"
#: The spike's field list (D116), kept by addendum A2 (D125).
SELECT = (
    "id,doi,title,display_name,publication_year,publication_date,ids,type,authorships,"
    "primary_location,locations,best_oa_location,open_access,cited_by_count,"
    "abstract_inverted_index,relevance_score"
)
#: A lookup's fields: the search's, less the score a lookup has no query for.
LOOKUP_SELECT = ",".join(field for field in SELECT.split(",") if field != "relevance_score")
#: OpenAlex's own ceiling for one page.
MAX_PER_PAGE = 200
#: A work id as search returns it: `W` and at least two digits. OpenAlex refuses
#: `W1` with a 400 for the whole request (measured 2026-10-07), so one digit is
#: refused here, where it costs the caller a 422 instead of the source a failure.
_WORK_ID = re.compile(r"^W[1-9][0-9]{1,11}$")
_WILDCARDS = re.compile(r"[?*]")
_WHITESPACE = re.compile(r"\s+")


def openalex_search_text(query: str) -> str:
    """The `search` parameter: the normalised query with OpenAlex's wildcards removed (D117)."""
    return _WHITESPACE.sub(" ", _WILDCARDS.sub(" ", normalise_query(query))).strip()


class OpenAlexSource:
    """The OpenAlex `AcademicSource`."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: httpx.Timeout | float = httpx.Timeout(20.0, connect=5.0),
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAlex requires an API key (D116); none was given")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return SOURCE_NAME

    def request(self, query: str, *, limit: int) -> httpx.Request:
        """The request `search` sends, built without sending it."""
        text = openalex_search_text(query)
        if not text:
            raise ValueError("the query has nothing OpenAlex can search for")
        if not 1 <= limit <= MAX_PER_PAGE:
            raise ValueError(f"limit must be 1..{MAX_PER_PAGE}, got {limit}")
        return httpx.Request(
            "GET",
            f"{self._base_url}/works",
            params={"search": text, "per_page": limit, "select": SELECT},
            headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT},
        )

    async def search(self, query: str, *, limit: int) -> SourcePage:
        return parse_works(await self._send(self.request(query, limit=limit)))

    def is_valid_id(self, source_id: str) -> bool:
        return bool(_WORK_ID.match(source_id))

    def lookup_request(self, source_ids: Sequence[str]) -> httpx.Request:
        """The request `lookup` sends, built without sending it."""
        if not 1 <= len(source_ids) <= MAX_LOOKUP_IDS:
            raise ValueError(f"a lookup carries 1..{MAX_LOOKUP_IDS} ids, got {len(source_ids)}")
        invalid = [source_id for source_id in source_ids if not self.is_valid_id(source_id)]
        if invalid:
            raise ValueError(f"not OpenAlex work ids: {invalid!r}")
        unique = list(dict.fromkeys(source_ids))
        return httpx.Request(
            "GET",
            f"{self._base_url}/works",
            params={
                "filter": "openalex:" + "|".join(unique),
                "per_page": len(unique),
                "select": LOOKUP_SELECT,
            },
            headers={"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT},
        )

    async def lookup(self, source_ids: Sequence[str]) -> Mapping[str, PaperCandidate]:
        page = parse_works(await self._send(self.lookup_request(source_ids)))
        found = {candidate.source_id: candidate for candidate in page.candidates}
        return {source_id: found[source_id] for source_id in source_ids if source_id in found}

    async def _send(self, request: httpx.Request) -> Any:
        """Send one request and return its JSON, or raise `SourceUnavailable`."""
        try:
            async with httpx.AsyncClient(
                timeout=self._timeout, follow_redirects=False, verify=tls_context()
            ) as client:
                response = await client.send(request)
        except httpx.TimeoutException as exc:
            raise SourceUnavailable(SOURCE_NAME, SourceFailure.TIMEOUT, type(exc).__name__) from exc
        except httpx.HTTPError as exc:
            # The exception names the URL at most, never the headers, so the key
            # cannot leak through this message.
            raise SourceUnavailable(
                SOURCE_NAME, SourceFailure.ERROR, f"{type(exc).__name__}: {exc}"[:200]
            ) from exc
        if response.status_code == 429:
            raise SourceUnavailable(
                SOURCE_NAME,
                SourceFailure.RATE_LIMITED,
                "HTTP 429",
                retry_after=retry_after_seconds(response),
            )
        if response.status_code != 200:
            raise SourceUnavailable(
                SOURCE_NAME,
                SourceFailure.ERROR,
                f"HTTP {response.status_code}: {_error_message(response)}",
                retry_after=retry_after_seconds(response),
            )
        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise SourceUnavailable(
                SOURCE_NAME, SourceFailure.ERROR, "response is not JSON"
            ) from exc
        return payload


def _error_message(response: httpx.Response) -> str:
    """OpenAlex's own explanation, which for a 400 says exactly what was wrong."""
    try:
        payload = response.json()
    except json.JSONDecodeError:
        return response.text[:200]
    if isinstance(payload, dict):
        return str(payload.get("message") or payload.get("error") or "")[:200]
    return str(payload)[:200]


def parse_works(payload: Any) -> SourcePage:
    """A `/works` search response to a page of candidates."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise SourceUnavailable(SOURCE_NAME, SourceFailure.ERROR, "response has no results list")
    candidates = []
    skipped = 0
    for index, work in enumerate(payload["results"]):
        candidate = _candidate(work, index + 1) if isinstance(work, dict) else None
        if candidate is None:
            skipped += 1
        else:
            candidates.append(candidate)
    count = (payload.get("meta") or {}).get("count")
    return SourcePage(
        SOURCE_NAME, tuple(candidates), count if isinstance(count, int) else None, skipped
    )


def _candidate(work: dict[str, Any], rank: int) -> PaperCandidate | None:
    title = (work.get("display_name") or work.get("title") or "").strip()
    work_id = str(work.get("id") or "").rsplit("/", 1)[-1]
    if not title or not work_id:
        return None
    doi = normalise_doi(work.get("doi"))
    locations = [loc for loc in work.get("locations") or [] if isinstance(loc, dict)]
    arxiv_id = normalise_arxiv_id(doi) if doi and doi.startswith(ARXIV_DOI_PREFIX) else None
    for location in locations:
        arxiv_id = (
            arxiv_id
            or arxiv_id_from_url(location.get("landing_page_url"))
            or arxiv_id_from_url(location.get("pdf_url"))
        )
    source = (work.get("primary_location") or {}).get("source") or {}
    cited = work.get("cited_by_count")
    score = work.get("relevance_score")
    return PaperCandidate(
        source=SOURCE_NAME,
        source_id=work_id,
        rank=rank,
        title=title,
        authors=tuple(
            name
            for authorship in work.get("authorships") or []
            if (name := ((authorship or {}).get("author") or {}).get("display_name"))
        ),
        year=work.get("publication_year")
        if isinstance(work.get("publication_year"), int)
        else None,
        published=_date(work.get("publication_date")),
        venue=source.get("display_name") or None,
        abstract=_abstract(work.get("abstract_inverted_index")),
        # Absent is not zero (PaperCandidate): only an integer the source sent counts.
        citation_count=cited if isinstance(cited, int) and cited >= 0 else None,
        doi=doi,
        arxiv_id=arxiv_id,
        oa_links=_oa_links(work, locations),
        source_score=float(score) if isinstance(score, int | float) else None,
        adapter_version=ADAPTER_VERSION,
    )


def _oa_links(work: dict[str, Any], locations: list[dict[str, Any]]) -> tuple[OpenAccessLink, ...]:
    """Every PDF URL OpenAlex offers, best location first, then `oa_url` as a landing page.

    `oa_url` is a landing page unless it repeats a PDF URL already listed: the spike
    found both `oa_url` HTML pages it fetched were landing pages, never offered as PDFs.
    """
    links: list[OpenAccessLink] = []
    seen: set[str] = set()

    def add(url: Any, kind: LinkKind, licence: Any) -> None:
        # Only an http(s) link may reach a browser (D122): a `javascript:` URL in a
        # poisoned record is dropped here, and the rest of the record is kept.
        if isinstance(url, str) and is_web_url(url) and url.strip() not in seen:
            seen.add(url.strip())
            links.append(
                OpenAccessLink(url.strip(), kind, licence if isinstance(licence, str) else None)
            )

    best = work.get("best_oa_location") or {}
    add(best.get("pdf_url"), LinkKind.PDF, best.get("license"))
    for location in locations:
        add(location.get("pdf_url"), LinkKind.PDF, location.get("license"))
    add((work.get("open_access") or {}).get("oa_url"), LinkKind.LANDING, None)
    return tuple(links)


def _abstract(index: Any) -> str | None:
    """OpenAlex ships abstracts as {word: [positions]}; this puts the words back in order."""
    if not isinstance(index, dict) or not index:
        return None
    placed: dict[int, str] = {}
    for word, positions in index.items():
        for position in positions if isinstance(positions, list) else []:
            if isinstance(position, int):
                placed[position] = word
    text = " ".join(placed[i] for i in sorted(placed))
    return text or None


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None
