"""Turn each source's raw response into one record shape (protocol: definitions).

Spike-grade on purpose: it reads what the three APIs return in the fields the
protocol names, and nothing more. Phase 2 writes the real adapters, against
fixtures cut from the pools these responses become.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any

from m4_common import arxiv_from_url, norm_arxiv, norm_doi, norm_title

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"


@dataclass
class Rec:
    source: str
    rank: int
    source_id: str
    title: str | None
    n_authors: int
    year: int | None
    venue: str | None
    has_abstract: bool
    citation_count: int | None
    doi: str | None
    arxiv_id: str | None
    full_date: str | None
    relevance: float | None = None
    #: (url, kind, licence) in the order the source lists them.
    oa: list[tuple[str, str, str | None]] = field(default_factory=list)
    #: OpenAlex only: every location as (landing_page_url, pdf_url, licence), and the
    #: best OA location's licence, for evidence_without_a_bar.licence.
    locations: list[tuple[str | None, str | None, str | None]] = field(default_factory=list)
    best_oa_licence: str | None = None

    @property
    def ntitle(self) -> str:
        return norm_title(self.title)


@dataclass
class Parsed:
    records: list[Rec]
    total: int | None
    error: str | None


def _openalex_work(work: dict[str, Any], rank: int) -> Rec:
    work_id = str(work.get("id") or "").rsplit("/", 1)[-1]
    doi = norm_doi(work.get("doi"))
    arxiv = norm_arxiv(doi) if doi and doi.startswith("10.48550/arxiv.") else None
    oa: list[tuple[str, str, str | None]] = []
    best = work.get("best_oa_location") or {}
    if best.get("pdf_url"):
        oa.append((best["pdf_url"], "openalex_best_pdf", best.get("license")))
    for location in work.get("locations") or []:
        location = location or {}
        arxiv = arxiv or arxiv_from_url(location.get("landing_page_url")) or arxiv_from_url(
            location.get("pdf_url")
        )
        if location.get("pdf_url"):
            oa.append((location["pdf_url"], "openalex_location_pdf", location.get("license")))
    oa_url = (work.get("open_access") or {}).get("oa_url")
    if oa_url:
        oa.append((oa_url, "openalex_oa_url", None))
    source = ((work.get("primary_location") or {}).get("source") or {})
    cited = work.get("cited_by_count")
    return Rec(
        source="openalex",
        rank=rank,
        source_id=work_id,
        title=work.get("display_name") or work.get("title"),
        n_authors=len(work.get("authorships") or []),
        year=work.get("publication_year"),
        venue=source.get("display_name"),
        has_abstract=bool(work.get("abstract_inverted_index")),
        citation_count=cited if isinstance(cited, int) else None,
        doi=doi,
        arxiv_id=arxiv,
        full_date=work.get("publication_date"),
        relevance=work.get("relevance_score"),
        oa=oa,
        locations=[
            ((loc or {}).get("landing_page_url"), (loc or {}).get("pdf_url"), (loc or {}).get("license"))
            for loc in work.get("locations") or []
        ],
        best_oa_licence=best.get("license"),
    )


def parse_openalex(text: str) -> Parsed:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return Parsed([], None, f"json: {exc}")
    if isinstance(payload, dict) and "results" in payload:
        works = payload.get("results") or []
        total = (payload.get("meta") or {}).get("count")
        return Parsed([_openalex_work(w, i + 1) for i, w in enumerate(works)], total, None)
    if isinstance(payload, dict) and payload.get("id"):
        return Parsed([_openalex_work(payload, 1)], 1, None)
    return Parsed([], None, str(payload)[:300])


def _s2_paper(paper: dict[str, Any], rank: int) -> Rec:
    external = paper.get("externalIds") or {}
    pdf = paper.get("openAccessPdf") or {}
    oa: list[tuple[str, str, str | None]] = []
    if pdf.get("url"):
        oa.append((pdf["url"], "s2_pdf", pdf.get("license")))
    venue = paper.get("venue") or (paper.get("publicationVenue") or {}).get("name") or (
        (paper.get("journal") or {}).get("name")
    )
    cited = paper.get("citationCount")
    return Rec(
        source="semantic_scholar",
        rank=rank,
        source_id=str(paper.get("paperId") or ""),
        title=paper.get("title"),
        n_authors=len(paper.get("authors") or []),
        year=paper.get("year"),
        venue=venue or None,
        has_abstract=bool(paper.get("abstract")),
        citation_count=cited if isinstance(cited, int) else None,
        doi=norm_doi(external.get("DOI")),
        arxiv_id=norm_arxiv(external.get("ArXiv")),
        full_date=paper.get("publicationDate"),
        oa=oa,
    )


def parse_s2(text: str) -> Parsed:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return Parsed([], None, f"json: {exc}")
    if isinstance(payload, dict) and "data" in payload:
        papers = payload.get("data") or []
        return Parsed([_s2_paper(p, i + 1) for i, p in enumerate(papers)], payload.get("total"), None)
    if isinstance(payload, list):  # the batch endpoint: one entry per id, null if unknown
        return Parsed(
            [_s2_paper(p, i + 1) for i, p in enumerate(payload) if isinstance(p, dict)],
            len(payload),
            None,
        )
    if isinstance(payload, dict) and payload.get("paperId"):
        return Parsed([_s2_paper(payload, 1)], 1, None)
    return Parsed([], None, str(payload)[:300])


def parse_arxiv(text: str) -> Parsed:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        return Parsed([], None, f"xml: {exc}")
    total_text = root.findtext(f"{OPENSEARCH}totalResults")
    total = int(total_text) if total_text and total_text.strip().isdigit() else None
    records: list[Rec] = []
    for index, entry in enumerate(root.findall(f"{ATOM}entry")):
        entry_id = entry.findtext(f"{ATOM}id") or ""
        title = " ".join((entry.findtext(f"{ATOM}title") or "").split())
        if "/api/errors" in entry_id or (title == "Error" and not norm_arxiv(entry_id)):
            return Parsed([], total, (entry.findtext(f"{ATOM}summary") or "error").strip()[:300])
        arxiv = norm_arxiv(entry_id)
        published = entry.findtext(f"{ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        pdf = next(
            (
                link.get("href")
                for link in entry.findall(f"{ATOM}link")
                if link.get("title") == "pdf" and link.get("href")
            ),
            None,
        )
        oa = [(pdf, "arxiv_pdf_link", None)] if pdf else []
        records.append(
            Rec(
                source="arxiv",
                rank=index + 1,
                source_id=arxiv or entry_id,
                title=title or None,
                n_authors=len(entry.findall(f"{ATOM}author")),
                year=year,
                venue=(entry.findtext(f"{ARXIV_NS}journal_ref") or "").strip() or None,
                has_abstract=bool((entry.findtext(f"{ATOM}summary") or "").strip()),
                citation_count=None,
                doi=norm_doi(entry.findtext(f"{ARXIV_NS}doi")),
                arxiv_id=arxiv,
                full_date=published or None,
                oa=oa,
            )
        )
    return Parsed(records, total, None)


PARSERS = {"openalex": parse_openalex, "semantic_scholar": parse_s2, "arxiv": parse_arxiv}


def parse(source: str, text: str) -> Parsed:
    return PARSERS[source](text)
