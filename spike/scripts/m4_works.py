"""The P3 work list: search results merged into unique works (D116).

Shared by 18_oa_probe.py, which probes the works, and 19_discovery_analysis.py,
which reports on them, so both read the same works in the same order.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from m4_common import RUN_DIR, norm_arxiv, read_jsonl
from m4_parse import Rec, parse

TOP_N = 10


@dataclass
class Candidate:
    url: str
    kind: str
    supplied_by: set[str] = field(default_factory=set)
    licences: dict[str, str | None] = field(default_factory=dict)


@dataclass
class Work:
    key: str
    records: list[Rec]
    topics: set[str]
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def sources(self) -> set[str]:
        return {r.source for r in self.records}


def search_records(
    pass_id: str, phrasing: str, top_n: int | None = None
) -> dict[str, list[Rec]]:
    """One pass's successful search results for one phrasing, keyed by topic."""
    by_topic: dict[str, list[Rec]] = {}
    for entry in read_jsonl(RUN_DIR / "bodies.jsonl"):
        if entry.get("pass") != pass_id or entry.get("phrasing") != phrasing:
            continue
        if entry.get("status") != 200 or entry.get("kind") != "search":
            continue
        parsed = parse(entry["source"], entry["body"])
        by_topic.setdefault(entry["topic"], []).extend(
            r for r in parsed.records if top_n is None or r.rank <= top_n
        )
    return by_topic


def p1_keyword_records() -> dict[str, list[Rec]]:
    """P3's input: session-1 keyword results, each source's ranks 1-10 (pdf_probe.works)."""
    return search_records("P1", "keyword", TOP_N)


def record_keys(rec: Rec) -> list[str]:
    """The identifiers a record is merged on: arXiv id (from any form) and DOI."""
    keys = []
    doi_is_arxiv = bool(rec.doi and rec.doi.startswith("10.48550/arxiv."))
    arxiv = rec.arxiv_id or (norm_arxiv(rec.doi) if doi_is_arxiv else None)
    if arxiv:
        keys.append(f"arxiv:{arxiv}")
    if rec.doi and not doi_is_arxiv:
        keys.append(f"doi:{rec.doi}")
    return keys


def build_works(by_topic: dict[str, list[Rec]]) -> list[Work]:
    """Union-find over shared DOI or arXiv id (definitions.unique_work)."""
    items = [(topic, rec) for topic in sorted(by_topic) for rec in by_topic[topic]]
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first: dict[str, int] = {}
    for i, (_, rec) in enumerate(items):
        for key in record_keys(rec):
            if key in first:
                parent[find(i)] = find(first[key])
            else:
                first[key] = i
    groups: dict[int, list[int]] = {}
    for i in range(len(items)):
        groups.setdefault(find(i), []).append(i)

    works = []
    for members in groups.values():
        recs = [items[i][1] for i in members]
        keys = sorted({k for r in recs for k in record_keys(r)})
        key = keys[0] if keys else f"{recs[0].source}:{recs[0].source_id}"
        work = Work(key, sorted(recs, key=lambda r: (r.source, r.rank)), {items[i][0] for i in members})
        work.candidates = candidates(work)
        works.append(work)
    return sorted(works, key=lambda w: w.key)


def candidates(work: Work) -> list[Candidate]:
    """Candidate URLs in the protocol's pdf_probe.url_order, de-duplicated."""
    out: list[Candidate] = []

    def add(url: str | None, kind: str, source: str, licence: str | None) -> None:
        if not url:
            return
        url = url.strip()
        for cand in out:
            if cand.url == url:
                cand.supplied_by.add(source)
                cand.licences.setdefault(source, licence)
                return
        out.append(Candidate(url, kind, {source}, {source: licence}))

    arxiv_ids = sorted(
        {k.split(":", 1)[1] for r in work.records for k in record_keys(r) if k.startswith("arxiv:")}
    )
    if arxiv_ids:
        suppliers = {r.source for r in work.records if f"arxiv:{arxiv_ids[0]}" in record_keys(r)}
        for source in sorted(suppliers):
            add(f"https://arxiv.org/pdf/{arxiv_ids[0]}", "arxiv_pdf", source, None)
    openalex = [r for r in work.records if r.source == "openalex"]
    for rec in openalex:
        for url, kind, licence in rec.oa:
            if kind == "openalex_best_pdf":
                add(url, kind, "openalex", licence)
    for rec in openalex:
        for url, kind, licence in rec.oa:
            if kind == "openalex_location_pdf":
                add(url, kind, "openalex", licence)
    for rec in (r for r in work.records if r.source == "semantic_scholar"):
        for url, kind, licence in rec.oa:
            add(url, kind, "semantic_scholar", licence)
    for rec in openalex:
        for url, kind, licence in rec.oa:
            if kind == "openalex_oa_url":
                add(url, kind, "openalex", licence)
    return out


def ordered(protocol: dict[str, Any], works: list[Work], stream: str) -> list[Work]:
    """Works in a seeded order, from a named stream of the protocol's seed."""
    rng = random.Random(f"{protocol['seed']}/{stream}")
    result = list(works)
    rng.shuffle(result)
    return result
