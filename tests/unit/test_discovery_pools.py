"""Phase 2's acceptance test: the production adapters against the spike's frozen pools (D119).

The spike measured ASM-1b's completeness by parsing its recorded responses with its
own throwaway parser, and committed both the responses (data/eval/discovery-pools)
and the numbers (spike/results/19_discovery_analysis.json). Replaying the same
responses through the production adapters must reproduce those numbers exactly,
record for record. A field the adapters lose, or invent, shows up as a mismatch
against a committed measurement rather than as a quietly different ranking later.

The second half pins the requests: for every recorded query, the adapter must build
the URL the pools were recorded with, so the pools stay a valid judging set for
phase 8 until a deliberate, re-recorded change (the pools caveat in D116).

Offline: everything read here is committed to the repository.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from scholiast.application.ports.discovery import SourcePage
from scholiast.domain.discovery import ARXIV_DOI_PREFIX, PaperCandidate
from scholiast.infrastructure.discovery import arxiv as arxiv_adapter
from scholiast.infrastructure.discovery.arxiv import ArxivSource, parse_feed
from scholiast.infrastructure.discovery.openalex import OpenAlexSource, parse_works

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "data" / "eval" / "discovery-pools.json.gz"
POOLS_A1 = ROOT / "data" / "eval" / "discovery-pools-arxiv-a1.json.gz"
ANALYSIS = ROOT / "spike" / "results" / "19_discovery_analysis.json"
PROTOCOL = ROOT / "spike" / "m4-protocol.json"
FIXTURES = ROOT / "tests" / "fixtures" / "discovery"

#: ASM-1b's fields, defined as the spike defined them (19_discovery_analysis.FIELDS).
FIELDS: dict[str, Callable[[PaperCandidate], bool]] = {
    "title": lambda c: bool(c.title),
    "authors": lambda c: len(c.authors) >= 1,
    "year": lambda c: c.year is not None,
    "venue": lambda c: bool(c.venue),
    "abstract": lambda c: bool(c.abstract),
    "citation_count": lambda c: c.citation_count is not None,
    "doi": lambda c: bool(c.doi) and not (c.doi or "").startswith(ARXIV_DOI_PREFIX),
    "arxiv_id": lambda c: c.arxiv_id is not None,
    "oa_link": lambda c: bool(c.oa_links),
    "full_date": lambda c: c.published is not None,
}
PARSERS: dict[str, Callable[[str], SourcePage]] = {
    "openalex": lambda body: parse_works(json.loads(body)),
    "arxiv": parse_feed,
}


@cache
def _entries(path: Path = POOLS) -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(gzip.decompress(path.read_bytes()))["entries"])


@cache
def _json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


@pytest.mark.parametrize("source", ["openalex", "arxiv"])
def test_the_adapters_see_exactly_what_the_spike_measured(source: str) -> None:
    pages = [
        PARSERS[source](entry["body"])
        for entry in _entries()
        if entry["source"] == source
        and entry["pass"] == "P1"
        and entry["kind"] == "search"
        and entry["status"] == 200
    ]
    candidates = [c for page in pages for c in page.candidates]
    measured = _json(ANALYSIS)["asm1"]["completeness"][source]

    assert sum(page.skipped for page in pages) == 0, "a record the spike counted was skipped"
    assert len(candidates) == measured["records"]
    got = {field: sum(1 for c in candidates if test(c)) for field, test in FIELDS.items()}
    want = {field: measured["fields"][field]["k"] for field in FIELDS}
    assert got == want


def _query_text(entry: dict[str, Any]) -> str:
    topic = next(t for t in _json(PROTOCOL)["topics"] if t["id"] == entry["topic"])
    return str(topic[entry["phrasing"]])


def test_openalex_builds_the_request_the_pools_recorded() -> None:
    source = OpenAlexSource("unused-in-the-url")
    compared = 0
    for entry in _entries():
        # The five 400s were sent before DEV-2 amended the query (D117); their
        # repeats, and everything after, ran with the construction shipped here.
        if entry["source"] != "openalex" or entry["kind"] != "search" or entry["status"] == 400:
            continue
        recorded = parse_qs(urlsplit(entry["url"]).query)
        built = parse_qs(urlsplit(str(source.request(_query_text(entry), limit=50).url)).query)
        assert built == recorded, entry["rid"]
        compared += 1
    assert compared == 30  # 20 P1 searches (5 of them DEV-2 repeats) + 10 P5


def test_arxiv_builds_the_request_the_pools_recorded_for_its_shipped_join() -> None:
    # The spike's pools and A1's both hold requests made with the join shipped here
    # (D120 kept AND), so the adapter must rebuild every one of them exactly.
    join = arxiv_adapter.DEFAULT_JOIN
    assert join == "AND", "D120 kept AND; shipping OR needs a new measurement and a new decision"
    spike = [e for e in _entries() if e["source"] == "arxiv" and e["kind"] == "search"]
    a1 = [e for e in _entries(POOLS_A1) if e.get("construction") == join]
    entries = spike + a1
    expected = 30 + 17  # spike: 20 P1 + 10 P5; A1: the AND responses that arrived (3 timed out)
    source = ArxivSource(join=join)
    for entry in entries:
        recorded = parse_qs(urlsplit(entry["url"]).query)
        built = parse_qs(urlsplit(str(source.request(_query_text(entry), limit=50).url)).query)
        assert built == recorded, entry["rid"]
    assert len(entries) == expected


def test_the_fixtures_are_cut_from_the_frozen_pools_not_invented() -> None:
    provenance = _json(FIXTURES / "provenance.json")
    assert hashlib.sha256(POOLS.read_bytes()).hexdigest() == provenance["pools_sha256"]
    by_rid = {e["rid"]: e for e in _entries()}
    for name, record in provenance["files"].items():
        text = (FIXTURES / name).read_text(encoding="utf-8")
        if record.get("hand_written"):
            continue
        body = by_rid[record["rid"]]["body"]
        if name.endswith(".json") and "kept_work_ids" in record:
            original = {w["id"]: w for w in json.loads(body)["results"]}
            assert [original[i] for i in record["kept_work_ids"]] == json.loads(text)["results"]
        elif name.endswith(".json"):
            assert json.loads(text) == json.loads(body)
        else:
            entries = text.split("<entry>")[1:]
            assert entries and all(("<entry>" + e.split("</entry>")[0]) in body for e in entries), (
                name
            )
