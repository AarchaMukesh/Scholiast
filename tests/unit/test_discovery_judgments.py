"""Phase 8's judged set holds still: the A3 pools, the grades, and their source (D149, D150).

What fails silently here is a ranker scored against judgments that no longer
describe the candidates: a pool file swapped or re-recorded under the same name, a
grade edited after the freeze, a paper left unjudged that a ranker could put in the
top ten (it would score as irrelevant), or a work split or merged differently from
the one judged. Offline: everything read here is committed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

from scholiast.domain import dedup
from scholiast.domain.dedup import MatchRule
from scholiast.domain.discovery import PaperCandidate, interleave
from scholiast.infrastructure.discovery.arxiv import parse_feed
from scholiast.infrastructure.discovery.openalex import parse_works

ROOT = Path(__file__).resolve().parents[2]
POOLS = ROOT / "data" / "eval" / "discovery-pools-a3.json.gz"
POOLS_MANIFEST = ROOT / "data" / "eval" / "discovery-pools-a3.manifest.json"
JUDGMENTS = ROOT / "data" / "eval" / "discovery-judgments.json"
JUDGMENTS_MANIFEST = ROOT / "data" / "eval" / "discovery-judgments.manifest.json"

#: D150: the A3 responses as recorded on 2026-10-08.
POOLS_SHA256 = "621a082a31aa0caf6885c20c5c4e9c3ad5951ea3796068c7caf8f1b2f557df18"
#: D149: the grades as frozen before any ranker code existed.
FROZEN_SHA256 = "14b7246e6a470f597c624a0bed0356ca5635a38fea5ecc0f127de0e7f7e19765"
#: D149: the same grades with the owner's review recorded beside them.
REVIEWED_SHA256 = "fdefe1261254829addb90b634d4d7e9b4d5195b8a09dba0ad7bf21629506dc20"
#: The matcher the judging units were built with (D148), named here rather than read
#: from SHIPPED_RULE: a later change to the shipped rule must not silently redefine
#: what was judged.
JUDGED_WITH = MatchRule(title_threshold=0.85, year_tolerance=2)

Key = tuple[str, str]


@cache
def _judgments() -> dict[str, Any]:
    return dict(json.loads(JUDGMENTS.read_text(encoding="utf-8")))


@cache
def _topic_records() -> dict[str, tuple[PaperCandidate, ...]]:
    """Each topic's A3 records, both phrasings, in the order the judging pool used."""
    pages: dict[tuple[str, str], dict[str, tuple[PaperCandidate, ...]]] = {}
    for entry in json.loads(gzip.decompress(POOLS.read_bytes()))["entries"]:
        if entry["status"] != 200:
            continue
        body = entry["body"]
        page = parse_works(json.loads(body)) if entry["source"] == "openalex" else parse_feed(body)
        pages.setdefault((entry["topic"], entry["phrasing"]), {})[entry["source"]] = page.candidates
    out: dict[str, tuple[PaperCandidate, ...]] = {}
    for topic, _ in sorted(pages):
        if topic in out:
            continue
        seen: set[Key] = set()
        records: list[PaperCandidate] = []
        for phrasing in ("keyword", "natural_language"):
            by_source = pages.get((topic, phrasing), {})
            for record in interleave([by_source.get("openalex", ()), by_source.get("arxiv", ())]):
                # A record both phrasings returned is one record.
                if (record.source, record.source_id) not in seen:
                    seen.add((record.source, record.source_id))
                    records.append(record)
        out[topic] = tuple(records)
    return out


def _member_sets(works: Sequence[Any]) -> set[frozenset[Key]]:
    return {frozenset((m[0], m[1]) for m in w["members"]) for w in works}


class TestThePools:
    def test_are_the_responses_a3_recorded(self) -> None:
        manifest = json.loads(POOLS_MANIFEST.read_text(encoding="utf-8"))

        assert hashlib.sha256(POOLS.read_bytes()).hexdigest() == manifest["sha256"] == POOLS_SHA256
        assert manifest["decision"] == "D150"
        assert manifest["by_source_and_status"] == {"arxiv/200": 20, "openalex/200": 20}


class TestTheJudgments:
    def test_are_the_file_the_manifest_and_d149_name(self) -> None:
        digest = hashlib.sha256(JUDGMENTS.read_bytes()).hexdigest()
        manifest = json.loads(JUDGMENTS_MANIFEST.read_text(encoding="utf-8"))

        assert digest == manifest["sha256"] == REVIEWED_SHA256

    def test_hold_exactly_the_grades_frozen_before_the_ranker(self) -> None:
        data = json.loads(JUDGMENTS.read_text(encoding="utf-8"))
        assert data["owner_review"]["labels_frozen_sha256"] == FROZEN_SHA256
        del data["owner_review"]
        frozen = json.dumps(data, ensure_ascii=False, indent=1) + "\n"

        assert hashlib.sha256(frozen.encode("utf-8")).hexdigest() == FROZEN_SHA256

    def test_are_made_from_these_pools(self) -> None:
        manifest = json.loads(POOLS_MANIFEST.read_text(encoding="utf-8"))

        assert _judgments()["pools"]["sha256"] == manifest["sha256"]

    def test_judge_every_paper_a_search_could_rank(self) -> None:
        # An unjudged paper in a top ten would score as irrelevant, rewarding any
        # ranker that buried it.
        records = _topic_records()
        for topic in _judgments()["topics"]:
            judged = {k for w in topic["works"] for k in map(tuple, w["members"])}
            returned = {(r.source, r.source_id) for r in records[topic["topic"]]}
            assert judged == returned, topic["topic"]

    def test_judge_the_works_the_matcher_of_record_makes(self) -> None:
        records = _topic_records()
        for topic in _judgments()["topics"]:
            recs = records[topic["topic"]]
            built = {
                frozenset((recs[i].source, recs[i].source_id) for i in members)
                for members in dedup.group(recs, JUDGED_WITH)
            }
            assert built == _member_sets(topic["works"]), topic["topic"]

    def test_split_by_topic_as_d146_and_d149_fixed_it(self) -> None:
        topics = _judgments()["topics"]
        for topic in topics:
            odd = int(topic["topic"].removeprefix("t")) % 2 == 1
            assert topic["split"] == ("train" if odd else "held_out"), topic["topic"]
        assert [t["topic"] for t in topics] == [f"t{n:02d}" for n in range(1, 11)]

    def test_grade_on_the_three_point_scale_only(self) -> None:
        grades = {w["grade"] for t in _judgments()["topics"] for w in t["works"]}

        assert grades <= {0, 1, 2}
