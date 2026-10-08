"""FR-D3's measurement holds still: the fixture, the fit and the held-out score (D146, D148).

What fails silently here is a measurement quietly becoming a different one: a
relabelled fixture still reporting the old numbers' provenance, a threshold edited
by hand while the log says it was fitted, a fit that peeked at the held-out topics,
or a matcher change that moves the reported precision and recall without anyone
re-recording them. Offline: everything read here is committed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from scholiast.domain import dedup
from scholiast.domain.discovery import PaperCandidate
from scholiast.eval import dedup as measure
from scholiast.eval.metrics import wilson_interval

#: D146: the labels as frozen before any matcher code existed.
FROZEN_SHA256 = "3aa8b15e9fb78a3b389d4e4f4db8e565ce7dc6487eff7aabcedf0cf88e6c1bb8"
#: D146: the same labels with the owner's review recorded beside them.
REVIEWED_SHA256 = "5a884a342aadf42ff61376f25eb525fb66da90117a91cc66ab39c0c47380ead5"


@cache
def _records() -> dict[str, tuple[PaperCandidate, ...]]:
    return measure.topic_records()


@cache
def _pairs() -> tuple[measure.LabelledPair, ...]:
    return measure.load_pairs()


def _split(name: str) -> list[measure.LabelledPair]:
    return [p for p in _pairs() if p.split == name]


class TestTheFixture:
    def test_is_the_file_the_manifest_and_d146_name(self) -> None:
        digest = hashlib.sha256(measure.FIXTURE.read_bytes()).hexdigest()
        manifest = json.loads(measure.MANIFEST.read_text(encoding="utf-8"))

        assert digest == manifest["sha256"] == REVIEWED_SHA256

    def test_holds_exactly_the_labels_frozen_before_the_matcher(self) -> None:
        # The owner's review sits beside the labeller's labels, never over them, so
        # removing it must give back the frozen bytes.
        data: dict[str, Any] = json.loads(measure.FIXTURE.read_text(encoding="utf-8"))
        assert data["owner_review"]["labels_frozen_sha256"] == FROZEN_SHA256
        del data["owner_review"]
        for pair in data["pairs"]:
            pair.pop("owner_label", None)
        frozen = json.dumps(data, ensure_ascii=False, indent=1) + "\n"

        assert hashlib.sha256(frozen.encode("utf-8")).hexdigest() == FROZEN_SHA256

    def test_every_pair_names_one_record_the_production_adapters_parse(self) -> None:
        records = _records()
        keys = {topic: [(c.source, c.source_id) for c in recs] for topic, recs in records.items()}
        for topic, found in keys.items():
            assert len(found) == len(set(found)), f"{topic} has a record twice"
        for p in _pairs():
            assert p.a in keys[p.topic] and p.b in keys[p.topic], p.pair

    def test_splits_by_topic_as_d146_fixed_it(self) -> None:
        for p in _pairs():
            odd = int(p.topic.removeprefix("t")) % 2 == 1
            assert p.split == ("train" if odd else "held_out"), p.pair
        assert (len(_split("train")), len(_split("held_out"))) == (181, 198)

    def test_an_unresolved_unsure_pair_cannot_be_scored(self, tmp_path: Path) -> None:
        data = json.loads(measure.FIXTURE.read_text(encoding="utf-8"))
        data["pairs"][0].update(label="unsure")
        data["pairs"][0].pop("owner_label", None)
        path = tmp_path / "fixture.json"
        path.write_text(json.dumps(data), encoding="utf-8")

        with pytest.raises(ValueError, match="unsure"):
            measure.load_pairs(path)


class TestTheFit:
    def test_the_shipped_rule_is_the_one_the_training_topics_fit(self) -> None:
        fitted, _ = measure.fit(_split("train"), _records())

        assert fitted == dedup.SHIPPED_RULE, (
            "SHIPPED_RULE is not what D148's fit produces; re-run `eval --dedup` and "
            "record the change rather than editing the threshold by hand"
        )

    def test_never_reads_a_held_out_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: list[str] = []
        real_fit = measure.fit

        def spy(
            pairs: Sequence[measure.LabelledPair], records: Any
        ) -> tuple[dedup.MatchRule, tuple[measure.GridRow, ...]]:
            seen.extend(p.split for p in pairs)
            return real_fit(pairs, records)

        monkeypatch.setattr(measure, "fit", spy)
        monkeypatch.setattr(measure, "LATENCY_REPEATS", 1)
        measure.run_dedup()

        assert seen and set(seen) == {"train"}

    def test_never_buys_recall_with_a_false_merge(self) -> None:
        # D148: a false merge hides a paper. Here the only rule that finds the one
        # true pair (T = 0.6) also merges two different papers, so the fit must
        # give up that recall. The real fixture cannot show this: its clean rows
        # happen to reach full training recall.
        def made(n: int, title: str) -> PaperCandidate:
            return PaperCandidate(
                source="s",
                source_id=str(n),
                rank=n,
                title=title,
                authors=("Ada Lovelace",),
                year=2020,
                adapter_version="t",
            )

        records = {
            "t1": (
                made(1, "alpha beta gamma delta"),
                made(2, "alpha beta gamma epsilon"),
                made(3, "alpha beta gamma zeta"),
            )
        }
        pairs = [
            measure.LabelledPair("p1", "t1", "train", ("s", "1"), ("s", "2"), same=True),
            measure.LabelledPair("p2", "t1", "train", ("s", "1"), ("s", "3"), same=False),
        ]

        fitted, grid = measure.fit(pairs, records)

        assert any(r.tp == 1 and r.fp > 0 for r in grid), "the trap is not on the grid"
        assert fitted.title_threshold > 0.6


class TestTheHeldOutScore:
    def test_is_the_one_d148_records(self) -> None:
        score = measure.score(dedup.SHIPPED_RULE, _split("held_out"), _records())

        assert (score.tp, score.fp, score.fn, score.tn) == (51, 0, 4, 143)
        assert {e.pair for e in score.errors} == {"t06-186", "t06-188", "t08-234", "t08-235"}
        assert {e.cause for e in score.errors} == {"title"}

    def test_the_title_rule_is_what_lifts_recall_above_identifiers(self) -> None:
        held_out = _split("held_out")

        with_titles = measure.score(dedup.SHIPPED_RULE, held_out, _records())
        ids_only = measure._ids_only_score(held_out, _records())

        assert (ids_only.tp, ids_only.fp) == (16, 0)
        assert with_titles.tp > ids_only.tp


class TestWilson:
    @pytest.mark.parametrize(
        ("k", "n", "low", "high"),
        [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634)],
    )
    def test_matches_the_published_score_interval(
        self, k: int, n: int, low: float, high: float
    ) -> None:
        interval = wilson_interval(k, n)

        assert interval.point == pytest.approx(k / n)
        assert (interval.low, interval.high) == (
            pytest.approx(low, abs=1e-4),
            pytest.approx(high, abs=1e-4),
        )

    @pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 5), (6, 5)])
    def test_refuses_a_count_that_is_not_a_proportion(self, k: int, n: int) -> None:
        with pytest.raises(ValueError):
            wilson_interval(k, n)
