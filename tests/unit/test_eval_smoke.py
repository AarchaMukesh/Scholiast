"""The CI smoke gate's decision logic (FR-E6, D82).

Everything that decides whether a build fails lives in `compare`, which is pure —
so the decision is tested here without a model, a store or a runner. The cases
that matter are the ones where a gate typically fails silently: a metric that
disappeared, a baseline that does not exist yet, and a change small enough to be
float drift rather than a regression.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholiast.eval.smoke import (
    TOLERANCE,
    Movement,
    compare,
    load_baseline,
    save_baseline,
)

BASE = {"dense": {"recall@10": 0.50, "mrr": 0.40}, "hybrid": {"recall@10": 0.60, "mrr": 0.50}}


def _moved(arm: str, metric: str, to: float) -> dict[str, dict[str, float]]:
    measured = {a: dict(m) for a, m in BASE.items()}
    measured[arm][metric] = to
    return measured


class TestWhatFailsTheBuild:
    def test_a_drop_beyond_tolerance_is_a_regression(self) -> None:
        comparison = compare(BASE, _moved("dense", "recall@10", 0.50 - TOLERANCE - 0.001))
        assert not comparison.passed
        assert [m.metric for m in comparison.regressions] == ["recall@10"]

    def test_a_drop_within_tolerance_passes(self) -> None:
        # Float drift between the machine that wrote the baseline and the runner
        # checking it must not break builds (D82).
        comparison = compare(BASE, _moved("dense", "recall@10", 0.50 - TOLERANCE + 0.001))
        assert comparison.passed
        assert not comparison.regressions

    def test_a_rise_is_reported_and_passes(self) -> None:
        # FR-E6 asks for a regression gate, not a snapshot test: an improvement
        # must not fail the build, but it must be visible so the baseline gets
        # regenerated deliberately.
        comparison = compare(BASE, _moved("hybrid", "mrr", 0.90))
        assert comparison.passed
        assert [m.metric for m in comparison.improvements] == ["mrr"]
        assert "improved" in comparison.render()
        assert "--update-baseline" in comparison.render()

    def test_a_metric_that_vanished_fails_rather_than_being_skipped(self) -> None:
        # The failure a naive comparison misses entirely: a run that emits nothing
        # would sail through a check that only looks at what it emitted.
        measured = {"dense": {"recall@10": 0.50}, "hybrid": {"recall@10": 0.60, "mrr": 0.50}}
        comparison = compare(BASE, measured)
        assert not comparison.passed
        assert comparison.missing == ("dense/mrr",)
        assert "MISSING" in comparison.render()

    def test_a_whole_arm_that_vanished_fails(self) -> None:
        comparison = compare(BASE, {"dense": BASE["dense"]})
        assert not comparison.passed
        assert set(comparison.missing) == {"hybrid/recall@10", "hybrid/mrr"}

    def test_a_new_metric_is_reported_but_does_not_gate(self) -> None:
        # Adding a metric should not fail a build; it should be visible so the
        # baseline can be regenerated to start gating it.
        measured = {a: dict(m) for a, m in BASE.items()}
        measured["dense"]["ndcg@10"] = 0.3
        comparison = compare(BASE, measured)
        assert comparison.passed
        assert comparison.added == ("dense/ndcg@10",)
        assert "not gated" in comparison.render()

    def test_an_identical_run_passes_with_nothing_flagged(self) -> None:
        comparison = compare(BASE, BASE)
        assert comparison.passed
        assert not comparison.regressions
        assert not comparison.improvements
        assert "PASS" in comparison.render()

    def test_every_baseline_metric_is_checked(self) -> None:
        # A gate that quietly checked a subset would be worse than none, because
        # the report would still say PASS.
        comparison = compare(BASE, BASE)
        assert {(m.arm, m.metric) for m in comparison.movements} == {
            ("dense", "recall@10"),
            ("dense", "mrr"),
            ("hybrid", "recall@10"),
            ("hybrid", "mrr"),
        }


class TestTheVerdictArithmetic:
    def test_the_boundary_is_not_a_regression(self) -> None:
        # Exactly at tolerance passes; one step beyond fails. Pinned because an
        # off-by-one here is the difference between a flaky gate and a silent one.
        exact = Movement(arm="dense", metric="recall@10", baseline=0.5, measured=0.5 - TOLERANCE)
        assert exact.verdict(TOLERANCE) == "ok"

    def test_the_rendered_line_shows_both_values_and_the_delta(self) -> None:
        line = str(Movement(arm="dense", metric="mrr", baseline=0.4, measured=0.3))
        assert "0.400" in line
        assert "0.300" in line
        assert "-0.100" in line


class TestTheBaselineFile:
    def test_a_missing_baseline_says_how_to_make_one(self, tmp_path: Path) -> None:
        # On a fresh checkout this is the first thing a contributor meets, and the
        # wrong recovery — generating it automatically — would disable the gate.
        with pytest.raises(FileNotFoundError, match="--update-baseline"):
            load_baseline(tmp_path / "absent.json")

    def test_it_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "smoke-baseline.json"
        report = {"config": {"seed": 1}, "metrics": BASE}
        save_baseline(report, path)
        assert load_baseline(path) == report

    def test_it_is_written_deterministically(self, tmp_path: Path) -> None:
        # It is committed, so an unstable key order would produce a diff on every
        # regeneration and make real changes hard to see in review.
        path = tmp_path / "smoke-baseline.json"
        save_baseline({"metrics": BASE, "config": {"b": 1, "a": 2}}, path)
        first = path.read_text(encoding="utf-8")
        save_baseline({"config": {"a": 2, "b": 1}, "metrics": BASE}, path)
        assert path.read_text(encoding="utf-8") == first
        assert json.loads(first)["metrics"] == BASE


class TestTheFlagsRefuseMisuse:
    def test_update_baseline_without_smoke_is_an_error(self) -> None:
        # Otherwise it runs an ordinary single-arm eval and reports success, and
        # the user believes they regenerated a baseline they did not touch.
        from scholiast.eval.cli import main

        with pytest.raises(SystemExit) as exit_info:
            main(["--update-baseline"])
        assert exit_info.value.code == 2, "argparse misuse exits 2"


class TestTheCommittedBaseline:
    """The real file, which the CI job checks against."""

    def test_it_exists_and_covers_both_arms(self) -> None:
        baseline = load_baseline()
        assert set(baseline["metrics"]) == {"dense", "hybrid"}
        for arm in baseline["metrics"].values():
            assert {"recall@10", "recall@20", "recall@50", "mrr", "ndcg@10"} <= set(arm)

    def test_it_says_these_are_not_the_published_numbers(self) -> None:
        # The one way this file could mislead: someone quoting a CPU/fp32 subset
        # score as a project result.
        config = load_baseline()["config"]
        assert "not the published results" in config["purpose"]
        assert config["device"] == "cpu"
        assert config["tolerance"] == TOLERANCE

    def test_it_records_what_produced_it(self) -> None:
        # NFR-M5. A baseline whose provenance is unknown cannot be regenerated or
        # trusted after any change to the corpus or the encoder.
        config = load_baseline()["config"]
        assert len(config["papers"]) == 10
        assert config["embedder_version"].startswith("BAAI/bge-base-en-v1.5")
        assert config["dataset_versions"]["chunker"]
        assert config["regenerate"].endswith("--update-baseline")
