"""The CI smoke gate (FR-E6, doc 8 deliverable 7, D82).

A full eval needs a GPU, 7,854 chunks and minutes. CI has none of those, so this
runs the same pipeline over a seeded 10-paper subset on CPU and compares every
metric with a committed baseline. **A drop beyond the tolerance fails the build; a
rise is reported and passes** — FR-E6 asks for a regression gate, not a snapshot
test (D82).

**These numbers are not the published ones.** The adapter forces fp32 on CPU
(D18), the corpus is a subset, and retrieval is scoped to it — so the values here
are a fixed point to detect *change* against, never a result to quote. The
baseline file repeats that where someone reading it would need it.

**What it covers, and what it cannot.** Chunking, embedding, indexing, lexical
tokenisation, fusion and the metric pipeline — everything that moves numbers
without raising. Reranking is absent on purpose: on CPU it is minutes for this
query set, which is the budget doc 10 §10.6 gives the whole job.

The comparison is pure and lives here; the measuring lives in the CLI, because it
needs adapters and a store. That split is what lets the gate's logic — the part
that decides whether a build fails — be tested without either.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Committed, unlike every other eval artifact: a gate needs its reference in the
#: repository, or there is nothing to regress against on a fresh checkout.
BASELINE_PATH = Path("data/eval/smoke-baseline.json")

#: Papers in the smoke corpus. Ten is ~290 chunks and ~30 queries, which embeds in
#: ~26 s at the 11.1 chunks/s measured on CPU here — inside doc 10 §10.6's ~2
#: minutes with room for a slower runner (D82).
SMOKE_PAPERS = 10
SMOKE_SEED = 20260925

#: A metric may fall this far before the build fails. About 1.5 queries' worth on
#: this subset: wide enough for float drift between the machine that wrote the
#: baseline and the one checking it, narrow enough that a broken pipeline — which
#: collapses metrics rather than nudging them — always trips it (D82).
TOLERANCE = 0.05


@dataclass(frozen=True, slots=True)
class Movement:
    """One metric, then and now."""

    arm: str
    metric: str
    baseline: float
    measured: float

    @property
    def delta(self) -> float:
        return self.measured - self.baseline

    def verdict(self, tolerance: float) -> str:
        if self.delta < -tolerance:
            return "REGRESSION"
        return "improved" if self.delta > tolerance else "ok"

    def __str__(self) -> str:
        return (
            f"{self.arm:<8} {self.metric:<12} {self.baseline:.3f} -> {self.measured:.3f}  "
            f"{self.delta:+.3f}"
        )


@dataclass(frozen=True, slots=True)
class Comparison:
    """Every metric's movement, and whether the build survives it."""

    movements: tuple[Movement, ...]
    missing: tuple[str, ...]
    added: tuple[str, ...]
    tolerance: float

    @property
    def regressions(self) -> tuple[Movement, ...]:
        return tuple(m for m in self.movements if m.verdict(self.tolerance) == "REGRESSION")

    @property
    def improvements(self) -> tuple[Movement, ...]:
        return tuple(m for m in self.movements if m.verdict(self.tolerance) == "improved")

    @property
    def passed(self) -> bool:
        """A missing metric fails too: it means the run stopped producing one.

        Silently passing on absence would make the gate weakest exactly when the
        pipeline is most broken — a configuration that emits nothing would sail
        through a comparison that only looks at what it emitted.
        """
        return not self.regressions and not self.missing

    def render(self) -> str:
        lines = [f"smoke gate: tolerance {self.tolerance:+.3f} on a drop, rises reported", ""]
        for movement in self.movements:
            verdict = movement.verdict(self.tolerance)
            marker = {"REGRESSION": "  <-- REGRESSION", "improved": "  <-- improved"}.get(
                verdict, ""
            )
            lines.append(f"  {movement}{marker}")
        if self.missing:
            lines.append("")
            lines.append(f"  MISSING from this run: {', '.join(self.missing)}")
        if self.added:
            lines.append(f"  new since the baseline (not gated): {', '.join(self.added)}")
        lines.append("")
        if self.passed:
            summary = "PASS"
            if self.improvements:
                summary += (
                    f" — {len(self.improvements)} metric(s) improved beyond tolerance; "
                    "regenerate the baseline with `uv run eval --smoke --update-baseline` "
                    "if that is intended"
                )
            lines.append(summary)
        else:
            lines.append(
                f"FAIL — {len(self.regressions)} regression(s), {len(self.missing)} missing"
            )
        return "\n".join(lines)


def compare(
    baseline: Mapping[str, Mapping[str, float]],
    measured: Mapping[str, Mapping[str, float]],
    *,
    tolerance: float = TOLERANCE,
) -> Comparison:
    """Line the two runs up metric by metric. Pure: no I/O, no adapters.

    Everything the gate decides happens here, so the decision can be tested
    without a model, a store or a network.
    """
    movements: list[Movement] = []
    missing: list[str] = []
    for arm in sorted(baseline):
        for metric in sorted(baseline[arm]):
            if arm not in measured or metric not in measured[arm]:
                missing.append(f"{arm}/{metric}")
                continue
            movements.append(
                Movement(
                    arm=arm,
                    metric=metric,
                    baseline=float(baseline[arm][metric]),
                    measured=float(measured[arm][metric]),
                )
            )
    added = [
        f"{arm}/{metric}"
        for arm in sorted(measured)
        for metric in sorted(measured[arm])
        if arm not in baseline or metric not in baseline[arm]
    ]
    return Comparison(
        movements=tuple(movements),
        missing=tuple(missing),
        added=tuple(added),
        tolerance=tolerance,
    )


def load_baseline(path: Path = BASELINE_PATH) -> dict[str, Any]:
    """Read the committed baseline, with a message that says how to make one."""
    if not path.exists():
        raise FileNotFoundError(
            f"no smoke baseline at {path}. Create one deliberately with "
            "`uv run eval --smoke --update-baseline`; a gate that writes its own "
            "reference on first run gates nothing"
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


def save_baseline(report: Mapping[str, Any], path: Path = BASELINE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
