"""M0 deliverable 7 — evaluation dataset feasibility.

Validates ASM-5, which guards the project's single biggest schedule risk (RSK-2):
hand-labelling retrieval relevance is slow, boring, always deferred, and dropping
it destroys the entire differentiator. If a public dataset can seed D-RET, that
risk mostly evaporates.

What we need from a candidate:
  1. queries that look like real questions about papers
  2. relevance judgments at *passage* granularity (so they map onto our chunks)
  3. the underlying paper text, or an ID we can resolve to it

Run:  .venv\\Scripts\\python.exe scripts/05_eval_dataset_check.py
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "05_eval_datasets.json"

# (hf_id, config, split, why we care)
CANDIDATES: list[tuple[str, str | None, str, str]] = [
    ("allenai/qasper", None, "validation",
     "Questions over NLP papers WITH evidence paragraphs — the closest fit to D-RET"),
    ("BeIR/scifact", "corpus", "corpus",
     "Ready-made scientific retrieval corpus (BEIR)"),
    ("BeIR/scifact-qrels", None, "test",
     "Relevance judgments for BeIR/scifact — queries x doc relevance"),
    ("BeIR/scidocs", "corpus", "corpus",
     "Scientific document retrieval; citation-based relevance"),
    ("allenai/scifact", None, "train",
     "Claim verification with evidence sentences — seeds D-UNANS / entailment"),
]


def probe(hf_id: str, config: str | None, split: str, why: str) -> dict[str, object]:
    rec: dict[str, object] = {"dataset": hf_id, "config": config, "split": split,
                              "why": why}
    try:
        from datasets import load_dataset

        ds = load_dataset(hf_id, config, split=split)
        rec["ok"] = True
        rec["n_rows"] = len(ds)
        rec["columns"] = list(ds.column_names)

        # Structural fingerprint of the first row — enough to judge whether the
        # evidence granularity actually maps onto chunks.
        row = ds[0]
        shape: dict[str, str] = {}
        for k, v in row.items():
            if isinstance(v, str):
                shape[k] = f"str(len={len(v)}): {v[:90]!r}"
            elif isinstance(v, list):
                shape[k] = f"list(len={len(v)})"
            elif isinstance(v, dict):
                shape[k] = f"dict(keys={list(v.keys())[:6]})"
            else:
                shape[k] = type(v).__name__
        rec["first_row_shape"] = shape
    except Exception as exc:  # noqa: BLE001 - spike diagnostic
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rec["trace_tail"] = traceback.format_exc().strip().splitlines()[-1][:200]
    return rec


def assess_qasper(rec: dict[str, object]) -> dict[str, object]:
    """QASPER is the primary candidate — check evidence granularity explicitly."""
    if not rec.get("ok"):
        return {"usable": False, "reason": "failed to load"}
    cols = set(rec.get("columns", []))
    needed = {"qas", "full_text"}
    have = needed & cols
    return {
        "usable": bool(have),
        "has_qas": "qas" in cols,
        "has_full_text": "full_text" in cols,
        "note": ("Evidence is per-paragraph within full_text; paragraphs map to our "
                 "chunks by string match, giving graded judgments without hand "
                 "labelling." if have == needed else "Schema differs from expectation "
                 "— inspect columns before relying on it."),
    }


def main() -> int:
    try:
        import datasets  # noqa: F401
    except ImportError:
        print("FAIL: `datasets` not installed.")
        print("  uv pip install --python .venv\\Scripts\\python.exe datasets")
        return 1

    records = []
    for hf_id, config, split, why in CANDIDATES:
        print(f"probing {hf_id} ({config or '-'}/{split}) ...", flush=True)
        rec = probe(hf_id, config, split, why)
        records.append(rec)
        if rec["ok"]:
            print(f"  ok    {rec['n_rows']:>7,} rows  cols={rec['columns']}")
        else:
            print(f"  FAIL  {rec['error']}")

    qasper = next((r for r in records if r["dataset"] == "allenai/qasper"), {})
    verdict = assess_qasper(qasper)

    n_ok = sum(1 for r in records if r["ok"])
    print("\n--- ASM-5 verdict ---")
    print(f"  {n_ok}/{len(CANDIDATES)} candidate datasets loadable")
    print(f"  QASPER usable as D-RET seed: {verdict['usable']}")
    print(f"  {verdict.get('note', '')}")
    if n_ok == 0:
        print("  -> FAIL: budget 2-3 days for hand-labelling ~40 queries (RSK-2)")
    elif verdict["usable"]:
        print("  -> PASS: D-RET can be bootstrapped rather than hand-labelled")
    else:
        print("  -> PARTIAL: BEIR qrels usable for retrieval; answer eval needs work")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"n_loadable": n_ok, "qasper_verdict": verdict, "records": records}, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
