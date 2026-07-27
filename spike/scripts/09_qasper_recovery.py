"""M0 deliverable 7 (part B) — recover a chunk-granularity eval seed.

`datasets` v4 dropped script-based loaders, which blocks allenai/qasper — the one
candidate with evidence annotations at *paragraph* granularity inside full papers.
BEIR loads fine but its unit is the abstract, so it cannot seed a chunk-level
retrieval set on its own.

Three recovery routes, cheapest first:
  A. HF auto-converts script datasets to a parquet branch (refs/convert/parquet)
  B. the community parquet mirrors
  C. BEIR scifact/scidocs with queries+qrels (document-level fallback)

If A or B works, ASM-5 passes and RSK-2 is contained. If only C works, D-RET is
document-level and the chunk-level judgments need hand-labelling (2-3 days).

Run:  .venv\\Scripts\\python.exe scripts/09_qasper_recovery.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "09_qasper_recovery.json"


def try_parquet_branch() -> dict[str, object]:
    """Route A — the auto-converted parquet revision."""
    from datasets import load_dataset

    rec: dict[str, object] = {"route": "A: refs/convert/parquet"}
    try:
        ds = load_dataset("allenai/qasper", revision="refs/convert/parquet",
                          split="validation")
        rec |= {"ok": True, "n_rows": len(ds), "columns": list(ds.column_names)}
    except Exception as exc:  # noqa: BLE001
        rec |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return rec


def try_mirrors() -> list[dict[str, object]]:
    """Route B — community parquet mirrors of QASPER."""
    from datasets import load_dataset

    out: list[dict[str, object]] = []
    for repo in ("nreimers/qasper", "qasper", "allenai/qasper_truncated_4000",
                 "TIGER-Lab/qasper"):
        rec: dict[str, object] = {"route": f"B: {repo}"}
        try:
            ds = load_dataset(repo, split="validation")
            rec |= {"ok": True, "n_rows": len(ds), "columns": list(ds.column_names)}
        except Exception as exc:  # noqa: BLE001
            rec |= {"ok": False, "error": f"{type(exc).__name__}"[:80]}
        out.append(rec)
        if rec.get("ok"):
            break
    return out


def beir_completeness() -> dict[str, object]:
    """Route C — confirm BEIR gives a complete queries+corpus+qrels triple."""
    from datasets import load_dataset

    rec: dict[str, object] = {"route": "C: BEIR scifact (document-level)"}
    try:
        queries = load_dataset("BeIR/scifact", "queries", split="queries")
        corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
        qrels = load_dataset("BeIR/scifact-qrels", split="test")
        rec |= {
            "ok": True,
            "n_queries": len(queries),
            "n_corpus": len(corpus),
            "n_qrels": len(qrels),
            "query_columns": list(queries.column_names),
            "sample_query": queries[0].get("text", "")[:140],
            "sample_doc_chars": len(corpus[0].get("text", "")),
            "graded": sorted({int(r["score"]) for r in qrels.select(
                range(min(500, len(qrels))))}),
        }
    except Exception as exc:  # noqa: BLE001
        rec |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return rec


def main() -> int:
    records: list[dict[str, object]] = []

    print("Route A: parquet branch ...", flush=True)
    a = try_parquet_branch()
    records.append(a)
    print(f"  {'ok' if a['ok'] else 'FAIL'}  "
          f"{a.get('n_rows', a.get('error', ''))}")

    if not a["ok"]:
        print("\nRoute B: community mirrors ...", flush=True)
        for b in try_mirrors():
            records.append(b)
            print(f"  {'ok' if b['ok'] else 'FAIL'}  {b['route']}  "
                  f"{b.get('n_rows', b.get('error', ''))}")

    print("\nRoute C: BEIR completeness ...", flush=True)
    c = beir_completeness()
    records.append(c)
    if c["ok"]:
        print(f"  ok  {c['n_queries']} queries / {c['n_corpus']} docs / "
              f"{c['n_qrels']} qrels, relevance grades {c['graded']}")
        print(f"  sample query: {c['sample_query']!r}")
        print(f"  mean doc size: ~{c['sample_doc_chars']} chars (abstract-level)")
    else:
        print(f"  FAIL  {c.get('error')}")

    chunk_level = any(r.get("ok") and r["route"].startswith(("A", "B")) for r in records)
    doc_level = c.get("ok", False)

    print("\n--- ASM-5 verdict ---")
    if chunk_level:
        print("  PASS: chunk-granularity evidence recovered -> D-RET bootstrappable")
    elif doc_level:
        print("  PARTIAL: BEIR gives a complete document-level retrieval set.")
        print("    D-RET plan: use BEIR scifact as the IR baseline (validates the")
        print("    retrieval stack end-to-end), and hand-label ~30-40 chunk-level")
        print("    queries over the fixture papers for the domain-specific set.")
        print("    Budget: 2-3 days (RSK-2 mitigation, still contained).")
    else:
        print("  FAIL: no usable public seed. Hand-labelling is the only path.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"chunk_level_available": chunk_level, "doc_level_available": doc_level,
         "records": records}, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
