"""M0 deliverable 4 (part C) — can production rerank without a GPU? (DEP-2)

First benchmark: bge-reranker-base on CPU took 3129 ms for top-20 against a
2000 ms production budget. Doc 10 assumed ONNX int8 would close the gap but never
validated it. Two cheaper levers first:

  1. int8 dynamic quantisation (torch built-in, no ONNX export needed)
  2. a much smaller cross-encoder (22M vs 278M)

If neither works, production drops reranking and serves fusion-ranked results —
which the degradation ladder already covers, but should be a measured decision
rather than a surprise.

Run:  .venv\\Scripts\\python.exe scripts/08_cpu_rerank.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import fitz
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "08_cpu_rerank.json"

BUDGET_MS = 2000  # doc 10 §10.3 production target
DEPTHS = (10, 20, 30)
QUERY = "Which methods perform best on long-document retrieval benchmarks?"


def load_chunks(n: int = 32) -> list[str]:
    chunks: list[str] = []
    for pdf in sorted((ROOT / "fixtures").glob("*.pdf")):
        if pdf.stem.startswith("_broken"):
            continue
        doc = fitz.open(pdf)
        buf = ""
        for page in doc:
            for para in page.get_text().split("\n\n"):
                para = " ".join(para.split())
                if not para:
                    continue
                buf += " " + para
                if len(buf) >= 1200:
                    chunks.append(buf.strip()[:1200])
                    buf = ""
                    if len(chunks) >= n:
                        break
            if len(chunks) >= n:
                break
        doc.close()
        if len(chunks) >= n:
            break
    return chunks[:n]


def bench(model_name: str, chunks: list[str], quantize: bool) -> dict[str, object]:
    model = CrossEncoder(model_name, device="cpu", max_length=512)

    if quantize:
        # Dynamic int8 on Linear layers — the standard CPU inference win for
        # transformer encoders. No ONNX export, no calibration data.
        model.model = torch.quantization.quantize_dynamic(
            model.model, {torch.nn.Linear}, dtype=torch.qint8
        )

    rec: dict[str, object] = {
        "model": model_name,
        "quantized": quantize,
        "threads": torch.get_num_threads(),
    }

    for depth in DEPTHS:
        pairs = [(QUERY, c) for c in chunks[:depth]]
        model.predict(pairs, batch_size=16, show_progress_bar=False)  # warm
        lat: list[float] = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict(pairs, batch_size=16, show_progress_bar=False)
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        rec[f"top{depth}_p50_ms"] = round(statistics.median(lat), 1)
        rec[f"top{depth}_max_ms"] = round(lat[-1], 1)

    del model
    return rec


def main() -> int:
    chunks = load_chunks()
    cpu_count = os.cpu_count()
    print(f"{len(chunks)} chunks | {cpu_count} logical CPUs | "
          f"torch threads {torch.get_num_threads()}\n")

    configs = [
        ("BAAI/bge-reranker-base", False),
        ("BAAI/bge-reranker-base", True),
        ("cross-encoder/ms-marco-MiniLM-L6-v2", False),
        ("cross-encoder/ms-marco-MiniLM-L6-v2", True),
    ]

    records: list[dict[str, object]] = []
    for name, quant in configs:
        label = f"{name.split('/')[-1]} {'int8' if quant else 'fp32'}"
        print(f"benchmarking {label} ...", flush=True)
        try:
            r = bench(name, chunks, quant)
            records.append(r)
            print("  " + "  ".join(f"top{d}={r[f'top{d}_p50_ms']:.0f}ms" for d in DEPTHS))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            records.append({"model": name, "quantized": quant, "error": repr(exc)[:200]})

    print(f"\n--- DEP-2: production CPU budget {BUDGET_MS} ms ---")
    print(f"{'config':44s} {'top10':>8s} {'top20':>8s} {'top30':>8s}")
    print("-" * 72)
    viable: list[str] = []
    for r in records:
        if "error" in r:
            continue
        label = f"{r['model'].split('/')[-1]} {'int8' if r['quantized'] else 'fp32'}"
        cells = []
        for d in DEPTHS:
            v = r[f"top{d}_p50_ms"]
            ok = v < BUDGET_MS
            cells.append(f"{v:7.0f}{'*' if ok else ' '}")
            if ok and d >= 20:
                viable.append(f"{label}@top{d}")
        print(f"{label:44s} " + " ".join(cells))

    print("\n* = within the production budget")
    if viable:
        print(f"\nPASS: viable production configs -> {', '.join(sorted(set(viable))[:4])}")
        print("  Recall cost of shallower reranking must still be measured in M2.")
    else:
        print("\nFAIL: no CPU config reranks >=20 candidates inside budget.")
        print("  Production should serve fusion-ranked results (degraded, disclosed),")
        print("  or the demo runs locally on GPU with a recorded walkthrough.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"budget_ms": BUDGET_MS, "viable": sorted(set(viable)), "records": records},
        indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
