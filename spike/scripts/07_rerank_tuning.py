"""M0 deliverable 4 (part B) — rescue NFR-P2.

The first benchmark showed the cross-encoder eating 739 ms of an 800 ms budget
that was meant to cover the whole retrieval stage. Before relaxing the budget,
check the levers that cost nothing in quality:

  1. fp16 instead of the fp32 default (should roughly halve latency)
  2. candidate depth: 50 -> 30 -> 20 -> 10  (trades recall ceiling for latency)
  3. a smaller cross-encoder as a speed reference point

Recall cost of shallower reranking is NOT measured here — that needs the labelled
eval set (M2). This only establishes what latency is achievable.

Run:  .venv\\Scripts\\python.exe scripts/07_rerank_tuning.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import fitz
import torch
from sentence_transformers import CrossEncoder

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "07_rerank_tuning.json"

MODELS = [
    ("BAAI/bge-reranker-base", "278M — the design's default"),
    ("cross-encoder/ms-marco-MiniLM-L6-v2", "22M — speed reference"),
]
DEPTHS = (10, 20, 30, 50)
QUERY = "Which methods perform best on long-document retrieval benchmarks?"


def vram_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024**3, 2)


def load_chunks(n: int = 64) -> list[str]:
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


def bench(model_name: str, chunks: list[str], fp16: bool) -> dict[str, object]:
    torch.cuda.empty_cache()
    before = vram_gb()

    kwargs: dict[str, object] = {"device": "cuda", "max_length": 512}
    if fp16:
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    model = CrossEncoder(model_name, **kwargs)
    weights = round(vram_gb() - before, 2)

    rec: dict[str, object] = {
        "model": model_name,
        "precision": "fp16" if fp16 else "fp32",
        "vram_weights_gb": weights,
    }

    for depth in DEPTHS:
        pairs = [(QUERY, c) for c in chunks[:depth]]
        for _ in range(3):  # warm
            model.predict(pairs, batch_size=32, show_progress_bar=False)
        torch.cuda.synchronize()

        lat: list[float] = []
        for _ in range(20):
            t0 = time.perf_counter()
            model.predict(pairs, batch_size=32, show_progress_bar=False)
            torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        rec[f"top{depth}_p50_ms"] = round(statistics.median(lat), 1)
        rec[f"top{depth}_p95_ms"] = round(lat[int(len(lat) * 0.95)], 1)

    del model
    torch.cuda.empty_cache()
    return rec


def main() -> int:
    chunks = load_chunks()
    print(f"{len(chunks)} chunks loaded\n")

    records: list[dict[str, object]] = []
    for name, note in MODELS:
        for fp16 in (False, True):
            print(f"benchmarking {name} [{'fp16' if fp16 else 'fp32'}] ...", flush=True)
            try:
                r = bench(name, chunks, fp16)
                r["note"] = note
                records.append(r)
                row = "  " + "  ".join(
                    f"top{d}={r[f'top{d}_p50_ms']:.0f}ms" for d in DEPTHS)
                print(f"{row}   weights={r['vram_weights_gb']} GB")
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
                records.append({"model": name, "precision": "fp16" if fp16 else "fp32",
                                "error": repr(exc)[:200]})

    print("\n--- against NFR-P2 (whole retrieval stage < 800 ms) ---")
    print("budget available to reranking, allowing ~120 ms for embed+search+fusion: 680 ms\n")
    print(f"{'model':40s} {'prec':5s} {'top20':>8s} {'top30':>8s} {'top50':>8s}")
    print("-" * 74)
    for r in records:
        if "error" in r:
            continue
        short = r["model"].split("/")[-1]
        flags = []
        for d in (20, 30, 50):
            v = r[f"top{d}_p95_ms"]
            flags.append(f"{v:7.0f}{'*' if v < 680 else ' '}")
        print(f"{short:40s} {r['precision']:5s} " + " ".join(flags))
    print("\n* = fits the reranking share of the budget")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"records": records}, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
