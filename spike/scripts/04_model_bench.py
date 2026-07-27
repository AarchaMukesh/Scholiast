"""M0 deliverable 4 — embedding + reranking benchmark.

Validates ASM-4 and the two budgets the retrieval design hangs on:

  NFR-P2  retrieval stage (incl. cross-encoder rerank of ~50 candidates) p95 <= 800 ms
  NFR-P7  embedding throughput >= 200 chunks/s on GPU

Also measures the CPU path, because production has no GPU (DEP-2) and the
deployed instance needs its own honest latency target rather than a borrowed one.

Chunks come from the real fixture PDFs, not synthetic text — token length
distribution drives both throughput and rerank cost, so synthetic input would
flatter the numbers.

Run:  .venv\\Scripts\\python.exe scripts/04_model_bench.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import fitz
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
RESULTS = ROOT / "results" / "04_model_bench.json"

EMBEDDERS = ["BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]
RERANKER = "BAAI/bge-reranker-base"

TARGET_CHUNK_CHARS = 1200   # ~300 tokens, matching the design's child-chunk size
N_CHUNKS = 512
N_RERANK_CANDIDATES = 50


def vram_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024**3, 2)


def build_chunks() -> list[str]:
    """Realistic chunk corpus from the fixture papers."""
    chunks: list[str] = []
    for pdf in sorted(FIXTURES.glob("*.pdf")):
        if pdf.stem.startswith("_broken"):
            continue
        try:
            doc = fitz.open(pdf)
        except Exception:  # noqa: BLE001
            continue
        buf = ""
        for page in doc:
            for para in page.get_text().split("\n\n"):
                para = " ".join(para.split())
                if not para:
                    continue
                buf += " " + para
                if len(buf) >= TARGET_CHUNK_CHARS:
                    chunks.append(buf.strip()[:TARGET_CHUNK_CHARS])
                    buf = ""
            if len(chunks) >= N_CHUNKS:
                break
        doc.close()
        if len(chunks) >= N_CHUNKS:
            break
    return chunks[:N_CHUNKS]


def bench_embedder(name: str, chunks: list[str], device: str) -> dict[str, object]:
    torch.cuda.empty_cache() if device == "cuda" else None
    before = vram_used_gb() if device == "cuda" else 0.0

    t0 = time.perf_counter()
    model = SentenceTransformer(name, device=device)
    load_s = time.perf_counter() - t0
    after_load = vram_used_gb() if device == "cuda" else 0.0

    dim = model.get_sentence_embedding_dimension()
    rec: dict[str, object] = {
        "model": name, "device": device, "dim": dim,
        "load_seconds": round(load_s, 2),
        "vram_weights_gb": round(after_load - before, 2) if device == "cuda" else None,
    }

    # Throughput across batch sizes — the ingestion path is throughput-bound and
    # batch size is the lever we trade against query-path latency (§5.7).
    results = {}
    for bs in (8, 16, 32, 64):
        model.encode(chunks[:bs], batch_size=bs, show_progress_bar=False)  # warm
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.encode(chunks, batch_size=bs, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
        if device == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        results[bs] = round(len(chunks) / el, 1)
    rec["throughput_chunks_per_s"] = results
    rec["peak_vram_gb"] = vram_used_gb() if device == "cuda" else None

    # Single-query latency — this is the query path, not ingestion.
    lat = []
    q = "What chunking strategy does this paper use for long documents?"
    for _ in range(20):
        t0 = time.perf_counter()
        model.encode([q], show_progress_bar=False)
        if device == "cuda":
            torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    rec["query_embed_ms_p50"] = round(statistics.median(lat), 1)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rec


def bench_reranker(chunks: list[str], device: str) -> dict[str, object]:
    if device == "cuda":
        torch.cuda.empty_cache()
    before = vram_used_gb() if device == "cuda" else 0.0

    t0 = time.perf_counter()
    model = CrossEncoder(RERANKER, device=device, max_length=512)
    load_s = time.perf_counter() - t0
    after_load = vram_used_gb() if device == "cuda" else 0.0

    query = "Which methods perform best on long-document retrieval benchmarks?"
    rec: dict[str, object] = {
        "model": RERANKER, "device": device,
        "load_seconds": round(load_s, 2),
        "vram_weights_gb": round(after_load - before, 2) if device == "cuda" else None,
    }

    for n in (20, 50):
        pairs = [(query, c) for c in chunks[:n]]
        model.predict(pairs, show_progress_bar=False)  # warm
        if device == "cuda":
            torch.cuda.synchronize()
        lat = []
        runs = 15 if device == "cuda" else 5
        for _ in range(runs):
            t0 = time.perf_counter()
            model.predict(pairs, batch_size=32, show_progress_bar=False)
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        rec[f"rerank_{n}_ms_p50"] = round(statistics.median(lat), 1)
        rec[f"rerank_{n}_ms_p95"] = round(lat[min(int(len(lat) * 0.95), len(lat) - 1)], 1)

    rec["peak_vram_gb"] = vram_used_gb() if device == "cuda" else None
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return rec


def bench_coresident(chunks: list[str]) -> dict[str, object]:
    """Both models resident at once — the §5.7 co-residency assumption."""
    torch.cuda.empty_cache()
    base = vram_used_gb()
    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    rr = CrossEncoder(RERANKER, device="cuda", max_length=512)

    emb.encode(chunks[:32], show_progress_bar=False)
    rr.predict([("q", c) for c in chunks[:50]], show_progress_bar=False)
    torch.cuda.synchronize()
    peak = vram_used_gb()

    del emb, rr
    torch.cuda.empty_cache()
    return {"baseline_gb": base, "both_resident_peak_gb": peak,
            "models_cost_gb": round(peak - base, 2)}


def main() -> int:
    print("building chunk corpus from fixtures...")
    chunks = build_chunks()
    lens = [len(c) for c in chunks]
    print(f"{len(chunks)} chunks, mean {statistics.mean(lens):.0f} chars\n")

    out: dict[str, object] = {
        "n_chunks": len(chunks),
        "mean_chunk_chars": round(statistics.mean(lens)),
        "vram_total_gb": round(torch.cuda.mem_get_info()[1] / 1024**3, 2),
        "vram_baseline_used_gb": vram_used_gb(),
    }
    print(f"VRAM total {out['vram_total_gb']} GB, "
          f"already used by desktop {out['vram_baseline_used_gb']} GB\n")

    # --- Embedders (GPU) ----------------------------------------------------
    out["embedders"] = []
    for name in EMBEDDERS:
        print(f"benchmarking {name} (cuda)...")
        r = bench_embedder(name, chunks, "cuda")
        out["embedders"].append(r)
        best = max(r["throughput_chunks_per_s"].values())
        print(f"  dim={r['dim']}  weights={r['vram_weights_gb']} GB  "
              f"best={best} chunks/s  query={r['query_embed_ms_p50']} ms")

    # --- Embedder (CPU, for the GPU-less production path) -------------------
    print(f"\nbenchmarking {EMBEDDERS[1]} (cpu)...")
    r_cpu = bench_embedder(EMBEDDERS[1], chunks[:128], "cpu")
    out["embedder_cpu"] = r_cpu
    print(f"  best={max(r_cpu['throughput_chunks_per_s'].values())} chunks/s")

    # --- Reranker -----------------------------------------------------------
    print(f"\nbenchmarking {RERANKER} (cuda)...")
    rr_gpu = bench_reranker(chunks, "cuda")
    out["reranker_gpu"] = rr_gpu
    print(f"  weights={rr_gpu['vram_weights_gb']} GB  "
          f"top20={rr_gpu['rerank_20_ms_p50']} ms  top50={rr_gpu['rerank_50_ms_p50']} ms")

    print(f"\nbenchmarking {RERANKER} (cpu)...")
    rr_cpu = bench_reranker(chunks, "cpu")
    out["reranker_cpu"] = rr_cpu
    print(f"  top20={rr_cpu['rerank_20_ms_p50']} ms  top50={rr_cpu['rerank_50_ms_p50']} ms")

    # --- Co-residency -------------------------------------------------------
    print("\nco-residency check...")
    co = bench_coresident(chunks)
    out["coresident"] = co
    print(f"  both models peak {co['both_resident_peak_gb']} GB "
          f"(models cost {co['models_cost_gb']} GB)")

    # --- Verdicts -----------------------------------------------------------
    base_emb = next(e for e in out["embedders"] if "base" in e["model"])
    thr = max(base_emb["throughput_chunks_per_s"].values())
    p95_gpu = rr_gpu["rerank_50_ms_p95"]
    p95_cpu_20 = rr_cpu["rerank_20_ms_p50"]

    print("\n--- verdicts ---")
    print(f"NFR-P7  embedding >=200 chunks/s : {thr} -> "
          f"{'PASS' if thr >= 200 else 'FAIL'}")
    print(f"NFR-P2  rerank 50 p95 <800 ms    : {p95_gpu} ms -> "
          f"{'PASS' if p95_gpu < 800 else 'FAIL'}")
    print(f"DEP-2   CPU rerank top-20        : {p95_cpu_20} ms "
          f"(prod budget 2000 ms) -> {'PASS' if p95_cpu_20 < 2000 else 'FAIL'}")
    free_after = out["vram_total_gb"] - out["vram_baseline_used_gb"]
    print(f"ASM-4   models fit in {free_after:.2f} GB usable: "
          f"{co['models_cost_gb']} GB -> "
          f"{'PASS' if co['models_cost_gb'] < free_after else 'FAIL'}")

    out["verdicts"] = {
        "nfr_p7_embedding": thr >= 200,
        "nfr_p2_rerank_gpu": p95_gpu < 800,
        "dep2_rerank_cpu_top20": p95_cpu_20 < 2000,
        "asm4_fits": co["models_cost_gb"] < free_after,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
