"""M0 deliverable 6 (part D) — unmanaged GPU contention baseline (§5.7).

Doc 5 §5.7 argues that one GPU with two consumers of opposite latency profiles is
the most important resource problem in the design, and proposes a priority
semaphore plus bounded ingestion batches. That scheme is built in M3. This
measures what happens *without* it — the baseline the design exists to improve on.

Script 11's contention figure (51 -> 58 ms, 1.15x) was measured while generation
had been silently placed on CPU, so it understated the problem. Here generation is
confirmed GPU-resident before the contended measurement is taken.

Two contended workloads:
  rerank while generating   the query path competing with the fallback generator
  rerank while embedding    the query path competing with bulk ingestion, which
                            is the case the semaphore actually targets

Run:  .venv\\Scripts\\python.exe scripts/13_gpu_contention.py
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "13_gpu_contention.json"
OLLAMA = "http://127.0.0.1:11434"

CHUNK = (
    "Structure-aware chunking splits documents at section boundaries rather than at "
    "fixed token counts, preserving the section context that disambiguates attribution "
    "between a paper's own contribution and prior work it cites. "
) * 5


def placement() -> str:
    out = subprocess.run(  # noqa: S603
        ["ollama", "ps"], capture_output=True, text=True, timeout=30, check=False
    ).stdout
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return "not resident"
    m = re.search(r"(\d+%\s*(?:GPU|CPU)(?:/\d+%\s*(?:GPU|CPU))?)", lines[1])
    return m.group(1) if m else "?"


def measure_rerank(rr: object, passages: list[str], runs: int = 7) -> list[float]:
    import torch

    lat: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        rr.predict([("which method performs best?", p) for p in passages],  # type: ignore[attr-defined]
                   show_progress_bar=False)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)
    return lat


def main() -> int:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with httpx.Client(timeout=10.0) as c:
        model = [m["name"] for m in c.get(f"{OLLAMA}/api/tags").json().get("models", [])][0]

    # Start from a clean placement so the sticky-keep_alive trap cannot skew this.
    subprocess.run(["ollama", "stop", model], capture_output=True, timeout=60, check=False)  # noqa: S603
    time.sleep(3)

    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    rr = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                      model_kwargs={"torch_dtype": torch.float16})
    passages = [CHUNK] * 20
    emb.encode(["warm"], show_progress_bar=False)
    measure_rerank(rr, passages, runs=3)  # warm

    out: dict[str, object] = {"model": model}

    # --- baseline: GPU otherwise idle ---------------------------------------
    idle = measure_rerank(rr, passages)
    idle_p50 = statistics.median(idle)
    print(f"rerank top-20, GPU idle              {idle_p50:6.0f} ms")
    out["rerank_idle_p50_ms"] = round(idle_p50, 1)

    # --- contended by generation --------------------------------------------
    with httpx.Client(timeout=600.0) as c:
        c.post(f"{OLLAMA}/api/generate", json={
            "model": model, "prompt": "Say OK.", "stream": False,
            "options": {"num_predict": 8, "temperature": 0}, "keep_alive": "5m",
        }).raise_for_status()
    place = placement()
    print(f"generation placement                 {place}")
    out["generation_placement"] = place

    gen_lat: list[float] = []
    gen_done: dict[str, float] = {}

    def run_generation() -> None:
        with httpx.Client(timeout=600.0) as c:
            r = c.post(f"{OLLAMA}/api/generate", json={
                "model": model,
                "prompt": "Explain retrieval-augmented generation in eight sentences.",
                "stream": False, "options": {"num_predict": 400, "temperature": 0},
            })
            p = r.json()
            gen_done["tps"] = round(
                p.get("eval_count", 0) / ((p.get("eval_duration", 1) or 1) / 1e9), 1
            )

    t = threading.Thread(target=run_generation)
    t.start()
    time.sleep(0.4)  # let generation actually get on the GPU
    gen_lat = measure_rerank(rr, passages)
    t.join()

    gen_p50 = statistics.median(gen_lat)
    print(f"rerank top-20, during generation     {gen_p50:6.0f} ms  "
          f"({gen_p50 / idle_p50:.2f}x)   generation {gen_done.get('tps')} tok/s")
    out |= {
        "rerank_during_generation_p50_ms": round(gen_p50, 1),
        "generation_slowdown_x": round(gen_p50 / idle_p50, 2),
        "generation_tps_contended": gen_done.get("tps"),
    }

    # --- contended by bulk embedding (the case the semaphore targets) -------
    corpus = [CHUNK] * 2000
    stop = threading.Event()

    def run_embedding() -> None:
        while not stop.is_set():
            emb.encode(corpus[:256], batch_size=64, show_progress_bar=False)

    t2 = threading.Thread(target=run_embedding)
    t2.start()
    time.sleep(1.0)
    emb_lat = measure_rerank(rr, passages)
    stop.set()
    t2.join()

    emb_p50 = statistics.median(emb_lat)
    print(f"rerank top-20, during embedding      {emb_p50:6.0f} ms  "
          f"({emb_p50 / idle_p50:.2f}x)")
    out |= {
        "rerank_during_embedding_p50_ms": round(emb_p50, 1),
        "embedding_slowdown_x": round(emb_p50 / idle_p50, 2),
    }

    del emb, rr
    torch.cuda.empty_cache()

    # --- verdict -------------------------------------------------------------
    budget = 800  # NFR-P2, whole retrieval stage
    worst = max(gen_p50, emb_p50)
    print("\n--- §5.7 unmanaged-contention baseline ---")
    print(f"  worst contended rerank            {worst:.0f} ms")
    print(f"  NFR-P2 whole-stage budget         {budget} ms")
    print(f"  -> {'within budget even unmanaged' if worst < budget * 0.85 else 'MANAGEMENT NEEDED'}")
    print("\n  This is the baseline the M3 priority semaphore is measured against.")

    out["verdict"] = {
        "worst_contended_ms": round(worst, 1),
        "nfr_p2_budget_ms": budget,
        "needs_management": worst >= budget * 0.85,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
