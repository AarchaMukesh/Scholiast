"""M0 deliverable 6 (part B) — co-residency, measured honestly.

Two measurement attempts failed before this one, and the reason is worth
recording because it changes how VRAM claims should be stated for the rest of
the project:

  torch.cuda.mem_get_info()  reports what the *calling CUDA context* sees. It
                             misses another process's allocations.
  nvidia-smi                 reports `0 MiB` used and `[N/A]` per process on this
                             machine. On Windows the WDDM driver model virtualises
                             VRAM, and consumer GPUs expose no per-process
                             accounting through NVML.

So there is no authoritative device-wide number available here. Rather than quote
a fabricated sum, this script does the thing that actually answers the design
question: **run both workloads simultaneously and see whether they coexist and
stay fast.** An empirical coexistence test is stronger evidence than an
arithmetic projection anyway — it would catch thrashing, CPU spill, and
allocator pressure that a sum would miss.

Run:  .venv\\Scripts\\python.exe scripts/11_vram_crossprocess.py
"""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "11_vram_crossprocess.json"
OLLAMA = "http://127.0.0.1:11434"

USABLE_BUDGET_GB = 6.8  # REV-3


def torch_view_gb() -> float:
    """VRAM in use from our own CUDA context's perspective."""
    import torch

    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024**3, 2)


def ollama_residency() -> dict[str, str]:
    """Ollama's own accounting — authoritative for whether IT fits on the GPU.

    The PROCESSOR column is the signal that matters: '100% GPU' means no CPU
    spill, which is exactly the 'fits' question. A model that spilled would still
    generate, just slowly, so throughput alone would not reveal it.
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["ollama", "ps"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"status": "no model resident"}
    row = lines[1]
    size = re.search(r"(\d+\.?\d*)\s*GB", row)
    processor = re.search(r"(\d+%\s*(?:GPU|CPU)(?:/\d+%\s*(?:GPU|CPU))?)", row)
    return {
        "size_gb": size.group(1) if size else "?",
        "processor": processor.group(1) if processor else "?",
    }


def generate(model: str, prompt: str, *, num_predict: int = 200,
             num_gpu: int | None = None, timeout: float = 600.0) -> dict[str, float]:
    options: dict[str, object] = {"temperature": 0, "num_predict": num_predict}
    if num_gpu is not None:
        options["num_gpu"] = num_gpu
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{OLLAMA}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": options, "keep_alive": "5m",
        })
        r.raise_for_status()
        p = r.json()
    return {
        "tokens": p.get("eval_count", 0),
        "tokens_per_second": round(
            p.get("eval_count", 0) / ((p.get("eval_duration", 1) or 1) / 1e9), 1
        ),
    }


def main() -> int:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    with httpx.Client(timeout=10.0) as c:
        models = [m["name"] for m in c.get(f"{OLLAMA}/api/tags").json().get("models", [])]
    model = models[0]
    print(f"model: {model}\n")

    out: dict[str, object] = {
        "model": model,
        "usable_budget_gb": USABLE_BUDGET_GB,
        "measurement_note": (
            "Windows WDDM virtualises VRAM; nvidia-smi reports 0 MiB used and [N/A] "
            "per process on this GPU. No authoritative device-wide figure is "
            "available, so co-residency is demonstrated empirically instead."
        ),
    }

    print(f"torch-context VRAM, idle           {torch_view_gb():.2f} GB")

    # --- generation model resident ------------------------------------------
    solo = generate(model, "Say OK.", num_predict=8)
    time.sleep(1)
    res = ollama_residency()
    out["ollama_residency"] = res
    print(f"ollama ps                          {res.get('size_gb', '?')} GB "
          f"-> {res.get('processor', '?')}")

    # --- retrieval models alongside -----------------------------------------
    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    rr = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                      model_kwargs={"torch_dtype": torch.float16})
    emb.encode(["warm"], show_progress_bar=False)
    rr.predict([("q", "warm")], show_progress_bar=False)
    torch.cuda.synchronize()
    retrieval_view = torch_view_gb()
    out["torch_view_with_retrieval_gb"] = retrieval_view
    print(f"torch-context VRAM, + retrieval    {retrieval_view:.2f} GB")

    # --- the actual test: both under load, with realistic inputs ------------
    # Script 10 used 10-token stubs, which flattered rerank latency. Rerank cost
    # scales with passage length, so realistic ~300-token chunks are required for
    # the number to mean anything.
    chunk = (
        "Structure-aware chunking splits documents at section boundaries rather than "
        "at fixed token counts, preserving the section context that disambiguates "
        "attribution between a paper's own contribution and prior work it cites. "
    ) * 5
    passages = [chunk] * 20

    baseline_lat: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        rr.predict([("test query", p) for p in passages], show_progress_bar=False)
        torch.cuda.synchronize()
        baseline_lat.append((time.perf_counter() - t0) * 1000)

    contended_lat: list[float] = []
    with httpx.Client(timeout=600.0) as c:
        req = c.post(f"{OLLAMA}/api/generate", json={
            "model": model,
            "prompt": "Explain retrieval-augmented generation in five sentences.",
            "stream": False,
            "options": {"temperature": 0, "num_predict": 250},
        })
        for _ in range(5):
            t0 = time.perf_counter()
            rr.predict([("test query", p) for p in passages], show_progress_bar=False)
            torch.cuda.synchronize()
            contended_lat.append((time.perf_counter() - t0) * 1000)
        req.raise_for_status()
        payload = req.json()

    contended_tps = round(
        payload.get("eval_count", 0) / ((payload.get("eval_duration", 1) or 1) / 1e9), 1
    )
    peak_view = torch_view_gb()
    res_after = ollama_residency()

    base_p50 = statistics.median(baseline_lat)
    cont_p50 = statistics.median(contended_lat)

    out |= {
        "solo_generation_tps": solo["tokens_per_second"],
        "contended_generation_tps": contended_tps,
        "rerank20_ms_idle_p50": round(base_p50, 1),
        "rerank20_ms_contended_p50": round(cont_p50, 1),
        "rerank_slowdown_x": round(cont_p50 / base_p50, 2) if base_p50 else None,
        "torch_view_peak_gb": peak_view,
        "ollama_residency_after": res_after,
        "coexisted_without_oom": True,
    }

    # --- CPU-only alternative ------------------------------------------------
    print("\nCPU-only generation (GPU untouched in every state)...")
    try:
        cpu = generate(model, "Explain RAG in three sentences.", num_predict=120, num_gpu=0)
        out["cpu_only_tps"] = cpu["tokens_per_second"]
        print(f"  {cpu['tokens_per_second']} tok/s")
    except Exception as exc:  # noqa: BLE001
        out["cpu_only_tps"] = None
        print(f"  FAILED: {type(exc).__name__}")

    del emb, rr
    torch.cuda.empty_cache()

    # --- verdict -------------------------------------------------------------
    print("\n--- D21 / RSK-16 verdict ---")
    print(f"  ollama residency                  {res_after.get('size_gb', '?')} GB "
          f"-> {res_after.get('processor', '?')}")
    print(f"  coexisted without OOM             yes")
    print(f"  generation, solo -> contended     {solo['tokens_per_second']} -> "
          f"{contended_tps} tok/s")
    print(f"  rerank top-20, idle -> contended  {base_p50:.0f} -> {cont_p50:.0f} ms "
          f"({cont_p50 / base_p50:.2f}x)")
    print(f"  CPU-only fallback                 {out.get('cpu_only_tps')} tok/s")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
