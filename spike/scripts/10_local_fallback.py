"""M0 deliverable 6 — local generation fallback (D15, D21, RSK-16).

The open question after REV-3: usable VRAM is ~6.8 GB, not 8 GB, and the
retrieval models already take 2.36 GB. That leaves ~4.4 GB — against a 7B Q4's
~4.7 GB plus KV cache. So the design's assumed fallback probably does not fit,
and doc 6 §6.16 lists three candidate resolutions.

This measures the real numbers for the leading candidate (a 3B-class model) and
answers four things:

  1. does it fit ALONGSIDE the retrieval models, in the degraded state that
     actually occurs? (loading it while nothing else is resident proves nothing)
  2. is generation fast enough to be a usable fallback?
  3. does CPU-only work — which would keep the GPU untouched in every state and
     preserve rerank latency even while degraded?
  4. does grammar-constrained JSON work, so FR-C4's schema is guaranteed rather
     than validated-and-repaired?

Run:  .venv\\Scripts\\python.exe scripts/10_local_fallback.py
"""

from __future__ import annotations

import json
import statistics
import subprocess
import time
from pathlib import Path

import httpx
import torch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "10_local_fallback.json"
OLLAMA = "http://127.0.0.1:11434"

#: Measured M0: total 7.96 GB, desktop compositor holds ~1.1-1.3 GB (REV-3).
USABLE_VRAM_GB = 6.8
#: Measured M0: embedder + reranker co-resident and active.
RETRIEVAL_MODELS_GB = 2.36

PROMPT = (
    "Answer using ONLY this evidence.\n\n"
    "[C1] Structure-aware chunking splits documents at section boundaries rather "
    "than at fixed token counts.\n"
    "[C2] The authors report nDCG@10 of 0.74 on their benchmark.\n\n"
    "Question: What nDCG did the authors report, and what chunking did they use? "
    "Cite with [Cn]."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "ndcg": {"type": "number"},
        "chunking": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ndcg", "chunking", "citations"],
}


def vram_used_gb() -> float:
    free, total = torch.cuda.mem_get_info()
    return round((total - free) / 1024**3, 2)


def ollama_ps() -> list[dict[str, str]]:
    """`ollama ps` reports the GPU/CPU split — the authoritative fit signal.

    A model that spills to CPU still 'works', just slowly, so this is what
    distinguishes 'fits' from 'runs'.
    """
    try:
        out = subprocess.run(  # noqa: S603
            ["ollama", "ps"], capture_output=True, text=True, timeout=30, check=False
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 5:
            rows.append({"name": parts[0], "size": f"{parts[2]} {parts[3]}",
                         "processor": " ".join(parts[4:-1]) or "?"})
    return rows


def generate(model: str, *, num_gpu: int | None = None,
             fmt: dict | None = None, timeout: float = 300.0) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model, "prompt": PROMPT, "stream": False,
        "options": {"temperature": 0, "num_predict": 200},
    }
    if num_gpu is not None:
        body["options"] = {**body["options"], "num_gpu": num_gpu}  # type: ignore[dict-item]
    if fmt is not None:
        body["format"] = fmt

    t0 = time.perf_counter()
    with httpx.Client(timeout=timeout) as c:
        r = c.post(f"{OLLAMA}/api/generate", json=body)
        r.raise_for_status()
        payload = r.json()
    elapsed = time.perf_counter() - t0

    eval_count = payload.get("eval_count", 0)
    eval_ns = payload.get("eval_duration", 1) or 1
    return {
        "wall_seconds": round(elapsed, 2),
        "tokens": eval_count,
        "tokens_per_second": round(eval_count / (eval_ns / 1e9), 1),
        "response": payload.get("response", ""),
    }


def load_retrieval_models() -> tuple[object, object]:
    """The degraded state is generation running *while retrieval still works*."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    rr = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                      model_kwargs={"torch_dtype": torch.float16})
    emb.encode(["warm up the allocator"], show_progress_bar=False)
    rr.predict([("q", "warm up the allocator")], show_progress_bar=False)
    torch.cuda.synchronize()
    return emb, rr


def main() -> int:
    out: dict[str, object] = {"usable_vram_gb": USABLE_VRAM_GB}

    # --- reachability -------------------------------------------------------
    try:
        with httpx.Client(timeout=10.0) as c:
            models = [m["name"] for m in c.get(f"{OLLAMA}/api/tags").json().get("models", [])]
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: Ollama not reachable at {OLLAMA} ({type(exc).__name__})")
        print("  Start it with: ollama serve")
        return 1

    if not models:
        print("FAIL: no models pulled. Try: ollama pull llama3.2")
        return 1

    model = models[0]
    out["model"] = model
    print(f"available models : {', '.join(models)}")
    print(f"benchmarking     : {model}\n")

    baseline = vram_used_gb()
    print(f"VRAM baseline (nothing loaded)      {baseline:.2f} GB")

    # --- 1. generation alone -----------------------------------------------
    print("\n--- generation alone (GPU) ---")
    alone = generate(model)
    vram_alone = vram_used_gb()
    ps_alone = ollama_ps()
    out["alone"] = {**alone, "vram_total_used_gb": vram_alone, "ollama_ps": ps_alone}
    print(f"  {alone['tokens_per_second']} tok/s, {alone['tokens']} tokens "
          f"in {alone['wall_seconds']}s")
    print(f"  VRAM in use                       {vram_alone:.2f} GB")
    for row in ps_alone:
        print(f"  ollama ps: {row['name']} {row['size']} -> {row['processor']}")

    # --- 2. THE question: co-resident with the retrieval models -------------
    print("\n--- degraded state: generation + retrieval models co-resident ---")
    emb, rr = load_retrieval_models()
    vram_retrieval = vram_used_gb()
    print(f"  retrieval models loaded           {vram_retrieval:.2f} GB")

    together = generate(model)
    vram_peak = vram_used_gb()
    ps_together = ollama_ps()

    # Rerank latency while generation holds the GPU — the number that says
    # whether the degraded state is usable or merely survivable.
    passages = ["Structure-aware chunking splits at section boundaries."] * 20
    lat = []
    for _ in range(5):
        t0 = time.perf_counter()
        rr.predict([("test query", p) for p in passages], show_progress_bar=False)  # type: ignore[attr-defined]
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1000)

    out["co_resident"] = {
        **together,
        "vram_retrieval_only_gb": vram_retrieval,
        "vram_peak_gb": vram_peak,
        "rerank20_ms_p50": round(statistics.median(lat), 1),
        "ollama_ps": ps_together,
    }
    print(f"  {together['tokens_per_second']} tok/s while co-resident")
    print(f"  VRAM peak                         {vram_peak:.2f} GB "
          f"of {USABLE_VRAM_GB} GB usable")
    print(f"  rerank top-20 during generation   {statistics.median(lat):.0f} ms")
    for row in ps_together:
        print(f"  ollama ps: {row['name']} {row['size']} -> {row['processor']}")

    del emb, rr
    torch.cuda.empty_cache()

    # --- 3. CPU-only --------------------------------------------------------
    print("\n--- CPU-only generation (keeps the GPU untouched) ---")
    try:
        cpu = generate(model, num_gpu=0, timeout=600.0)
        out["cpu_only"] = cpu
        print(f"  {cpu['tokens_per_second']} tok/s, {cpu['wall_seconds']}s wall")
    except Exception as exc:  # noqa: BLE001
        out["cpu_only"] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        print(f"  FAILED: {type(exc).__name__}")

    # --- 4. grammar-constrained JSON (FR-C4) --------------------------------
    print("\n--- schema-constrained output ---")
    try:
        structured = generate(model, fmt=SCHEMA)
        parsed = json.loads(str(structured["response"]))
        ok = all(key in parsed for key in SCHEMA["required"])
        out["structured_output"] = {"ok": ok, "parsed": parsed}
        print(f"  {'ok' if ok else 'PARTIAL'}: {parsed}")
    except Exception as exc:  # noqa: BLE001
        out["structured_output"] = {"ok": False, "error": f"{type(exc).__name__}"}
        print(f"  FAILED: {type(exc).__name__}: {exc}")

    # --- verdicts -----------------------------------------------------------
    headroom = USABLE_VRAM_GB - RETRIEVAL_MODELS_GB
    model_cost = vram_peak - vram_retrieval
    fits = vram_peak < USABLE_VRAM_GB
    on_gpu = all("cpu" not in r["processor"].lower() for r in ps_together) if ps_together else None

    print("\n--- D21 / RSK-16 verdict ---")
    print(f"  headroom after retrieval models   {headroom:.2f} GB")
    print(f"  generation model cost             {model_cost:.2f} GB")
    print(f"  peak co-resident                  {vram_peak:.2f} GB "
          f"-> {'FITS' if fits else 'DOES NOT FIT'}")
    if on_gpu is not None:
        print(f"  fully GPU-resident                {'yes' if on_gpu else 'NO — spilled to CPU'}")

    out["verdict"] = {
        "headroom_gb": round(headroom, 2),
        "model_cost_gb": round(model_cost, 2),
        "fits_co_resident": fits,
        "fully_gpu_resident": on_gpu,
    }

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
