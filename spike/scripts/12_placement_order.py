"""M0 deliverable 6 (part C) — does the fallback actually get the GPU?

Scripts 10 and 11 disagreed: the same model reported `100% GPU` at 128 tok/s in
one run and `100% CPU` at ~21 tok/s in the next. The difference was load order.
That is not a measurement artefact — it is the behaviour that decides D21.

The degraded state is specific: the retrieval models are ALREADY resident (they
serve every query, degraded or not) and generation loads *after* them. So the
question is not "does a 3B fit on an empty GPU" — it obviously does — but
"does Ollama still place it on the GPU once 2 GB is already spoken for, and if
not, does it tell us?"

Two orderings, one variable:

  A. generation first, retrieval second   (what script 10 happened to do)
  B. retrieval first, generation second   (what actually happens in production)

Run:  .venv\\Scripts\\python.exe scripts/12_placement_order.py
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "12_placement_order.json"
OLLAMA = "http://127.0.0.1:11434"


def ollama_placement() -> dict[str, str]:
    out = subprocess.run(  # noqa: S603
        ["ollama", "ps"], capture_output=True, text=True, timeout=30, check=False
    ).stdout
    lines = [ln for ln in out.splitlines() if ln.strip()]
    if len(lines) < 2:
        return {"processor": "not resident"}
    row = lines[1]
    size = re.search(r"(\d+\.?\d*)\s*GB", row)
    proc = re.search(r"(\d+%\s*(?:GPU|CPU)(?:/\d+%\s*(?:GPU|CPU))?)", row)
    return {
        "size_gb": size.group(1) if size else "?",
        "processor": proc.group(1) if proc else "?",
    }


def unload(model: str) -> None:
    """Force the model out so each ordering starts from the same state."""
    subprocess.run(  # noqa: S603
        ["ollama", "stop", model], capture_output=True, text=True, timeout=60, check=False
    )
    time.sleep(3)


def generate(model: str, prompt: str, num_predict: int = 120) -> float:
    with httpx.Client(timeout=600.0) as c:
        r = c.post(f"{OLLAMA}/api/generate", json={
            "model": model, "prompt": prompt, "stream": False,
            "options": {"temperature": 0, "num_predict": num_predict},
            "keep_alive": "5m",
        })
        r.raise_for_status()
        p = r.json()
    return round(p.get("eval_count", 0) / ((p.get("eval_duration", 1) or 1) / 1e9), 1)


def load_retrieval() -> tuple[object, object]:
    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer

    emb = SentenceTransformer("BAAI/bge-base-en-v1.5", device="cuda")
    rr = CrossEncoder("BAAI/bge-reranker-base", device="cuda", max_length=512,
                      model_kwargs={"torch_dtype": torch.float16})
    emb.encode(["warm"], show_progress_bar=False)
    rr.predict([("q", "warm")], show_progress_bar=False)
    torch.cuda.synchronize()
    return emb, rr


def main() -> int:
    import torch

    with httpx.Client(timeout=10.0) as c:
        model = [m["name"] for m in c.get(f"{OLLAMA}/api/tags").json().get("models", [])][0]
    print(f"model: {model}\n")
    out: dict[str, object] = {"model": model}

    # --- ordering A: generation first ---------------------------------------
    print("A. generation loads FIRST (empty GPU)")
    unload(model)
    tps_a = generate(model, "Explain retrieval-augmented generation briefly.")
    place_a = ollama_placement()
    print(f"   placement {place_a['processor']:<12} {tps_a} tok/s")

    emb, rr = load_retrieval()
    print(f"   torch view after retrieval load: {round((torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3, 2)} GB")
    tps_a2 = generate(model, "Explain vector search briefly.")
    place_a2 = ollama_placement()
    print(f"   after retrieval loaded: {place_a2['processor']:<12} {tps_a2} tok/s")
    out["ordering_a"] = {
        "placement_before": place_a, "tps_before": tps_a,
        "placement_after": place_a2, "tps_after": tps_a2,
    }

    del emb, rr
    torch.cuda.empty_cache()
    time.sleep(2)

    # --- ordering B: retrieval first (the real degraded state) --------------
    print("\nB. retrieval loads FIRST, then generation  <- the production case")
    unload(model)
    emb, rr = load_retrieval()
    used = round(
        (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 1024**3, 2
    )
    print(f"   retrieval resident: {used} GB (torch view)")

    tps_b = generate(model, "Explain retrieval-augmented generation briefly.")
    place_b = ollama_placement()
    print(f"   placement {place_b['processor']:<12} {tps_b} tok/s")
    out["ordering_b"] = {
        "retrieval_gb_torch_view": used,
        "placement": place_b,
        "tps": tps_b,
    }

    del emb, rr
    torch.cuda.empty_cache()

    # --- verdict -------------------------------------------------------------
    def on_gpu(p: dict[str, str]) -> bool:
        return "GPU" in p.get("processor", "") and "CPU" not in p.get("processor", "")

    b_gpu = on_gpu(place_b)
    print("\n--- D21 / RSK-16 verdict ---")
    print(f"  A: empty GPU first          {place_a['processor']:<12} {tps_a} tok/s")
    print(f"  A: after retrieval loaded   {place_a2['processor']:<12} {tps_a2} tok/s")
    print(f"  B: retrieval first          {place_b['processor']:<12} {tps_b} tok/s")
    print()
    if b_gpu:
        print("  -> 3B fallback gets the GPU even with retrieval models resident.")
    else:
        print("  -> Ollama places the fallback on CPU once retrieval models hold VRAM.")
        print("     It does NOT fail or warn — throughput just drops ~6x.")
        print("     Consequence: the local-generation rung must report its actual")
        print("     placement, or a silent CPU fallback masquerades as a working")
        print("     GPU path (NFR-R4 requires degradation to be disclosed).")

    out["verdict"] = {
        "fallback_gets_gpu_in_degraded_state": b_gpu,
        "silent_cpu_fallback": not b_gpu,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
