"""M0 deliverable 5 (part B) — diagnose the two provider findings.

Two things the first pass surfaced that need resolving before D14 can close:

  Gemini returned 429, not 401. A rejected key gives 401, so the credential was
  *accepted* and the project has no usable quota. Worth distinguishing, because
  the fixes are completely different (mint a proper key vs. enable the API).

  Groq rejected `response_format: json_schema` with 400. Groq supports JSON mode
  broadly but strict schema only on some models, so FR-C4's approach on the API
  path depends on which variant actually works.

Also records the real rate limits, which look generous per-request and are not:
12,000 tokens/minute against a ~5-7k-token RAG request is ~2 requests/minute.

Run:  .venv\\Scripts\\python.exe scripts/14_provider_diagnose.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "14_provider_diagnose.json"

SCHEMA = {
    "type": "object",
    "properties": {
        "ndcg": {"type": "number"},
        "chunking": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["ndcg", "chunking", "citations"],
    "additionalProperties": False,
}

PROMPT = (
    "Answer using ONLY this evidence.\n\n"
    "[C1] Structure-aware chunking splits documents at section boundaries.\n"
    "[C2] The authors report nDCG@10 of 0.74.\n\n"
    "Return the nDCG, the chunking strategy, and the citation labels you used."
)


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


def diagnose_gemini() -> dict[str, object]:
    key = os.getenv("GEMINI_API_KEY", "")
    rec: dict[str, object] = {
        "key_prefix": key[:4] if key else None,
        "looks_like_api_key": key.startswith("AIza"),
    }
    if not key:
        return rec | {"status": "not configured"}

    url = "https://generativelanguage.googleapis.com/v1beta/models"
    try:
        # Listing models is the cheapest possible authenticated call — it
        # separates "bad credential" from "no quota for generation".
        with httpx.Client(timeout=30.0) as c:
            r = c.get(url, params={"key": key})
        rec["list_models_status"] = r.status_code
        if r.status_code == 200:
            names = [m["name"] for m in r.json().get("models", [])][:5]
            rec["reachable"] = True
            rec["sample_models"] = names
        else:
            rec["reachable"] = False
            rec["error"] = r.text[:900]
    except Exception as exc:  # noqa: BLE001
        rec |= {"reachable": False, "error": f"{type(exc).__name__}: {exc}"[:200]}
    return rec


def groq_structured(model: str, mode: str) -> dict[str, object]:
    key = os.getenv("GROQ_API_KEY", "")
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 300,
        "temperature": 0,
    }
    if mode == "json_object":
        body["response_format"] = {"type": "json_object"}
    elif mode == "json_schema":
        body["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": SCHEMA},
        }

    try:
        with httpx.Client(timeout=90.0) as c:
            r = c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json=body,
            )
        if r.status_code != 200:
            return {"mode": mode, "model": model, "ok": False,
                    "status": r.status_code, "error": r.text[:280]}
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {"mode": mode, "model": model, "ok": True,
                "parsed": parsed,
                "has_required": all(k in parsed for k in SCHEMA["required"])}
    except Exception as exc:  # noqa: BLE001
        return {"mode": mode, "model": model, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"[:200]}


def groq_limits() -> dict[str, str]:
    key = os.getenv("GROQ_API_KEY", "")
    with httpx.Client(timeout=60.0) as c:
        r = c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                  "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5},
        )
    return {k: v for k, v in r.headers.items() if "ratelimit" in k.lower()}


def main() -> int:
    load_env()
    out: dict[str, object] = {}

    # --- Gemini -------------------------------------------------------------
    print("=== Gemini credential diagnosis ===")
    g = diagnose_gemini()
    out["gemini"] = g
    print(f"  key prefix              {g.get('key_prefix')!r} "
          f"(AI Studio keys start 'AIza')")
    print(f"  looks like an API key   {g.get('looks_like_api_key')}")
    print(f"  list-models status      {g.get('list_models_status')}")
    if g.get("reachable"):
        print(f"  credential VALID — models visible: {g.get('sample_models')}")
        print("  -> 429 on generate means the project has no generation quota.")
    else:
        print(f"  error: {str(g.get('error'))[:400]}")

    # --- Groq structured output --------------------------------------------
    print("\n=== Groq structured output (FR-C4) ===")
    models = [os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"), "llama-3.1-8b-instant"]
    attempts = []
    for model in models:
        for mode in ("json_object", "json_schema"):
            res = groq_structured(model, mode)
            attempts.append(res)
            flag = "ok  " if res.get("ok") else "FAIL"
            detail = res.get("parsed") if res.get("ok") else str(res.get("error", ""))[:90]
            print(f"  {flag} {model:26s} {mode:12s} {detail}")
    out["groq_structured"] = attempts

    # --- Groq rate limits ---------------------------------------------------
    print("\n=== Groq rate limits (the real capacity constraint) ===")
    limits = groq_limits()
    out["groq_limits"] = limits
    for k, v in sorted(limits.items()):
        print(f"  {k:36s} {v}")

    tpm = limits.get("x-ratelimit-limit-tokens")
    if tpm:
        per_request = 6000  # ~4-6k evidence + question + answer
        print(f"\n  A RAG request is ~{per_request} tokens (evidence + question + answer).")
        print(f"  At {tpm} tokens/min that is ~{int(tpm) // per_request} requests/minute "
              f"sustained.")
        print("  -> quota, not CPU or RAM, is the binding capacity constraint (NFR-S4).")

    working = 1 + (1 if g.get("reachable") else 0)
    print(f"\n--- D14 status: {working} usable provider(s) ---")
    if working < 2:
        print("  Chain still has no secondary rung. Cheapest fixes, in order:")
        print("   1. mint a proper Gemini API key at https://aistudio.google.com/apikey")
        print("   2. add a Cerebras or OpenRouter key (both free, ~2 min)")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
