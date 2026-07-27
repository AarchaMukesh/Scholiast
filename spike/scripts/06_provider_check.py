"""M0 deliverable 5 — free-tier LLM provider evaluation (D14).

Answers the questions that decide the generation architecture:
  - which free tiers actually work, and what are the real rate limits?
  - latency and streaming behaviour (FR-Q5 / NFR-P3)
  - does the provider support server-side JSON-schema output (FR-C4)?
  - what are the data-use terms? (RSK-15 — cannot be probed, must be read)

Deliberately uses raw httpx rather than provider SDKs: the point is to compare
providers behind one interface, which is exactly what the LLMProvider port does
in the real system.

Setup — copy .env.example to .env and fill in whichever keys you have. Every
provider is optional; unconfigured ones are skipped, not failed.

Run:  .venv\\Scripts\\python.exe scripts/06_provider_check.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "06_providers.json"
ENV = ROOT / ".env"

PROMPT = (
    "Answer using ONLY this evidence.\n\n"
    "[C1] Structure-aware chunking splits documents at section boundaries "
    "rather than at fixed token counts.\n"
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
    "additionalProperties": False,
}

# OpenAI-compatible providers: same request shape, different base URL.
OPENAI_COMPAT: dict[str, dict[str, str]] = {
    "groq": {
        "env": "GROQ_API_KEY",
        "base": "https://api.groq.com/openai/v1",
        "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    },
    "cerebras": {
        "env": "CEREBRAS_API_KEY",
        "base": "https://api.cerebras.ai/v1",
        "model": os.getenv("CEREBRAS_MODEL", "llama-3.3-70b"),
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "base": "https://openrouter.ai/api/v1",
        "model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
    },
    "mistral": {
        "env": "MISTRAL_API_KEY",
        "base": "https://api.mistral.ai/v1",
        "model": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    },
}


def load_env() -> None:
    if not ENV.exists():
        return
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def rate_limit_headers(headers: httpx.Headers) -> dict[str, str]:
    """Providers advertise quota differently; capture anything limit-shaped."""
    keys = [k for k in headers
            if "ratelimit" in k.lower() or "rate-limit" in k.lower()
            or k.lower() in {"retry-after", "x-request-id"}]
    return {k: headers[k] for k in keys}


def probe_openai_compat(name: str, cfg: dict[str, str]) -> dict[str, object]:
    key = os.getenv(cfg["env"])
    if not key:
        return {"provider": name, "configured": False,
                "hint": f"set {cfg['env']} in spike/.env"}

    rec: dict[str, object] = {"provider": name, "configured": True,
                              "model": cfg["model"], "base": cfg["base"]}
    url = f"{cfg['base']}/chat/completions"
    hdrs = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": cfg["model"],
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 300, "temperature": 0}

    # --- basic completion + latency ----------------------------------------
    lat: list[float] = []
    text = ""
    try:
        with httpx.Client(timeout=90.0) as c:
            for i in range(3):
                t0 = time.perf_counter()
                r = c.post(url, headers=hdrs, json=body)
                lat.append((time.perf_counter() - t0) * 1000)
                if i == 0:
                    rec["status"] = r.status_code
                    rec["rate_limit_headers"] = rate_limit_headers(r.headers)
                if r.status_code != 200:
                    rec["ok"] = False
                    rec["error"] = r.text[:300]
                    return rec
                text = r.json()["choices"][0]["message"]["content"]
                time.sleep(0.5)
        rec["ok"] = True
        rec["latency_ms_median"] = round(statistics.median(lat))
        rec["sample_answer"] = text[:220]
        rec["cited_correctly"] = ("0.74" in text) and ("C2" in text)
    except Exception as exc:  # noqa: BLE001
        rec["ok"] = False
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return rec

    # --- streaming (FR-Q5) --------------------------------------------------
    try:
        with httpx.Client(timeout=90.0) as c:
            t0 = time.perf_counter()
            ttft = None
            chunks = 0
            with c.stream("POST", url, headers=hdrs,
                          json={**body, "stream": True}) as r:
                for line in r.iter_lines():
                    if line.startswith("data:") and "[DONE]" not in line:
                        if ttft is None:
                            ttft = (time.perf_counter() - t0) * 1000
                        chunks += 1
            rec["streaming"] = True
            rec["ttft_ms"] = round(ttft) if ttft else None
            rec["stream_chunks"] = chunks
    except Exception as exc:  # noqa: BLE001
        rec["streaming"] = False
        rec["streaming_error"] = f"{type(exc).__name__}"[:80]

    # --- structured output (FR-C4) ------------------------------------------
    try:
        with httpx.Client(timeout=90.0) as c:
            r = c.post(url, headers=hdrs, json={
                **body,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "strict": True,
                                    "schema": SCHEMA},
                },
            })
        if r.status_code == 200:
            payload = r.json()["choices"][0]["message"]["content"]
            json.loads(payload)  # must actually parse
            rec["structured_output"] = "native_json_schema"
        else:
            rec["structured_output"] = f"unsupported (HTTP {r.status_code})"
    except Exception as exc:  # noqa: BLE001
        rec["structured_output"] = f"failed: {type(exc).__name__}"

    return rec


def probe_gemini() -> dict[str, object]:
    key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    if not key:
        return {"provider": "gemini", "configured": False,
                "hint": "set GEMINI_API_KEY in spike/.env"}

    rec: dict[str, object] = {"provider": "gemini", "configured": True, "model": model}
    base = "https://generativelanguage.googleapis.com/v1beta"
    url = f"{base}/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": PROMPT}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 300}}

    lat: list[float] = []
    try:
        with httpx.Client(timeout=90.0) as c:
            for i in range(3):
                t0 = time.perf_counter()
                r = c.post(url, params={"key": key}, json=body)
                lat.append((time.perf_counter() - t0) * 1000)
                if i == 0:
                    rec["status"] = r.status_code
                    rec["rate_limit_headers"] = rate_limit_headers(r.headers)
                if r.status_code != 200:
                    rec["ok"] = False
                    rec["error"] = r.text[:300]
                    return rec
                text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
                time.sleep(0.5)
        rec |= {"ok": True, "latency_ms_median": round(statistics.median(lat)),
                "sample_answer": text[:220],
                "cited_correctly": ("0.74" in text) and ("C2" in text)}
    except Exception as exc:  # noqa: BLE001
        return rec | {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

    # Gemini exposes JSON-schema output via responseSchema.
    try:
        gem_schema = {k: v for k, v in SCHEMA.items() if k != "additionalProperties"}
        with httpx.Client(timeout=90.0) as c:
            r = c.post(url, params={"key": key}, json={
                **body,
                "generationConfig": {**body["generationConfig"],
                                     "responseMimeType": "application/json",
                                     "responseSchema": gem_schema},
            })
        if r.status_code == 200:
            json.loads(r.json()["candidates"][0]["content"]["parts"][0]["text"])
            rec["structured_output"] = "native_response_schema"
        else:
            rec["structured_output"] = f"unsupported (HTTP {r.status_code})"
    except Exception as exc:  # noqa: BLE001
        rec["structured_output"] = f"failed: {type(exc).__name__}"

    # Streaming endpoint is a different verb.
    try:
        with httpx.Client(timeout=90.0) as c:
            t0 = time.perf_counter()
            ttft = None
            with c.stream("POST", f"{base}/models/{model}:streamGenerateContent",
                          params={"key": key, "alt": "sse"}, json=body) as r:
                for line in r.iter_lines():
                    if line.startswith("data:") and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1000
            rec |= {"streaming": True, "ttft_ms": round(ttft) if ttft else None}
    except Exception as exc:  # noqa: BLE001
        rec |= {"streaming": False, "streaming_error": type(exc).__name__}

    return rec


def main() -> int:
    load_env()
    records = [probe_gemini()]
    for name, cfg in OPENAI_COMPAT.items():
        records.append(probe_openai_compat(name, cfg))

    print(f"{'provider':12s} {'cfg':4s} {'ok':4s} {'lat_ms':>7s} {'ttft':>6s}  structured")
    print("-" * 74)
    for r in records:
        if not r.get("configured"):
            print(f"{r['provider']:12s} {'-':4s} {'skip':4s} {'':>7s} {'':>6s}  "
                  f"({r.get('hint', '')})")
            continue
        ok = "ok" if r.get("ok") else "FAIL"
        print(f"{r['provider']:12s} {'yes':4s} {ok:4s} "
              f"{r.get('latency_ms_median', 0):7} {r.get('ttft_ms') or 0:6}  "
              f"{r.get('structured_output', '-')}")
        if not r.get("ok"):
            print(f"             error: {r.get('error', '')[:100]}")
        elif not r.get("cited_correctly"):
            print("             WARN: did not cite [C2]/0.74 correctly — "
                  "check instruction-following before trusting it for generation")

    working = [r for r in records if r.get("ok")]
    print(f"\n--- D14 verdict ---")
    print(f"  {len(working)} provider(s) working")
    if len(working) >= 2:
        print("  -> PASS: provider chain viable (primary + secondary)")
    elif len(working) == 1:
        print("  -> PARTIAL: only one provider; the chain has no secondary rung")
    else:
        print("  -> BLOCKED: no working provider. Local fallback becomes primary.")
    print("\n  RSK-15 reminder: data-use terms cannot be probed — read each")
    print("  provider's policy and record it in SPIKE-FINDINGS.md.")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"n_working": len(working), "records": records},
                                  indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
