"""M0 addendum — does the OpenAI key have usable quota? (D14 exception check)

OpenAI is paid, so this is a deliberate exception to D14 ("no paid inference").
Before granting it any role, establish three things cheaply:

  1. does the key authenticate?          GET /v1/models — free, no tokens
  2. does it have usable quota?          one ~10-token completion
  3. what does that cost?                so the ceiling can be set from data

The Gemini credential taught the lesson this script encodes: **a key can
authenticate perfectly and still have zero quota.** A 401 means "bad key" and a
429 means "right key, no quota" — telling them apart saved a pointless re-paste
last time.

Intended role if it passes: **eval judge only** (doc 9 §9.6). Offline, ~50
examples for the judge-agreement study, never on the request path. Local NLI
stays the primary groundedness metric (D16).

Setup: add OPENAI_API_KEY to spike/.env yourself — do not paste it into a chat.

Run:  ..\\.venv\\Scripts\\python.exe scripts/16_openai_probe.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "16_openai_probe.json"
BASE = "https://api.openai.com/v1"

#: Judge candidates, cheapest-capable first. Probed against the live model list
#: rather than assumed — the catalogue moves.
JUDGE_PREFERENCES = (
    "gpt-5-mini", "gpt-5", "o4-mini", "gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1", "gpt-4o",
)

#: A real judging task in miniature: does the evidence support the claim?
#: Cheaper than a synthetic ping and tells us whether it can actually do the job.
JUDGE_PROMPT = (
    "Evidence: The authors report nDCG@10 of 0.74 using structure-aware chunking.\n"
    "Claim: The authors achieved nDCG@10 of 0.74.\n\n"
    "Is the claim supported by the evidence? Reply with exactly one word: "
    "SUPPORTED or UNSUPPORTED."
)


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v:
                os.environ.setdefault(k.strip(), v)


def main() -> int:
    load_env()
    key = os.getenv("OPENAI_API_KEY", "").strip()

    out: dict[str, object] = {}

    if not key:
        print("OPENAI_API_KEY is not set.")
        print("\n  Add it to spike/.env yourself:")
        print("      OPENAI_API_KEY=sk-...")
        print("  (do not paste it into the chat — two keys are already exposed there)")
        return 1

    out["key_prefix"] = key[:7]
    print(f"key prefix: {key[:7]}…   (never logged in full)\n")
    headers = {"Authorization": f"Bearer {key}"}

    # --- 1. authentication (free) -------------------------------------------
    try:
        with httpx.Client(timeout=30.0) as c:
            r = c.get(f"{BASE}/models", headers=headers)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: could not reach the API — {type(exc).__name__}: {exc}")
        return 1

    out["models_status"] = r.status_code
    if r.status_code == 401:
        print("FAIL 401 — the key is invalid or revoked. Mint a new one.")
        out["verdict"] = "invalid_key"
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1
    if r.status_code != 200:
        print(f"FAIL {r.status_code}: {r.text[:200]}")
        out["verdict"] = f"http_{r.status_code}"
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    available = sorted({m["id"] for m in r.json().get("data", [])})
    out["n_models"] = len(available)
    print(f"authenticated OK — {len(available)} models visible")

    judge = os.getenv("OPENAI_JUDGE_MODEL") or next(
        (m for m in JUDGE_PREFERENCES if m in available), None
    )
    if judge is None:
        judge = next((m for m in available if "gpt" in m and "instruct" not in m), None)
    if judge is None:
        print("FAIL: no usable chat model in the account's catalogue.")
        out["verdict"] = "no_usable_model"
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    out["judge_model"] = judge
    print(f"judge candidate: {judge}")
    matched = [m for m in JUDGE_PREFERENCES if m in available]
    if matched:
        print(f"  (preferred models present: {', '.join(matched[:5])})")

    # --- 2. quota (one tiny request) ----------------------------------------
    print("\nprobing quota with a single ~10-token judging call...")
    try:
        with httpx.Client(timeout=90.0) as c:
            r = c.post(
                f"{BASE}/chat/completions",
                headers=headers,
                json={
                    "model": judge,
                    "messages": [{"role": "user", "content": JUDGE_PROMPT}],
                    "max_completion_tokens": 16,
                },
            )
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {type(exc).__name__}: {exc}")
        out["verdict"] = "request_failed"
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    out["completion_status"] = r.status_code

    if r.status_code == 429:
        # The Gemini lesson: authenticated but unusable.
        print("\n429 — the key authenticates but has NO usable quota.")
        print("  Same shape as the Gemini credential: right key, empty project.")
        print("  Add credit at platform.openai.com/settings/organization/billing,")
        print("  or leave D14 as-is and keep the free-tier-only design.")
        print(f"\n  detail: {r.text[:200]}")
        out["verdict"] = "authenticated_no_quota"
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(out, indent=2))
        return 2

    if r.status_code != 200:
        print(f"\nFAIL {r.status_code}: {r.text[:300]}")
        out["verdict"] = f"http_{r.status_code}"
        RESULTS.write_text(json.dumps(out, indent=2))
        return 1

    payload = r.json()
    answer = payload["choices"][0]["message"]["content"].strip()
    usage = payload.get("usage", {})
    prompt_toks = usage.get("prompt_tokens", 0)
    completion_toks = usage.get("completion_tokens", 0)

    out |= {
        "verdict": "usable",
        "answer": answer,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion_toks,
        "judged_correctly": answer.upper().startswith("SUPPORTED"),
    }

    print(f"\n  response: {answer!r}")
    print(f"  tokens:   {prompt_toks} in / {completion_toks} out")
    print(f"  correct:  {out['judged_correctly']} (expected SUPPORTED)")

    # --- 3. what would the judge study actually cost? -----------------------
    # doc 9 §9.6: ~50 examples for the judge-agreement study. Judging an answer
    # needs the evidence set too, so budget ~2k prompt tokens per example rather
    # than this probe's ~60.
    per_example_prompt = 2000
    per_example_completion = 150
    n = 50
    print("\n--- projected cost of the judge-agreement study (doc 9 §9.6) ---")
    print(f"  {n} examples x ~{per_example_prompt} prompt + "
          f"~{per_example_completion} completion tokens")
    print(f"  = ~{n * per_example_prompt // 1000}k prompt / "
          f"~{n * per_example_completion // 1000}k completion tokens per full run")
    print("  Well under a dollar on any current mini-tier model — but set the")
    print("  ceiling from the account's real pricing, not from this estimate.")

    print("\n--- verdict ---")
    print(f"  key usable: YES — judge model {judge}")
    print("  Recommended scope: EVAL JUDGE ONLY (offline, budgeted).")
    print("  Local NLI stays the primary groundedness metric (D16); this")
    print("  sharpens judge-validation and the reported Cohen's κ (RSK-4).")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
