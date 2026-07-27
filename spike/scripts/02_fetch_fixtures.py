"""M0 deliverable 2 — build the paper fixture set.

Validates ASM-3 by giving the parser evaluation something realistic to fail on.
The set deliberately spans the categories that break naive PDF extraction:

  clean       two-column ML papers, well-formed
  tables      results-heavy papers where the facts live in tables
  long        100+ page reports that stress batching and memory
  broken      synthesised: image-only (scanned), encrypted, zero-text

The "broken" fixtures are generated locally rather than hunted for, so the set is
reproducible and the failure taxonomy (FR-I9) has deterministic test inputs.

Run:  .venv\\Scripts\\python.exe scripts/02_fetch_fixtures.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import fitz  # PyMuPDF
import httpx

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
RESULTS = ROOT / "results" / "02_fixtures.json"

# (arxiv_id, short name, category) — chosen for layout diversity, not fame.
PAPERS: list[tuple[str, str, str]] = [
    ("1706.03762", "attention-is-all-you-need", "clean"),
    ("1810.04805", "bert", "clean"),
    ("1908.10084", "sentence-bert", "clean"),
    ("2004.04906", "dpr", "clean"),
    ("2005.11401", "rag-lewis", "clean"),
    ("2112.09118", "contriever", "clean"),
    ("2201.11903", "chain-of-thought", "clean"),
    ("2310.11511", "self-rag", "clean"),
    ("2401.18059", "raptor", "clean"),
    ("2307.03172", "lost-in-the-middle", "tables"),
    ("2104.08663", "beir", "tables"),
    ("2212.03533", "e5-embeddings", "tables"),
    ("2309.07597", "bge-c-pack", "tables"),
    ("2302.13971", "llama", "tables"),
    ("2007.14062", "big-bird", "tables"),
    ("2105.03011", "qasper", "tables"),
    ("2005.14165", "gpt-3", "long"),
    ("2307.09288", "llama-2", "long"),
    ("2312.10997", "rag-survey", "long"),
    ("2303.08774", "gpt-4-report", "long"),
]

UA = "Scholiast-M0-Spike/0.1 (research prototype; contact: local)"


def fetch(client: httpx.Client, arxiv_id: str, name: str) -> tuple[bool, str, int]:
    dest = FIXTURES / f"{name}.pdf"
    if dest.exists() and dest.stat().st_size > 10_000:
        return True, "cached", dest.stat().st_size

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        r = client.get(url, follow_redirects=True, timeout=60.0)
        r.raise_for_status()
        if not r.content.startswith(b"%PDF"):
            return False, "not-a-pdf", 0
        dest.write_bytes(r.content)
        return True, "downloaded", len(r.content)
    except Exception as exc:  # noqa: BLE001 - spike diagnostic
        return False, f"error: {type(exc).__name__}", 0


def make_broken_fixtures(source: Path) -> list[dict[str, object]]:
    """Synthesise the pathological cases from a known-good PDF."""
    made: list[dict[str, object]] = []

    # 1. Image-only ("scanned"): rasterise 2 pages, no text layer at all.
    #    This is what FR-I9 / SCANNED_IMAGE must detect.
    try:
        src = fitz.open(source)
        out = fitz.open()
        for page_no in range(min(2, src.page_count)):
            pix = src[page_no].get_pixmap(dpi=150)
            page = out.new_page(width=pix.width, height=pix.height)
            page.insert_image(fitz.Rect(0, 0, pix.width, pix.height), pixmap=pix)
        dest = FIXTURES / "_broken-scanned.pdf"
        out.save(dest)
        out.close()
        src.close()
        made.append({"name": "_broken-scanned", "category": "broken", "kind": "image-only",
                     "bytes": dest.stat().st_size})
    except Exception as exc:  # noqa: BLE001
        made.append({"name": "_broken-scanned", "error": repr(exc)})

    # 2. Encrypted / password protected.
    try:
        src = fitz.open(source)
        dest = FIXTURES / "_broken-encrypted.pdf"
        src.save(dest, encryption=fitz.PDF_ENCRYPT_AES_256,
                 owner_pw="owner-secret", user_pw="user-secret")
        src.close()
        made.append({"name": "_broken-encrypted", "category": "broken", "kind": "encrypted",
                     "bytes": dest.stat().st_size})
    except Exception as exc:  # noqa: BLE001
        made.append({"name": "_broken-encrypted", "error": repr(exc)})

    # 3. Structurally valid but empty — zero extractable text.
    try:
        out = fitz.open()
        out.new_page()
        dest = FIXTURES / "_broken-empty.pdf"
        out.save(dest)
        out.close()
        made.append({"name": "_broken-empty", "category": "broken", "kind": "no-text",
                     "bytes": dest.stat().st_size})
    except Exception as exc:  # noqa: BLE001
        made.append({"name": "_broken-empty", "error": repr(exc)})

    return made


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    ok = 0

    with httpx.Client(headers={"User-Agent": UA}) as client:
        for arxiv_id, name, category in PAPERS:
            success, status, size = fetch(client, arxiv_id, name)
            records.append({
                "arxiv_id": arxiv_id, "name": name, "category": category,
                "ok": success, "status": status, "bytes": size,
            })
            flag = "ok " if success else "FAIL"
            print(f"{flag} {name:28s} {category:6s} {status:12s} {size/1e6:6.2f} MB")
            ok += success
            if status == "downloaded":
                time.sleep(1.0)  # be polite to arXiv (CON-7)

    # Pathological fixtures, built from a paper we know downloaded cleanly.
    seed = next((FIXTURES / f"{r['name']}.pdf" for r in records if r["ok"]), None)
    if seed is not None:
        print("\nsynthesising broken fixtures...")
        for rec in make_broken_fixtures(seed):
            records.append(rec)
            if "error" in rec:
                print(f"FAIL {rec['name']:28s} {rec['error']}")
            else:
                print(f"ok   {rec['name']:28s} {rec['kind']}")

    summary = {
        "downloaded_ok": ok,
        "requested": len(PAPERS),
        "total_fixtures": len([r for r in records if r.get("ok") or "kind" in r]),
        "records": records,
    }
    RESULTS.write_text(json.dumps(summary, indent=2))
    print(f"\n{ok}/{len(PAPERS)} papers + {len(records) - len(PAPERS)} synthetic")
    print(f"wrote {RESULTS}")
    return 0 if ok >= 15 else 1


if __name__ == "__main__":
    raise SystemExit(main())
