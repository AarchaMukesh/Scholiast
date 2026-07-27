"""M0 deliverable 3 (part A) — PyMuPDF-only parsing baseline.

GROBID needs Docker + Java, neither of which is installed yet, so this measures
the *fallback* path first. That ordering is deliberate: knowing how good
PyMuPDF-plus-heuristics is on its own tells us how much GROBID actually has to
buy before it justifies a Java container in the stack (doc 7 §7.4).

Two questions:
  ASM-3   what fraction of real ML papers yield usable section structure?
  FR-I9   are the pathological documents detected rather than silently indexed
          as garbage?

Section detection combines two weak signals — a heading-shaped line, and a font
larger/bolder than body text. Neither alone is reliable on two-column layouts.

Run:  .venv\\Scripts\\python.exe scripts/03_parse_baseline.py
"""

from __future__ import annotations

import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
RESULTS = ROOT / "results" / "03_parse_baseline.json"

# Canonical taxonomy from doc 6 §6.2. `related_work` matters most: claims there
# belong to *other* papers, and misattributing them is the failure the whole
# contextual-header design exists to prevent.
CANONICAL: dict[str, list[str]] = {
    "abstract": ["abstract"],
    "introduction": ["introduction", "overview"],
    "related_work": ["related work", "background", "prior work", "related research"],
    "method": ["method", "methodology", "approach", "model", "architecture",
               "our approach", "proposed method"],
    "experiments": ["experiment", "experimental setup", "setup", "evaluation",
                    "empirical", "implementation details"],
    "results": ["result", "findings", "main results"],
    "discussion": ["discussion", "analysis", "ablation"],
    "limitations": ["limitation", "broader impact", "ethical"],
    "conclusion": ["conclusion", "concluding remarks", "future work", "summary"],
    "references": ["reference", "bibliography"],
    "appendix": ["appendix", "supplementary"],
}

# A heading line: optional numbering, then Title Case / ALL CAPS, and short.
HEADING_RE = re.compile(
    r"^\s*(?:(?:\d{1,2}(?:\.\d{1,2})*\.?)|(?:[IVXLC]{1,5}\.)|(?:[A-Z]\.))?\s*"
    r"([A-Z][A-Za-z][A-Za-z \-&:,']{2,60})\s*$"
)


def classify_section(title: str) -> str | None:
    low = title.lower().strip()
    for canon, keys in CANONICAL.items():
        if any(k in low for k in keys):
            return canon
    return None


def analyse(path: Path) -> dict[str, object]:
    rec: dict[str, object] = {"name": path.stem}
    t0 = time.perf_counter()

    # --- Open / encryption -------------------------------------------------
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        rec |= {"verdict": "unusable", "failure_class": "UNPARSEABLE",
                "detail": type(exc).__name__}
        return rec

    if doc.needs_pass:
        doc.close()
        rec |= {"verdict": "unusable", "failure_class": "UNPARSEABLE",
                "detail": "encrypted / password required"}
        return rec

    rec["pages"] = doc.page_count

    # --- Text extraction with font metadata --------------------------------
    sizes: list[float] = []
    lines: list[tuple[str, float, bool]] = []  # (text, size, bold)
    total_chars = 0

    for page in doc:
        d = page.get_text("dict")
        for block in d.get("blocks", []):
            for line in block.get("lines", []):
                text = "".join(s.get("text", "") for s in line.get("spans", []))
                if not text.strip():
                    continue
                total_chars += len(text)
                spans = line.get("spans", [])
                size = max((s.get("size", 0.0) for s in spans), default=0.0)
                bold = any("bold" in str(s.get("font", "")).lower() for s in spans)
                sizes.append(size)
                lines.append((text.strip(), size, bold))

    elapsed = time.perf_counter() - t0
    doc.close()

    rec |= {
        "chars": total_chars,
        "chars_per_page": round(total_chars / max(rec["pages"], 1)),
        "parse_seconds": round(elapsed, 2),
    }

    # --- Zero-text detection (scanned / image-only) ------------------------
    if total_chars < 200:
        rec |= {"verdict": "unusable", "failure_class": "SCANNED_IMAGE",
                "detail": f"only {total_chars} extractable chars"}
        return rec

    # --- Heading detection --------------------------------------------------
    body = statistics.median(sizes) if sizes else 0.0
    rec["body_font_size"] = round(body, 2)

    found: list[dict[str, object]] = []
    for text, size, bold in lines:
        if len(text) > 70:
            continue
        m = HEADING_RE.match(text)
        if not m:
            continue
        # Require a typographic signal too — heading-shaped text alone matches
        # far too many ordinary lines in two-column papers.
        if not (size > body * 1.04 or bold):
            continue
        canon = classify_section(m.group(1))
        found.append({"raw": text, "canonical": canon, "size": round(size, 2)})

    canon_hits = [f["canonical"] for f in found if f["canonical"]]
    distinct = sorted(set(canon_hits))
    rec |= {
        "headings_detected": len(found),
        "canonical_sections": distinct,
        "n_canonical": len(distinct),
    }

    # --- Verdict ------------------------------------------------------------
    # "Usable" = enough structure to support section-aware chunking and correct
    # attribution. The core four are the ones downstream logic depends on.
    core = {"abstract", "introduction", "method", "experiments", "results",
            "conclusion", "related_work"}
    core_hits = core & set(distinct)

    if len(core_hits) >= 4:
        verdict = "usable"
    elif len(core_hits) >= 2:
        verdict = "partial"
    else:
        verdict = "poor"

    rec |= {"verdict": verdict, "core_sections_found": sorted(core_hits)}
    return rec


def main() -> int:
    pdfs = sorted(FIXTURES.glob("*.pdf"))
    if not pdfs:
        print("No fixtures. Run 02_fetch_fixtures.py first.")
        return 1

    records = [analyse(p) for p in pdfs]

    print(f"{'fixture':30s} {'pages':>5s} {'chars':>8s} {'sec':>5s} {'canon':>5s}  verdict")
    print("-" * 78)
    for r in records:
        print(f"{r['name']:30s} {r.get('pages', 0):5d} {r.get('chars', 0):8d} "
              f"{r.get('parse_seconds', 0):5.2f} {r.get('n_canonical', 0):5d}  "
              f"{r['verdict']}"
              + (f"  [{r.get('failure_class')}]" if r.get("failure_class") else ""))

    real = [r for r in records if not r["name"].startswith("_broken")]
    broken = [r for r in records if r["name"].startswith("_broken")]

    verdicts = Counter(r["verdict"] for r in real)
    usable = verdicts["usable"]
    rate = usable / len(real) if real else 0.0

    print("\n--- ASM-3: structure extraction on real papers ---")
    for v in ("usable", "partial", "poor", "unusable"):
        if verdicts[v]:
            print(f"  {v:9s} {verdicts[v]:2d}/{len(real)}")
    print(f"  usable rate: {rate:.0%}   (ASM-3 bar: >=85%)")
    print(f"  {'PASS' if rate >= 0.85 else 'FAIL — GROBID likely required'}")

    print("\n--- FR-I9: pathological document detection ---")
    for r in broken:
        detected = r["verdict"] == "unusable"
        print(f"  {'ok  ' if detected else 'MISS'} {r['name']:24s} "
              f"-> {r.get('failure_class', 'NOT DETECTED')}")

    avg_ms = statistics.mean(r.get("parse_seconds", 0) for r in real) * 1000
    print(f"\nmean parse time: {avg_ms:.0f} ms/paper")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(
        {"usable_rate": rate, "verdicts": dict(verdicts), "records": records}, indent=2))
    print(f"wrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
