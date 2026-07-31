"""M0 deliverable 3 (part B) — GROBID vs PyMuPDF, settling D19.

Script 03 measured PyMuPDF + heading heuristics at 95% "usable structure" and
that number downgraded GROBID from required to pending (REV-5). But it carried an
explicit caveat: **the metric was heading *detection*, a proxy.** Finding the
string "3 Method" is not the same as knowing where that section starts and stops.

So this compares the things the proxy could not see:

  section BOUNDARIES   PyMuPDF finds headings; it does not know what text belongs
                       to which section. Chunking depends on boundaries, not
                       headings — and attribution (doc 2 §2.3) depends on chunking.
  metadata             title / authors / abstract as structured fields
  references           parsed bibliography vs. an undifferentiated text blob
  figure/table captions
  cost                 GROBID is a Java container and is slower — that is the
                       price D19 has to justify

Requires GROBID running:
  docker run -d --rm -p 8070:8070 --name grobid grobid/grobid:0.8.1

Run:  .venv\\Scripts\\python.exe scripts/15_grobid_compare.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module

# Reuse the baseline's taxonomy so the section comparison is apples-to-apples.
_baseline = import_module("03_parse_baseline")
classify_section = _baseline.classify_section

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
RESULTS = ROOT / "results" / "15_grobid_compare.json"
GROBID = "http://localhost:8070"
TEI = {"tei": "http://www.tei-c.org/ns/1.0"}


def grobid_alive() -> bool:
    try:
        with httpx.Client(timeout=10.0) as c:
            return c.get(f"{GROBID}/api/isalive").status_code == 200
    except Exception:  # noqa: BLE001
        return False


def process(pdf: Path) -> tuple[str | None, float, str | None]:
    """Full-text extraction. Returns (tei_xml, seconds, error)."""
    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=300.0) as c:
            r = c.post(
                f"{GROBID}/api/processFulltextDocument",
                files={"input": (pdf.name, pdf.read_bytes(), "application/pdf")},
                data={"consolidateHeader": "0", "consolidateCitations": "0",
                      "includeRawCitations": "0"},
            )
        elapsed = time.perf_counter() - t0
        if r.status_code != 200:
            return None, elapsed, f"HTTP {r.status_code}: {r.text[:120]}"
        return r.text, elapsed, None
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - t0, f"{type(exc).__name__}: {exc}"[:160]


def analyse_tei(xml: str) -> dict[str, object]:
    root = ET.fromstring(xml)  # noqa: S314 - GROBID output, locally generated

    def text_of(el: ET.Element | None) -> str:
        return " ".join("".join(el.itertext()).split()) if el is not None else ""

    title = text_of(root.find(".//tei:titleStmt/tei:title", TEI))
    authors = root.findall(".//tei:sourceDesc//tei:author/tei:persName", TEI)
    abstract = text_of(root.find(".//tei:profileDesc/tei:abstract", TEI))

    # Sections WITH their content — the thing the heading heuristic cannot give.
    sections: list[dict[str, object]] = []
    for div in root.findall(".//tei:text/tei:body/tei:div", TEI):
        head = text_of(div.find("tei:head", TEI))
        body = " ".join(text_of(p) for p in div.findall("tei:p", TEI))
        if not head and not body:
            continue
        sections.append({
            "head": head,
            "canonical": classify_section(head) if head else None,
            "chars": len(body),
        })

    refs = root.findall(".//tei:back//tei:listBibl/tei:biblStruct", TEI)
    figures = root.findall(".//tei:figure", TEI)

    canonical = sorted({s["canonical"] for s in sections if s["canonical"]})  # type: ignore[misc]
    body_chars = sum(int(s["chars"]) for s in sections)  # type: ignore[arg-type]

    return {
        "title": title[:120],
        "has_title": bool(title.strip()),
        "n_authors": len(authors),
        "has_abstract": len(abstract) > 50,
        "abstract_chars": len(abstract),
        "n_sections": len(sections),
        "n_sections_with_body": sum(1 for s in sections if int(s["chars"]) > 0),  # type: ignore[arg-type]
        "canonical_sections": canonical,
        "n_canonical": len(canonical),
        "body_chars": body_chars,
        "n_references": len(refs),
        "n_figures": len(figures),
    }


def main() -> int:
    if not grobid_alive():
        print("FAIL: GROBID is not responding at " + GROBID)
        print("  docker run -d --rm -p 8070:8070 --name grobid grobid/grobid:0.8.1")
        return 1
    print(f"GROBID alive at {GROBID}\n")

    baseline_path = ROOT / "results" / "03_parse_baseline.json"
    baseline = {r["name"]: r for r in json.loads(baseline_path.read_text())["records"]}

    pdfs = sorted(p for p in FIXTURES.glob("*.pdf") if not p.stem.startswith("_broken"))
    broken = sorted(FIXTURES.glob("_broken*.pdf"))

    records: list[dict[str, object]] = []
    print(f"{'fixture':28s} {'sec':>4s} {'canon':>6s} {'refs':>5s} {'figs':>5s} "
          f"{'body_kc':>8s} {'sec_s':>6s}   vs PyMuPDF")
    print("-" * 92)

    for pdf in pdfs:
        xml, secs, err = process(pdf)
        base = baseline.get(pdf.stem, {})
        rec: dict[str, object] = {
            "name": pdf.stem,
            "grobid_seconds": round(secs, 1),
            "pymupdf_canonical": base.get("n_canonical"),
            "pymupdf_verdict": base.get("verdict"),
        }
        if err or xml is None:
            rec |= {"ok": False, "error": err}
            print(f"{pdf.stem:28s} {'FAILED':>36s}   {err}")
        else:
            a = analyse_tei(xml)
            rec |= {"ok": True, **a}
            delta = int(a["n_canonical"]) - int(base.get("n_canonical") or 0)  # type: ignore[arg-type]
            print(f"{pdf.stem:28s} {a['n_sections']:4d} {a['n_canonical']:6d} "
                  f"{a['n_references']:5d} {a['n_figures']:5d} "
                  f"{int(a['body_chars']) // 1000:8d} {secs:6.1f}   "  # type: ignore[arg-type]
                  f"{base.get('n_canonical', '?')} canon ({delta:+d})")
        records.append(rec)

    # --- pathological documents: does GROBID also reject them cleanly? -------
    print("\npathological fixtures:")
    broken_recs = []
    for pdf in broken:
        xml, secs, err = process(pdf)
        if err or xml is None:
            outcome, detail = "rejected", err
        else:
            a = analyse_tei(xml)
            usable = int(a["body_chars"]) > 500  # type: ignore[arg-type]
            outcome = "EXTRACTED CONTENT" if usable else "empty result"
            detail = f"{a['body_chars']} body chars"
        broken_recs.append({"name": pdf.stem, "outcome": outcome, "detail": str(detail)[:120]})
        print(f"  {pdf.stem:24s} {outcome:20s} {str(detail)[:70]}")

    # --- verdict -------------------------------------------------------------
    ok = [r for r in records if r.get("ok")]
    if not ok:
        print("\nNo successful extractions — cannot compare.")
        return 1

    g_canon = statistics.mean(int(r["n_canonical"]) for r in ok)  # type: ignore[arg-type]
    p_canon = statistics.mean(
        int(r["pymupdf_canonical"] or 0) for r in ok  # type: ignore[arg-type]
    )
    with_body = sum(1 for r in ok if int(r["n_sections_with_body"]) > 0)  # type: ignore[arg-type]
    with_title = sum(1 for r in ok if r["has_title"])
    with_abstract = sum(1 for r in ok if r["has_abstract"])
    with_refs = sum(1 for r in ok if int(r["n_references"]) > 0)  # type: ignore[arg-type]
    with_figs = sum(1 for r in ok if int(r["n_figures"]) > 0)  # type: ignore[arg-type]
    mean_s = statistics.mean(float(r["grobid_seconds"]) for r in ok)  # type: ignore[arg-type]
    n = len(ok)

    print(f"\n--- D19: what GROBID adds over PyMuPDF ({n}/{len(pdfs)} extracted) ---")
    print(f"  canonical sections/paper   GROBID {g_canon:.1f}  vs  PyMuPDF {p_canon:.1f}")
    print(f"  sections WITH body text    {with_body}/{n}   <- PyMuPDF cannot do this at all")
    print(f"  structured title           {with_title}/{n}")
    print(f"  structured abstract        {with_abstract}/{n}")
    print(f"  parsed references          {with_refs}/{n}")
    print(f"  figure/table elements      {with_figs}/{n}")
    print(f"\n  cost: {mean_s:.1f} s/paper (PyMuPDF ~0.34 s) "
          f"+ a Java container in the stack")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({
        "n_extracted": n, "n_attempted": len(pdfs),
        "grobid_mean_canonical": round(g_canon, 2),
        "pymupdf_mean_canonical": round(p_canon, 2),
        "sections_with_body": with_body,
        "with_title": with_title, "with_abstract": with_abstract,
        "with_references": with_refs, "with_figures": with_figs,
        "mean_seconds": round(mean_s, 2),
        "records": records, "broken": broken_recs,
    }, indent=2))
    print(f"\nwrote {RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
