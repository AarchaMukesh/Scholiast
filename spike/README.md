# M0 Validation Spike — archived

Throwaway code, kept for the evidence rather than the implementation. Every number in
[`../SPIKE-FINDINGS.md`](../SPIKE-FINDINGS.md) traces to a script here and a JSON file
in `results/`.

**`spike/.venv` has been removed.** The root project venv now carries the ML stack, so
run these against it instead:

```bash
.\..\.venv\Scripts\python.exe scripts\01_gpu_check.py
```

Scripts 05 and 09 are the exception — they need `datasets`, which **must not be
installed** (D25: it breaks `sentence-transformers` via a blocked pyarrow DLL). Their
results are already captured in `results/`, and D20 mandates vendoring the QASPER
subset as plain JSON so the dependency is never needed again.

## What each script established

| Script | Deliverable | Outcome |
|---|---|---|
| `01_gpu_check.py` | GPU environment | RSK-7 closed — native `sm_120`, no PTX JIT |
| `02_fetch_fixtures.py` | Fixture set | 20 arXiv papers + 3 synthesised pathological PDFs |
| `03_parse_baseline.py` | Parser baseline | PyMuPDF 95% *heading* detection — the proxy that misled REV-5 |
| `04_model_bench.py` | Model benchmark | Refuted NFR-P7; exposed the fp32 rerank problem |
| `05_eval_dataset_check.py` | Eval datasets | `datasets` v4 dropped script loaders — QASPER blocked |
| `06_provider_check.py` | Providers | Groq works (255 ms TTFT); Gemini quota-blocked |
| `07_rerank_tuning.py` | Rerank tuning | **fp16 → 3.3× faster**, rescued NFR-P2 (D18) |
| `08_cpu_rerank.py` | CPU reranking | MiniLM @ top-20 = 493 ms; ONNX unnecessary (REV-4) |
| `09_qasper_recovery.py` | Eval datasets | QASPER recovered via parquet branch (D20) |
| `10_local_fallback.py` | Local fallback | First pass — VRAM reading wrong (see 11) |
| `11_vram_crossprocess.py` | Measurement | WDDM exposes no cross-process VRAM (D24) |
| `12_placement_order.py` | Local fallback | 3B keeps GPU in production ordering — **D21 resolved** |
| `13_gpu_contention.py` | §5.7 baseline | **9.17× rerank slowdown under bulk embedding** |
| `14_provider_diagnose.py` | Providers | Gemini 429≠401; Groq quota ≈ 2 req/min (D26, D27) |
| `15_grobid_compare.py` | GROBID | Section bodies 659/694 vs none — **D19 resolved** |

`fixtures/` (57 MB of arXiv PDFs) is gitignored but **kept on disk** — M1 uses the same
papers for integration tests, so the ingestion pipeline is exercised against documents
whose parse behaviour is already characterised.

`.env` holds live API keys and is gitignored. **Rotate them** — they were pasted into a
chat transcript.

## M4 discovery spike (D116, D117, D120, D125)

Pre-registered: `m4-protocol.json` is hash-locked, and every script refuses to run if its
canonical hash differs from the approved one. Findings in
[`../SPIKE-FINDINGS-M4.md`](../SPIKE-FINDINGS-M4.md), with what they do not establish
above the numbers.

| Script | Pass | Outcome |
|---|---|---|
| `19_discovery_analysis.py selftest` | P0, offline | Address table, URL rules, pacer, budget, the fetch path against a mock transport, and the log checker; 14/14 mutants caught (11 before the first live call, 3 added for A2's rule) |
| `17_discovery_search.py session1` | P1, P2 | OpenAlex and arXiv answered all 20 queries; S2 keyless 429ed 5/5 and was stopped; DEV-2 repeated 5 OpenAlex searches |
| `18_oa_probe.py probe` | P3 | 151 of 180 works fetched a PDF (83.9%): **ASM-2 PASS** |
| `18_oa_probe.py download` | P4 | 30/30 passed `validate_upload` and `preflight` |
| `17_discovery_search.py session2 --wait` | P5 | 6 h 49 min later: top-10 overlap 0.97 (OpenAlex), 0.99 (arXiv); S2 keyless 429ed 5/5 again |
| `19_discovery_analysis.py check / analyse / finalize / leakcheck` | P6 | **ASM-1 FAIL**; tracked outputs written S2-free |
| `17_discovery_search.py addendum-a1`, then `19_discovery_analysis.py finalize-a1` | A1 (D120) | arXiv joined with OR vs a same-day AND: 4 vs 6 of 20 anchors in the top 50, so **AND stays**; its own pools file, the spike's left unchanged |
| `17_discovery_search.py addendum-a2`, then `19_discovery_analysis.py finalize-a2` | A2 (D125) | OpenAlex at 50, 25 and 15 results per page: all inside the 2.0 s deadline, none at 50% of anchors because OpenAlex's ranking changed (D126), so the owner chose **25**; its own pools file |

`m4_common.py`, `m4_http.py`, `m4_parse.py` and `m4_works.py` are shared by the three.
They are throwaway too: phase 2 writes the real adapters, and phase 5 the D115 guard.

**Every live command is resumable and never repeats a request.** A `sending` line is
logged before each request leaves, and a rerun skips any request id already in the log.
Run logs, S2 responses and downloaded PDFs stay under `var/spike/` (gitignored). The
tracked outputs (`results/17_*`, `18_*`, `19_*` and `data/eval/discovery-pools.*`) hold
no S2 content and no key, which `leakcheck` verifies. The live passes need
`OPENALEX_API_KEY` in `.env` (see `.env.example`).
