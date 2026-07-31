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
