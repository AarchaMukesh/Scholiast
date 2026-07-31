# M0 — Validation Spike Findings

**Run date:** 2026-07-27 · **Machine:** HP OMEN, RTX 5060 Laptop (8 GB), 24 GB RAM, Windows 11, Python 3.12.9

Purpose (doc 8 §M0): kill or confirm every load-bearing assumption **before** any
architectural decision is committed to code. Scripts under `spike/scripts/`, raw
results under `spike/results/`. All of it is throwaway by design.

---

## Status

| # | Deliverable | Status | Verdict |
|---|---|---|---|
| 1 | GPU environment | ✅ done | **PASS** — native `sm_120`, no JIT fallback |
| 2 | Paper fixture set | ✅ done | 20 papers + 3 synthetic pathological cases |
| 3 | Parser evaluation | ⚠️ partial | PyMuPDF baseline **95% PASS**; GROBID **blocked** (no Docker/WSL2) |
| 4 | Model benchmark | ✅ done | **2 assumptions refuted**, both recoverable |
| 5 | Free-tier providers | ⛔ blocked | Harness written; **still needs API keys in `spike/.env`** |
| 6 | Local LLM fallback | ✅ done | **PASS** — D21 resolved; 3B fits on GPU co-resident |
| 7 | Eval dataset feasibility | ✅ done | **PASS** — QASPER recovered; RSK-2 contained |
| 8 | Hosting sanity check | ⛔ not started | — |
| 9 | This document | ✅ done | — |

---

## 1. GPU environment — **PASS** (RSK-7 closed)

The roadmap's highest-risk item. Blackwell (`sm_120`) needs CUDA 12.8+ wheels;
older builds fail or silently fall back to slow PTX JIT.

| | |
|---|---|
| torch | `2.11.0+cu128`, built against CUDA 12.8 |
| Driver | 592.19 (supports CUDA 13.1) |
| Device | RTX 5060 Laptop, compute capability 12.0 (`sm_120`) |
| Native arch list | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120` ✅ |
| Device matmul | correct (max abs err 4.9e-04 vs CPU) |
| fp16 throughput | 29.0 TFLOP/s |

> **⚠️ REV-3 — VRAM budget was wrong.** Total is **7.96 GB**, but the Windows desktop
> compositor holds **1.1–1.3 GB** before we allocate anything. **Usable ≈ 6.7–6.9 GB**,
> not 8 GB. Docs 5 §5.7 and 6 §6.16 budget against 8 GB and must be corrected.
>
> Consequence: normal operation is unaffected (2.36 GB measured, §4 below), but the
> degraded path with a co-resident 7B Q4 (~4.7 GB + KV cache) **does not fit** alongside
> the retrieval models. D15's load-on-demand decision is now load-bearing rather than
> merely tidy — and even then the margin is thin. Deliverable 6 must settle it.

**Install note:** `uv pip install` targeted the system Python rather than `.venv` when
invoked as `python -m uv` without `VIRTUAL_ENV` set. Use `--python .venv\Scripts\python.exe`
explicitly. (Torch also landed in system site-packages; harmless, worth cleaning up.)

---

## 2. Fixture set — done

20 arXiv papers spanning *clean* two-column (9), *table-heavy* (7), and *long* 75–100 pp
(4), plus 3 synthesised pathological cases: image-only ("scanned"), AES-256 encrypted,
and structurally-valid-but-empty. Synthesising the broken cases rather than hunting for
them makes the failure-taxonomy tests deterministic and reproducible.

---

## 3. Parsing — PyMuPDF **95% PASS**, GROBID blocked

| Metric | Result |
|---|---|
| Usable section structure | **19/20 (95%)** vs. ASM-3's 85% bar → **PASS** |
| Failure | `rag-survey` — survey formatting, 1 canonical section detected |
| FR-I9 pathological detection | **3/3** (encrypted → `UNPARSEABLE`; image-only + empty → `SCANNED_IMAGE`) |
| Mean parse time | 339 ms/paper (worst 2.4 s on 75 pp) — far inside NFR-P6's 90 s |

**This is PyMuPDF + heading heuristics alone, with no GROBID.**

> **⚠️ REV-5 — GROBID's justification is weaker than doc 7 §7.4 assumed.** A 95% hit rate
> from the fallback path substantially undercuts the case for adding a Java container,
> which was also the awkward part of DEP-5's production story.
>
> **Honest caveat:** the metric is *heading detection*, a proxy. GROBID additionally
> provides section **boundaries**, hierarchy, reference parsing, and figure/table
> captions — none measured here. **Do not drop GROBID on this evidence alone**; run the
> comparison once Docker is available, then decide with both numbers.

---

## 4. Model benchmark — 2 assumptions refuted

All measurements on 512 real chunks (~1200 chars) extracted from the fixture papers.

### 4.1 Embedding

| Model | Device | VRAM | Best throughput | Query latency |
|---|---|---|---|---|
| bge-small-en-v1.5 (384d) | cuda | 0.14 GB | **196 chunks/s** | 7.2 ms |
| bge-base-en-v1.5 (768d) | cuda | 0.44 GB | **75.8 chunks/s** | 8.6 ms |
| bge-base-en-v1.5 | cpu | — | 6.9 chunks/s | — |

> **⚠️ REV-1 — NFR-P7 (≥200 chunks/s) is refuted for bge-base and was a made-up number.**
> The real constraint is NFR-P6 (≤90 s/paper): a 20-page paper ≈ 150 chunks ⇒ **~2 s at
> 75.8 chunks/s**, passing with a ~45× margin. Even CPU-only (6.9 chunks/s ⇒ ~22 s/paper)
> passes comfortably.
>
> **Action:** delete NFR-P7's asserted target and derive it from NFR-P6 instead. An
> arbitrary threshold that "fails" while the actual requirement passes by 45× is a
> specification bug, not a performance problem.

> **⚠️ REV-7 — bge-small vs bge-base is now a genuine decision, not a default.** Small is
> **2.6× faster**, uses **⅓ the VRAM**, and **halves vector-store memory** (384 vs 768
> dims). Whether it costs retrieval quality is unknown until D-RET exists. **Add to the
> M2 ablation** — this is exactly the "benchmark on your own eval set" principle (ML-4).

### 4.2 Reranking — fp16 rescues NFR-P2

| Model | Precision | top-10 | top-20 | top-30 | top-50 |
|---|---|---|---|---|---|
| bge-reranker-base | fp32 | 126 ms | 297 ms | 437 ms | **743 ms** |
| bge-reranker-base | **fp16** | 40 ms | 91 ms | 136 ms | **224 ms** |
| ms-marco-MiniLM-L6 | fp32 | 29 ms | 64 ms | 96 ms | 153 ms |
| ms-marco-MiniLM-L6 | fp16 | 14 ms | 26 ms | 40 ms | 59 ms |

> **⚠️ REV-2 — the design never specified inference precision, and that omission nearly
> cost a design compromise.** At fp32, reranking top-50 consumed 743 ms of an 800 ms
> budget meant to cover the *entire* retrieval stage — effectively a failure, and worse
> under the concurrent-ingestion scenario §5.7 exists to handle. **fp16 gives a 3.3×
> speedup to 224 ms**, leaving ~450 ms of headroom.
>
> **Action:** specify fp16 for all local inference in docs 6 §6.16 and 7 §7.4. NFR-P2
> then passes properly rather than by 7%.

### 4.3 CPU reranking (DEP-2) — resolved without ONNX

| Config | top-10 | top-20 | top-30 |
|---|---|---|---|
| bge-reranker-base fp32 | 1292 ms | 2679 ms | 4250 ms |
| ms-marco-MiniLM-L6 fp32 | 234 ms | **493 ms** | **827 ms** |

> **⚠️ REV-4 — doc 10 §10.3's plan was wrong in method but right in conclusion.** It
> assumed ONNX int8 would close the gap. In fact `torch.quantization.quantize_dynamic`
> **errors** under sentence-transformers 5.x, and it isn't needed: a smaller cross-encoder
> already clears the 2000 ms production budget at fp32.
>
> **Production now has a measurable choice rather than a compromise:**
> - **(a)** same model, shallower — bge-reranker fp32 @ top-10, 1292 ms → consistent
>   per-pair quality, lower recall ceiling
> - **(b)** faster model, deeper — MiniLM fp32 @ top-30, 827 ms → higher recall ceiling,
>   lower per-pair quality
>
> **M2 decides it on end-to-end nDCG.** Neither is obviously right.

### 4.4 Co-residency (ASM-4) — **PASS**

Both models resident and active: **2.36 GB** of ~6.87 GB usable. Comfortable, and the
§5.7 priority-semaphore design stands.

*Measurement caveat:* per-model VRAM deltas across sequential loads are unreliable (the
CUDA allocator doesn't return memory to the OS; fp16 reported 0.0 GB). The co-residency
figure is the trustworthy one.

**Remaining headroom for local generation: ~4.5 GB** — below a 7B Q4's ~4.7 GB + KV cache.
Confirms the concern raised in §1.

---

## 7. Evaluation datasets — **PASS** (RSK-2 contained)

| Source | Status | Notes |
|---|---|---|
| `allenai/qasper` (default loader) | ❌ | `datasets` v4 removed script-based loaders |
| **`allenai/qasper` @ `refs/convert/parquet`** | ✅ | **281 validation papers**; columns `id, title, abstract, full_text, qas, figures_and_tables` |
| `BeIR/scifact` corpus + queries + qrels | ✅ | 5,183 docs / 1,109 queries / 339 qrels — **binary** relevance, abstract-level |
| `BeIR/scidocs` corpus | ✅ | 25,657 docs |
| `allenai/scifact` (default loader) | ❌ | same loader issue |

> **⚠️ REV-6 — ASM-5's path changed but the conclusion holds.** QASPER is reachable via
> the auto-converted parquet branch, giving **paragraph-level evidence inside full
> papers** — the granularity D-RET needs and the thing BEIR cannot provide (its unit is
> the abstract).
>
> **Action:** pin `revision="refs/convert/parquet"` and record the commit SHA — an
> auto-generated branch is less stable than a released dataset. Snapshot the extracted
> subset into the repo so eval reproducibility doesn't depend on it staying available.

**D-RET plan:** QASPER for chunk-level judgments (primary); BEIR scifact as an
independent document-level IR baseline that validates the retrieval stack end-to-end
against published numbers. Note BEIR's relevance is **binary**, so nDCG's graded
advantage only applies to the QASPER-derived set.

---

## 6. Local generation fallback — **PASS**, D21 resolved

Model: `llama3.2` (3B, Q4, 2.6 GB) via Ollama. Retrieval models resident in fp16.

### Placement — the question that mattered

D21 predicted a 7B would not fit and named a 3B or CPU-only as the fallbacks. The
3B does better than "fits": it keeps the **GPU** even in the production ordering,
where the retrieval models load *first* and generation loads after them.

| Ordering | Placement | Throughput |
|---|---|---|
| Generation on an empty GPU | 100% GPU | 123.1 tok/s |
| Generation, retrieval loaded after | 100% GPU | 120.1 tok/s |
| **Retrieval first, then generation** *(production case)* | **100% GPU** | **125.0 tok/s** |
| Forced CPU-only | 100% CPU | 17.6 tok/s |

**Grammar-constrained JSON works**, and produced correct output on the first try:
`{"ndcg": 0.74, "chunking": "Structure-aware", "citations": ["C2"]}` — right value,
right citation. FR-C4's schema is structurally guaranteed on the local path, which
is genuinely stronger than the API path's validate-and-repair.

> **⚠️ Operational gotcha — placement is sticky within `keep_alive`.** An
> intermediate run reported `100% CPU` at ~21 tok/s and briefly looked like a
> refutation. Cause: the previous script ended with a deliberate `num_gpu: 0`
> request, and the next run **reused the already-CPU-placed model** inside the
> 5-minute `keep_alive` window. One forced-CPU request poisons every subsequent
> one until the model unloads.
>
> **Action for the adapter:** never mix `num_gpu` settings against the same
> loaded model, and **report actual placement** (`ollama ps`) rather than assuming
> it — otherwise a silent CPU fallback masquerades as a working GPU path, which is
> precisely the undisclosed degradation NFR-R4 forbids.

### §5.7 contention baseline — the number that justifies the priority semaphore

Measured with generation confirmed GPU-resident (an earlier attempt measured it
while generation had silently moved to CPU, which understated the problem):

| Condition | rerank top-20 | vs. idle |
|---|---|---|
| GPU idle | 47 ms | — |
| During generation (103.9 tok/s) | 96 ms | 2.05× |
| **During bulk embedding** | **429 ms** | **9.17×** |

**Ingestion, not generation, is the threat to query latency** — exactly what
§5.7 argued, now with a number. At the measured top-20 depth 429 ms still fits the
800 ms budget, but the design specifies **top-50**, which costs 224 ms idle
(fp16). *Projecting* the same 9.17× gives **~2,050 ms against an 800 ms budget** —
roughly 2.5× over.

So the priority semaphore and bounded ingestion batches are load-bearing rather
than defensive. **This is the baseline M3's implementation is measured against.**
(The top-50 figure is a projection from the measured top-20 slowdown, not a direct
measurement — worth confirming once the semaphore exists.)

---

## Environment findings (not deliverables, but they bite)

| # | Finding | Consequence |
|---|---|---|
| E1 | **Windows WDDM exposes no reliable cross-process VRAM accounting.** `nvidia-smi` reports `0 MiB` used and `[N/A]` per process; `torch.cuda.mem_get_info()` only sees the calling context. | Co-residency claims must be demonstrated **empirically** (run both, confirm no OOM and no CPU spill) rather than by summing per-model figures. Any VRAM total quoted in the docs is an estimate — say so. |
| E2 | **A Windows Application Control policy blocks pyarrow's `_dataset` DLL.** Reinstalling does not help. | `datasets` is unusable on this machine — and because `sentence-transformers` 5.x imports it at module load, **installing `datasets` breaks the core ML path**. Mitigation is already mandated by D20: extract QASPER once and **vendor the snapshot as plain JSON**, then drop the `datasets` dependency entirely. |

---

## Blocked — requires action

| # | Blocker | Needed | Impact if unresolved |
|---|---|---|---|
| 3 | GROBID comparison | Docker Desktop + WSL2 | REV-5 stays undecided; PyMuPDF fallback is viable meanwhile |
| 5 | Free-tier providers | **API keys** for ≥2 of: Gemini, Groq, Cerebras, OpenRouter, Mistral | **D14's whole generation path is unvalidated.** Highest-priority blocker |
| 6 | Local LLM fallback | Ollama install | D15 unvalidated; §1 suggests a 7B may not fit — a 3B or CPU-only may be required |
| 8 | Hosting (ASM-9) | Account signups / research | Discovering this at M6 is the most avoidable schedule failure in the roadmap |

---

## Design revisions — **all applied 2026-07-27**

| ID | Change | Docs updated | Status |
|---|---|---|---|
| REV-1 | Delete NFR-P7's asserted 200 chunks/s; derive from NFR-P6 | 3 (NFR-P6/P7) | ✅ applied |
| REV-2 | **Specify fp16 for all local inference** | 5 §5.7, 6 §6.16 + ML-8, 7 §7.4 | ✅ applied (→ **D18**) |
| REV-3 | VRAM budget → **~6.8 GB usable**, not 8 GB | 5 §5.7, 6 §6.16, 0 baseline | ✅ applied (→ **D21**) |
| REV-4 | Production reranker = two measurable options; drop the ONNX-int8 assumption | 10 §10.3 + DEP-2, 9 §9.3 | ✅ applied |
| REV-5 | Downgrade GROBID from required to *pending comparison* | 7 §7.4, 10 DEP-5 | ✅ applied (→ **D19**) |
| REV-6 | QASPER via pinned parquet revision + vendored snapshot | 9 §9.2 | ✅ applied (→ **D20**) |
| REV-7 | Add bge-small vs bge-base to the M2 ablation axes | 6 §6.17, 9 §9.3 | ✅ applied |

Four findings were consequential enough to become binding decisions in their own right:
**D18** (fp16), **D19** (GROBID pending), **D20** (D-RET source), **D21** (RSK-16 escalated).

---

## Assumption ledger

| ID | Assumption | Verdict |
|---|---|---|
| ASM-3 | Parser handles ML papers ≥85% | ✅ **95%** (PyMuPDF alone) |
| ASM-4 | Embedder + reranker fit and run fast enough in 8 GB | ✅ **2.36 GB**, both budgets met at fp16 |
| ASM-5 | Public datasets can bootstrap the eval set | ✅ QASPER via parquet branch |
| ASM-9 | Free hosting can run the stack | ⛔ untested |
| RSK-7 | Blackwell/CUDA toolchain works | ✅ closed |
| RSK-2 | Eval-set construction is the hidden time sink | ✅ contained |
| RSK-16 | Local fallback too large/slow for 8 GB | ⚠️ **likely a real problem** — ~4.5 GB headroom vs ~4.7 GB+ needed |
