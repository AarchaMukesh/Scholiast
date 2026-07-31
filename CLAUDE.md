# Scholiast

Retrieval-and-reasoning platform for academic literature, built so retrieval quality
and groundedness are **measured rather than asserted**. Portfolio project for ML
Engineer interviews; the measurement discipline *is* the differentiator, not the
feature list.

**Status:** M0 (validation spike) complete. M1 (walking skeleton) in progress —
foundation done, adapters and the end-to-end slice remain.

---

## Read this first

**`docs/00-decisions-log.md` is the source of truth.** D1–D35, and several *reverse*
what the earlier design docs say. Reading doc 7 without the log will lead you to
recommend things already measured against:

- GROBID: required → pending → **required again** (D19) — the 95% PyMuPDF figure
  measured heading *detection*; the chunker needs section *bodies*
- Gemini: "strong first candidate" → **unsuitable** (D29) — its free tier trains on
  submitted content, and this system ingests possibly-unpublished manuscripts
- Local model: fallback → **throughput path** (D26) — free-tier quota is ~2 requests
  per minute

`docs/` is **gitignored (D12)** — read it from disk; it is not in git history.

| File | What it holds |
|---|---|
| `docs/00-decisions-log.md` | D1–D35 + measured hardware baseline. **Start here.** |
| `docs/03-requirements.md` | FR/NFR IDs referenced throughout the code |
| `docs/05-system-architecture.md` | §5.3 layering · §5.6 schema · §5.7 GPU contention |
| `docs/06-ml-system-design.md` | chunking, retrieval, citation integrity, ablations |
| `docs/08-roadmap.md` | milestone scope + acceptance criteria |
| `docs/09-evaluation-strategy.md` | metrics, datasets, statistical rigour |
| `SPIKE-FINDINGS.md` | M0 measurements — every number is real, none projected |

---

## Commands

Five gates. All must be green before any commit; CI runs the same chain.

```bash
uv run ruff format --check src tests
uv run ruff check src tests
uv run mypy src
uv run lint-imports
uv run pytest
```

Direct interpreter (Windows, when `uv run` is awkward):
`.\.venv\Scripts\python.exe -m pytest`

Services: `docker compose up -d` (Postgres · Qdrant · Redis · GROBID).

---

## Architecture rules

**Layer contracts are machine-checked** (`lint-imports`, 3 contracts in
`pyproject.toml`). Dependencies point inward; `application` never imports adapters;
`domain` imports **no framework at all** — stdlib only.

**Never weaken a contract to make code compile.** The contract is the design. If a
violation appears, the import is wrong, not the rule.

```
src/scholiast/
├── domain/          entities, invariants, chunking, fusion — stdlib only, no I/O
├── application/     use cases + ports (interfaces)
├── infrastructure/  adapters — the ONLY layer that knows vendor names
├── api/             HTTP surface
├── workers/         async pipeline tasks
└── eval/            evaluation harness (a real package, not a script folder)
```

Repositories require a `TenantContext` by constructor — a repository that cannot
filter by tenant will not type-check. `SystemContext` is the single sanctioned
bypass; it demands a stated reason so it stays greppable.

---

## Conventions

- **Tests name the failure they prevent**, not the function they call.
  `test_fusion_is_immune_to_score_scale`, not `test_rrf`.
- **Version-stamp derived artifacts** — `chunker_version`, `embedder_version`,
  `prompt_version` (NFR-M5). This is what makes "was it the reranker or the re-chunk?"
  answerable.
- **Reference requirement IDs in comments** where behaviour is non-obvious (FR-I5,
  NFR-P2, D18…). The docs explain *why*; the code should point at them.
- **If a measurement contradicts a doc, the measurement wins.** Update the doc and add
  a decisions-log entry — that is what M0 did (REV-1…7, D18–D35).

---

## ⚠️ Environment traps

Every one of these cost real time in M0. Do not rediscover them.

| Trap | Rule |
|---|---|
| **`uv pip install` targets the wrong Python** | Always pass `--python .venv\Scripts\python.exe`, or it silently installs into the system interpreter |
| **`datasets` breaks the ML stack** | A Windows Application Control policy blocks pyarrow's `_dataset` DLL, and `sentence-transformers` imports `datasets` at module load. **Do not install it** (D25). QASPER is vendored as plain JSON (D20). |
| **torch must be a CUDA 12.8+ build** | RTX 5060 is Blackwell (`sm_120`). Verify with `'sm_120' in torch.cuda.get_arch_list()` — `torch.cuda.is_available()` is not sufficient |
| **fp16 for all local inference** | fp32 reranking is 3.3× slower (743 ms vs 224 ms @ top-50) and blows NFR-P2 (D18) |
| **GROBID image tag** | `grobid/grobid:0.8.2.1-crf` (0.41 GB) on :8070 — **not** the default `0.8.1` (32.5 GB) (D31) |
| **Ollama placement is sticky** | Within `keep_alive`, one `num_gpu: 0` request pins the model to CPU for later requests. Read actual placement, never assume (D23) |
| **No cross-process VRAM accounting** | Windows WDDM: `nvidia-smi` returns `0 MiB`/`[N/A]`, torch sees only its own context. Demonstrate co-residency empirically (D24) |

**Dev topology:** backing services in Docker; **API and worker run natively on Windows**
for direct GPU access. CI runs the fully containerized Linux path so the two cannot
silently diverge.

---

## Measured baseline (M0)

Real numbers. Quote these, not estimates.

| | |
|---|---|
| GPU | RTX 5060 Laptop, `sm_120` native, 29.0 TFLOP/s fp16 |
| **Usable VRAM** | **~6.8 GB** of 7.96 GB — the desktop compositor holds the rest |
| Embedder + reranker co-resident | 2.36 GB |
| Rerank 50 @ fp16 / fp32 | **224 ms** / 743 ms |
| Rerank 20 on CPU (MiniLM) | 493 ms |
| Embedding | 196 chunks/s (384-dim) · 75.8 (768-dim) · 6.9 CPU |
| Local generation (3B, GPU) | 125 tok/s, 100% GPU co-resident |
| Groq free tier | 12,000 tokens/min ≈ **~2 RAG requests/min** |
| GROBID CRF | 3.1 s/paper, section bodies for 95% of sections |
| Rerank under bulk embedding | **9.17× slowdown** — why the priority semaphore exists |

---

## Working style

Challenge assumptions rather than agreeing by default. Explain tradeoffs before
recommending. Prefer boring, simple solutions; complexity must answer "what breaks
without it". When a decision is genuinely the user's, ask; otherwise decide and say
why. Report failures with actual output — a negative result with a good explanation is
worth more than an unmeasured claim.
