# Scholiast

Retrieval-and-reasoning platform for academic literature, built so retrieval quality
and groundedness are **measured rather than asserted**. Portfolio project for ML
Engineer interviews; the measurement discipline *is* the differentiator, not the
feature list.

**Status:** M0 complete. M1 (walking skeleton) in progress on `m1-walking-skeleton`,
built one phase per commit — **phases 1–4 done**: schema + migrations, tenant-scoped
repositories, GROBID parser, embedder + Qdrant index. **Phases 5–7 remain**: ingestion
pipeline, retrieval + generation, API + UI. The adapters exist behind ports but are
not yet wired to each other; `git log` carries the per-phase rationale and a recorded
list of debt deferred to M2/M3/M6.

---

## Read this first

**`docs/00-decisions-log.md` is the source of truth.** D1–D46, and several *reverse*
what the earlier design docs say — including two that reverse *other decisions*.
Reading doc 5, 6 or 7 without the log will lead you to recommend things already
measured against:

- GROBID: required → pending → **required again** (D19) — the 95% PyMuPDF figure
  measured heading *detection*; the chunker needs section *bodies*
- Provenance: "TEI lacks page offsets" → **false** (D39, reverses D19) — M0 simply
  never sent `teiCoordinates`. With it, 10,621/10,621 sentences carry coordinates
- Chunk offsets: document-relative → **section-relative** (D42) — they index
  `sections.text`, and `ParsedSection` enforces the round trip at construction
- Gemini: "strong first candidate" → **unsuitable** (D29) — its free tier trains on
  submitted content, and this system ingests possibly-unpublished manuscripts
- Local model: fallback → **throughput path** (D26) — free-tier quota is ~2 requests
  per minute
- Vector filtering: `workspace_id` → **`paper_id` set** (D36) — a public paper is one
  shared row, so a workspace filter would duplicate every vector per tenant

`docs/` is **gitignored (D12)** — read it from disk; it is not in git history.

| File | What it holds |
|---|---|
| `docs/00-decisions-log.md` | D1–D46 + measured hardware baseline. **Start here.** |
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
  a decisions-log entry — that is what M0 did (REV-1…7, D18–D35) and M1 kept doing
  (D36–D46, two of which reverse M0's own conclusions).

---

## ⚠️ Environment traps

Every one of these cost real time in M0 or M1. Do not rediscover them.

| Trap | Rule |
|---|---|
| **`uv pip install` targets the wrong Python** | Always pass `--python .venv\Scripts\python.exe`, or it silently installs into the system interpreter |
| **`datasets` breaks the ML stack** | A Windows Application Control policy blocks pyarrow's `_dataset` DLL, and `sentence-transformers` imports `datasets` at module load. **Do not install it** (D25). QASPER is vendored as plain JSON (D20). |
| **Smart App Control blocks `_rustgrimp`** | Windows SAC is *enforced* on this machine and blocks grimp 3.15's DLL by reputation, killing the `lint-imports` gate. Pinned `grimp<3.15`; 3.14 loads and keeps all three contracts. Same class as the `datasets` trap |
| **A model download is a network call** | Tests that load a real encoder must skip unless it is already in the local HF cache. `bge-base` is 419 MB, and NFR-T3 requires the default test run to make **zero** network calls. A warm cache on the dev machine hides this completely — CI is where it bites |
| **Alembic does not compare CHECK constraints** | Neither `--autogenerate` nor `alembic check` sees them. Write them by hand in the migration and cover each with an *attempted violation* — a constraint that silently failed to be created looks identical to one that works |
| **GROBID nests nothing** | `<figure>` is a **sibling** of `<div>` under `<body>`, never inside it, and the body has zero nested divs. Looking for tables inside a div finds exactly zero and raises nothing; section hierarchy comes from `head/@n` numbering |
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
