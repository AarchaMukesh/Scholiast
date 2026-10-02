# Scholiast

Retrieval-and-reasoning platform for academic literature, built so retrieval quality
and groundedness are **measured rather than asserted**. Portfolio project for ML
Engineer interviews; the measurement discipline *is* the differentiator, not the
feature list.

**Status:** M0, M1, M2 and **M3 complete**, built one phase per commit. M1 was the spine —
schema, repositories, parser, index, pipeline, QA, API. M2 is the thesis: D-RET, the metric
pipeline, four retrieval arms, the ablation axes, the GPU contention number and a CI
regression gate. M3 is the operational surface: auth, three-layer tenant isolation, Celery,
the §4.4.1 failure taxonomy, cascade deletion across three stores, a provider circuit
breaker, observability, and rate limiting. **1218 tests, five gates green.** M4 is next:
the discovery engine.

**Read `docs/milestone-logs/M3-MILESTONE-LOG.md` before starting M4**, then M2's for the
retrieval numbers. Milestone logs are the only tracked files under `docs/` (D67), and each
one's "what this does **not** establish" section sits above its findings on purpose.

**M4 has two recorded preconditions** (M3 log §9): build the SSRF allowlist D115 moved
there — discovery is the first thing that dereferences a URL it did not choose — and
produce work on the `backfill` queue, which exists and is empty.

**One of M3's eight acceptance criteria was amended rather than met**, by two decisions,
both because a design document described behaviour the code could not perform: §5.11's `reranker_unavailable`
and `dense_retrieval_unavailable` rungs are vacuous (D105), and `local_generation` now
describes the happy path (D106). FR-O4 tracing is deferred (D109). Do not re-argue these
either — they are in the M3 log §3.

**Retrieval quality is now measured, and three of its four headline results are
negative.** Do not re-argue any of these — they cost a phase each:

- **Hybrid beats dense-only** on D-RET-CORE: recall@10 **+0.088** [+0.004, +0.177],
  nDCG@10 **+0.061**, significant on four of five metrics. Adopted? **Not yet** — FR-Q3's
  decision is deliberately open and production is still dense-only.
- **Reranking is a null against hybrid**, and the cause is measured: at the adopted top-50
  budget **two-thirds of relevant chunks never reach the cross-encoder**, and the
  pre-rerank ceiling still climbs at depth 100 (CORE 0.342 → 0.417). It is a starved
  reranker, not a weak one.
- **bge-small ties bge-base** on quality (all nine deltas within noise) while indexing
  2.1× faster (D71). **Weighted fusion ties RRF** with the weight fitted on the train
  split (D76). Neither changed production.

---

## Read this first

**`docs/00-decisions-log.md` is the source of truth.** D1–D115, and several *reverse*
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
- Chat transport: `EventSource` → **POST + `fetch`** (D61, narrows D37) — `EventSource`
  can only GET, which would put the question in a URL; the SSE wire format is kept
- FR-Q2: one requirement → **two** (D66) — citation *validity* is M1 and guaranteed;
  citation *coverage* is FR-Q11, measured in M5. A real 3B cited nothing on a broad
  question, and the requirement as written could not tell the difference
- Rerank under ingestion: "9.17× slowdown" → **fp32 artifact** (D74) — M0 measured it at
  top-20 with an fp32 embedder. At production fp16 and top-50 it is **1.9× ungated, 1.5×
  gated**, and the gate costs 17% of ingestion throughput
- Reranking: doc 6's "single most reliable quality win" → **null on this eval set**
  (phases 4–5). The expectation is doc 6's; the measurement is the project's
- D68's PDF check: "report the Recall@k delta" → **the delta is not obtainable that way**
  (D81) — QASPER's text is LaTeX-derived, so **40.6% of evidence strings carry a `BIBREF`
  placeholder** the typeset PDF never contains and judgments do not transfer by string
  match. Only 15 of 112 survived

`docs/` is **gitignored (D12)** — read it from disk. The one exception is
`docs/milestone-logs/`, which **is** committed (D67): a milestone log reports
measurements against a specific commit, so it belongs in history with it.

| File | What it holds |
|---|---|
| `docs/00-decisions-log.md` | D1–D115 + measured hardware baseline. **Start here.** |
| `docs/milestone-logs/M3-MILESTONE-LOG.md` | M3's close-out: 7/8 criteria as written and 1 amended with cause, 15 defects and what found them, 96 mutations, debt to M4/M6 |
| `docs/milestone-logs/M2-MILESTONE-LOG.md` | M2's close-out: 8/8 criteria, every measurement with its interval, 16 defects and what found them, debt to M3/M5/M6 |
| `docs/milestone-logs/M1-MILESTONE-LOG.md` | M1's close-out: 7/7 criteria, what is *not* measured, debt to M2/M3/M5 |
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

**The eval (M2).** Every mode archives a date-stamped JSON under `var/eval/` with a full
config snapshot; the plain command measures what production runs.

```bash
uv run eval                      # dense-only, the D56 ablation baseline
uv run eval --compare-arms       # dense / lexical / RRF hybrid / hybrid+rerank
uv run eval --compare-depths     # candidate depth 10/20/30/50 + a 100 diagnostic (D77)
uv run eval --compare-fusion     # RRF vs weighted, weight fitted on the TRAIN split (D76)
uv run eval --contention         # rerank latency with the GPU gate on and off (D79)
uv run eval --pdf-check          # D68's real-PDF validity check — fetches from arXiv
uv run eval --smoke              # FR-E6's CI gate; exits 1 on a regression (D82)
```

Reruns are cheap: a complete collection is never rebuilt, and the PDF cache is reused.
**Never point a gated embedder at the eval collection** — it re-batches and perturbs fp16
vectors (D74), which is why `--contention` refuses to build one.

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
├── eval/            evaluation harness (a real package, not a script folder)
└── main.py          composition root — in NO layer, wires api + workers (D63)
```

`main.py` exists because `import-linter` rejected an API lifespan that constructed
the worker: `api` and `workers` are independent siblings, and an HTTP layer that
imports a background worker cannot be deployed without one. `uvicorn scholiast.main:app`
is the one command M1's acceptance criterion asks for — **not** `scholiast.api.app`,
which exposes only a factory.

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
| **`datasets` breaks the ML stack** | A Windows Application Control policy blocks pyarrow's `_dataset` DLL, and `sentence-transformers` imports `datasets` at module load. **Do not install it** (D25). QASPER is therefore vendored as plain JSON (D20) — done in M2 phase 1: `data/eval/qasper-validation.json.gz`, 281 papers, with a manifest. Its text is LaTeX-derived, which matters more than it sounds; see the `BIBREF` row below. |
| **⚠️ Smart App Control can block compiled extensions without warning — it has taken gates 3 and 4 down before (D72)** | Windows SAC is *enforced* (`VerifiedAndReputablePolicyState = 1`) and judges unsigned native DLLs by **reputation, which can change while the files stay byte-identical**. Pinning `grimp<3.15` dodged an earlier block on 3.15 but is not a fix: on 2026-09-16 SAC blocked 3.14's `_rustgrimp` too (killing `lint-imports`, **gate 4**) and mypy's `__mypyc` .pyd (**gate 3**), and D72 records one-off `torch` and `grpc._cython.cygrpc` load failures. **The block then lifted on its own** — no block logged after 11:13 that day, and all five gates ran green that evening — **but it can recur at any time.** If gate 3 or 4 dies loading a DLL, check Code Integrity event **3077** (`Microsoft-Windows-CodeIntegrity/Operational`) before debugging anything else. **Reinstalling cannot help**: the installed `_rustgrimp` matches its wheel's RECORD hash, so a reinstall rewrites bytes SAC has already judged. **A module-level route sometimes works, so try it before giving up**: during the M2 phase 6 recurrence the `lint-imports` LAUNCHER was blocked while `grimp._rustgrimp` imported fine, and `importlinter.cli:lint_imports_command` ran the same three contracts in-process. D72's earlier claim that every route is blocked by one DLL is false in general. **"Five gates green" is a claim about a run, not the machine**: make it only when all five ran and passed in that session, and never rewrite the layering acceptance tests to route around a block. Options are recorded in D72; none is taken. CI's containerized Linux path is unaffected |
| **A model download is a network call** | Tests that load a real encoder must skip unless it is already in the local HF cache. `bge-base` is 419 MB, and NFR-T3 requires the default test run to make **zero** network calls. A warm cache on the dev machine hides this completely — CI is where it bites |
| **Alembic does not compare CHECK constraints** | Neither `--autogenerate` nor `alembic check` sees them. Write them by hand in the migration and cover each with an *attempted violation* — a constraint that silently failed to be created looks identical to one that works |
| **GROBID nests nothing** | `<figure>` is a **sibling** of `<div>` under `<body>`, never inside it, and the body has zero nested divs. Looking for tables inside a div finds exactly zero and raises nothing; section hierarchy comes from `head/@n` numbering |
| **torch must be a CUDA 12.8+ build** | RTX 5060 is Blackwell (`sm_120`). Verify with `'sm_120' in torch.cuda.get_arch_list()` — `torch.cuda.is_available()` is not sufficient |
| **fp16 for all local inference** | fp32 reranking is 3.3× slower (743 ms vs 224 ms @ top-50) and blows NFR-P2 (D18) |
| **GROBID image tag** | `grobid/grobid:0.8.2.1-crf` (0.41 GB) on :8070 — **not** the default `0.8.1` (32.5 GB) (D31) |
| **Ollama placement is sticky** | Within `keep_alive`, one `num_gpu: 0` request pins the model to CPU for later requests. Read actual placement, never assume (D23) |
| **No cross-process VRAM accounting** | Windows WDDM: `nvidia-smi` returns `0 MiB`/`[N/A]`, torch sees only its own context. Demonstrate co-residency empirically (D24) |
| **Coverage under-reports async DB code** | `[tool.coverage.run]` has no `concurrency`, and SQLAlchemy's asyncio layer runs inside greenlets — coverage.py's default `thread` tracer does not follow a greenlet switch, so lines executing under an `await` on the database read as dead. Measured: one passing test covers `remove_paper.py` at 72% (`thread`) vs 86% (`greenlet,thread`). **Do not read a low number on a repository or use case as a missing test** until re-run with `concurrency = greenlet, thread`. Unfixed on purpose — changing it changes what CI computes (M1 log §7) |
| **`.gitignore` cannot re-include under an excluded directory** | `docs/` + `!docs/milestone-logs/` silently does nothing: git never descends into an excluded *directory*, so the negation is unreachable. It must be `docs/*` + `!docs/milestone-logs/` (D67). Verify with `git check-ignore -v <path>`, never by eye |
| **⚠️ An exit code read through a pipe is the pipe's** | `cmd \| grep ...; echo $?` reports GREP's status, which is 0 whatever `cmd` did. This "verified" the CI smoke gate's failure path once — a gate that would have passed every build forever. Check an exit code on the command itself, or use `PIPESTATUS`. The same trap hides behind `\| tail` in a background wrapper |
| **⚠️ `python -m importlinter.cli` is a NO-OP that exits 0** | It has no `if __name__ == "__main__"` guard, so `-m` imports the module, defines the click commands and exits — **accepting any arguments and checking nothing**. `python -m importlinter.cli lint-imports`, `--help` and `total nonsense --bogus` all print zero output and return 0, which in a CI log is indistinguishable from a passing gate. It reported "imports=0" for a gate that never ran, during M3 phase 4. **Gate 4 is `lint-imports` (or `.\.venv\Scripts\lint-imports.exe`), and a real pass always prints `Contracts: 3 kept, 0 broken`** — treat a silent gate 4 as a failed one. This is NOT the SAC workaround above: that route is the `lint_imports_command` *function* called in-process, not `-m` on the module |
| **⚠️ Latency drifts with machine state; compare WITHIN a run** | Idle rerank p50 measured **156 ms and 228 ms on the same tree an hour apart** — the whole machine was ~1.4× slower, which turned an in-budget stage into an over-budget one. One contention run was discarded for this. Every latency comparison must measure its own baseline in the same process; cross-run absolute latencies are not comparable |
| **Warm models before any clock that divides by wall-clock** | Whichever condition touches a model first pays its load. Per-query percentiles survive it (they have warm-ups); throughput does not — it once deflated a gated arm by a fifth, nearly publishing "the gate costs 32% of ingestion" when it costs 17% |
| **The corpus text is LaTeX-derived, and carries `BIBREF` markers** | QASPER's `full_text` comes from arXiv LaTeX, not the typeset PDF: **47.8% of indexed chunks contain `BIBREF<n>`**, which the lexical tokenizer indexes as real terms (`bibref0`…). It qualifies every lexical result, and it is why judgments cannot be transferred to a GROBID build by string matching (T12, D81) |
| **The Grep tool cannot see `docs/`** | `docs/` is gitignored (D12) and ripgrep skips ignored files, so a Grep over `docs/` returns "No matches found" even when the text is there. This nearly produced a false report that a docstring misattributed a quote. Use bash `grep -rn ... docs/` for anything under `docs/` |

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
| Rerank under bulk embedding | **~3.2× at production fp16** (rerank@50 p95 578–707 ms). M0's **9.17×** was an **fp32-embedder artifact** measured at top-20 — reproduced as 8.8–10× only with fp32 (D74) |
| Rerank@50 idle (fp16, real passages) | **160–165 ms p50** — resolves doc 6 §6.7's `[SPIKE]` |

---

## Measured retrieval (M2)

Quote these, with their intervals. Full detail in the M2 milestone log; the artifacts are
under `var/eval/` (gitignored).

| | |
|---|---|
| D-RET | 281 papers · **7,854 chunks** · 888 queries (629 train / 259 eval) · **CORE 53 over 41 papers** · 1,876 judgments, all grade 2 |
| Split | by paper — 207 train / 69 eval, **0 overlap**, verified not asserted |
| Dense baseline (CORE) | recall@10 **0.177** · nDCG@10 **0.108** · MRR 0.113 |
| Hybrid vs dense (CORE) | recall@10 **+0.088** [+0.004, +0.177] · nDCG@10 **+0.061** · significant on 4 of 5 |
| Rerank vs hybrid | **within noise on both splits**, and CORE/FULL disagree in sign — a starved candidate set, not a weak reranker |
| Pre-rerank ceiling (CORE) | 0.262 → 0.319 → 0.329 → **0.342** → 0.417 at depth 10/20/30/50/100 — still climbing at 100 |
| Dominant failure mode | **paper confusion**: only 2.28 of the top 10 come from the query's own paper (dense). Own-paper share 0.228 → 0.266 → 0.275 for dense → lexical → hybrid |
| Retrieval stage, idle | rerank p50 **156–159 ms**, whole stage p95 **~221 ms** against NFR-P2's 800 ms |
| Under concurrent ingestion | rerank p95 **372 ms ungated → 289 ms gated**; ingestion pays 17% (209 → 173 chunks/s). NFR-P2 holds in every condition |
| Reproducibility | **exact** for metrics — 124/124 values identical across runs. Latency is not, and says so |

---

## Working style

Challenge assumptions rather than agreeing by default. Explain tradeoffs before
recommending. Prefer boring, simple solutions; complexity must answer "what breaks
without it". When a decision is genuinely the user's, ask; otherwise decide and say
why. Report failures with actual output — a negative result with a good explanation is
worth more than an unmeasured claim.
