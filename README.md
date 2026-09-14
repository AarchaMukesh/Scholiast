# Scholiast

A retrieval-and-reasoning platform for academic literature, built so that
**retrieval quality and groundedness are measured rather than asserted**.

Search a topic, get ranked and deduplicated papers, build a workspace, and ask
questions answered with verified citations. The features are ordinary; the point is
the engineering underneath — ablations with confidence intervals for every
retrieval decision, a calibrated abstention threshold, and honest latency numbers
for two different deployment targets.

> **Status: early.** M0 (validation spike) is complete; M1 (walking skeleton) is in
> progress. Retrieval-quality numbers arrive with the evaluation harness in M2.
> The table below is what has actually been measured so far — nothing here is
> projected.

## Measured so far (M0)

| Measurement | Result |
|---|---|
| GPU | RTX 5060 Laptop, `sm_120` native kernels, 29.0 TFLOP/s fp16 |
| Usable VRAM | **6.8 GB** of 7.96 GB nominal — the desktop compositor holds the rest |
| Embedder + reranker, co-resident | **2.36 GB** |
| Rerank 50 candidates, fp16 GPU | **224 ms** (743 ms at fp32 — precision matters) |
| Rerank 20 candidates, CPU | **493 ms** — the GPU-less production path |
| Embedding throughput | 196 chunks/s (384-dim) · 75.8 (768-dim) · 6.9 CPU |
| Section-structure extraction | **95%** usable (19/20 real ML papers), PyMuPDF only |
| Broken-PDF detection | 3/3 correctly classified |

Full spike report, including the seven design revisions it forced:
[SPIKE-FINDINGS.md](SPIKE-FINDINGS.md).

## What makes it different

- **Measured, not asserted.** A labelled evaluation set, bootstrap confidence
  intervals on every ablation, and error attribution across six failure modes — so
  "retrieval improved" is a number with an interval, not a claim.
- **Grounding is deterministic first.** A local NLI model, not an LLM, computes the
  primary groundedness metric — the same mechanism that runs in production. The
  LLM judge is a sampled cross-check on a *different* provider, validated against
  human labels with Cohen's κ reported.
- **Constraints are the design.** 8 GB VRAM and zero inference budget forced the
  local/remote split, a GPU priority semaphore, an on-demand local fallback, and a
  quota-aware provider chain. Those are the most interesting decisions in the
  project and none would exist unconstrained.

## Architecture

Modular monolith plus a worker, ports and adapters, PostgreSQL as source of truth
with the vector index as rebuildable derived state.

```
API (FastAPI)  ──ports──▶  PostgreSQL · Qdrant · Redis · object store
      │                          ▲
      └── Worker (Celery) ───────┘
             │
             ├─ local GPU: embedder · reranker · NLI  (fp16)
             └─ generation: free-tier provider chain → on-demand local fallback
```

The layer boundaries are **enforced in CI**, not merely documented:

```bash
lint-imports
```

Three contracts — dependencies point inward, the application layer never touches
adapters, and the domain imports no framework at all. A violation fails the build.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

Run the gate chain exactly as CI does:

```bash
uv run ruff format --check src tests && uv run ruff check src tests && uv run mypy src && uv run lint-imports && uv run pytest
```

> Docker Desktop with the WSL2 backend is required from M1 onward (PostgreSQL,
> Qdrant, Redis). The domain layer and its tests need nothing but Python.

### Running the stack

Two steps, as M1's acceptance criterion specifies (NFR-DOC1):

```bash
docker compose up -d
```

```bash
uv run alembic upgrade head && uv run uvicorn scholiast.main:app --reload
```

Then open <http://localhost:8000>. Upload a PDF, wait for it to reach `indexed`,
and ask a question; every citation the answer emits clicks through to the exact
span it came from, validated rather than trusted (FR-Q2). Whether the model
cites *every* claim is citation coverage (FR-Q11), a measured property of
generation reported in M5 — not something M1 guarantees (D66).

`scholiast.main` is the composition root: it runs the API **and** the ingestion
worker in one process, because M1's queue is in-process by design (doc 8). Set
`WORKER_IN_PROCESS=false` to run the API alone — which is what M3 does when
Celery takes the worker over.

Generation needs a model. The default is a local Ollama (`ollama pull llama3.2`),
which carries sustained load at no quota cost; a `GROQ_API_KEY` adds a hosted
fallback behind it. With neither, questions still return their retrieved evidence
and say plainly that they were answered without synthesis.

### GPU note

The RTX 5060 is Blackwell (`sm_120`). Install torch from a CUDA 12.8+ index — a
plain `pip install torch` may give a build with no native kernels for this
architecture, which fails at runtime or silently falls back to slow PTX JIT:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Verify with `spike/scripts/01_gpu_check.py`, which asserts `sm_120` is in the
build's arch list rather than trusting `torch.cuda.is_available()`.

## Layout

```
src/scholiast/
├── domain/          entities, invariants, chunking, fusion — stdlib only, no I/O
├── application/     use cases and ports (interfaces)
├── infrastructure/  adapters — the only layer that knows vendor names
├── api/             HTTP surface
├── workers/         async pipeline tasks
└── eval/            evaluation harness (a first-class package, not a script folder)

spike/               M0 validation scripts and results — throwaway by design
tests/unit/          domain and pipeline logic
tests/integration/   real backing services via testcontainers
```

## Limitations

Stated deliberately, because the project's claim is measurement discipline and that
only means something if the gaps are visible too.

- **No retrieval numbers yet.** The evaluation harness lands in M2; until then
  there are no quality claims to make.
- **Laptop scale.** No production traffic, no real load. Scaling *reasoning* is
  demonstrated; scaling *experience* is not.
- **Small evaluation set (~50 queries).** Only large effects will be
  distinguishable from noise, which is why intervals are reported rather than point
  estimates.
- **Single annotator** on relevance judgments — no inter-annotator agreement, so
  systematic bias is possible.
- **English, ML papers only.** The pipeline is domain-general; it was validated on
  one domain.
- **Free-tier providers may train on submitted content.** Generation runs on free
  API tiers whose terms commonly permit this. A configuration flag pins private
  uploads to the local model instead; read the terms before enabling a provider.
