# M1 — Walking Skeleton: Milestone Log

**Opened:** 2026-07-30 · **Closed:** 2026-09-14 · **10 commits — 7 phases, plus foundation, pre-work and a docs pass**
**Machine:** HP OMEN, RTX 5060 Laptop (8 GB), 24 GB RAM, Windows 11, Python 3.12.9

> **Objective (doc 8 §M1):** One PDF in, one cited answer out — through the real layering,
> with trivial adapters. Ugly is fine. Incomplete is fine. **Broken end-to-end is not.**

This log lives under `docs/milestone-logs/`, alongside the design documents it refers to.
Those documents are gitignored (D12); this one is not (D67). A design doc is a living
draft; a milestone log is written once at a milestone's close and reports measurements
taken against a specific commit — so it belongs in history with that commit, for the same
reason M0 committed `SPIKE-FINDINGS.md`. Read the design docs from disk; read this one
from either.

Every number below was measured on the commit it is reported against. Nothing here is
projected. Where M1 did not measure something, it says so rather than estimating.

---

## Status — 7/7 acceptance criteria

| # | Acceptance criterion | Status | What proves it |
|---|---|---|---|
| 1 | Upload a real ML paper → ask a question → receive a streamed, cited answer | ✅ **PASS** | `test_the_answer_streams_and_carries_citations`, plus the real-service run in §3 |
| 2 | Clicking a citation resolves to the correct paper, section, page, character span | ✅ **PASS** | `test_clicking_a_citation_resolves_to_the_exact_span` |
| 3 | Every emitted citation ID exists in the retrieved context (validated, not trusted) | ✅ **PASS** | `test_every_emitted_citation_id_exists_in_the_retrieved_context` |
| 4 | Chunks carry full provenance and `chunker_version` / `embedder_version` stamps | ✅ **PASS** | `test_chunks_carry_provenance_and_both_version_stamps` |
| 5 | Tables survive chunking intact | ✅ **PASS** | `test_tables_survive_chunking_intact` |
| 6 | `import-linter` fails the build on a deliberate layer violation | ✅ **PASS** | `test_a_domain_module_importing_sqlalchemy_breaks_the_build`, with `test_the_real_tree_still_passes` as the positive control |
| 7 | Full stack runs from `docker compose up` + one command | ✅ **PASS** | `TestOneCommandStartsTheWholeThing` — 3 tests |

All seven live in `tests/integration/test_m1_acceptance.py`, each named after its
checkbox, and none reaches past the API into a use case — a criterion satisfied only by
calling internals is not satisfied.

**What "real" means here, precisely.** Criteria 1–5 run against real Postgres, real
Qdrant and real GROBID over real fixture PDFs; criterion 1 additionally requires a live
Ollama. **The embedder is a deterministic stand-in**, because a 419 MB model download on
a cold cache would break NFR-T3's zero-network-calls rule for the default run — the real
`bge-base` path is covered by phase 4's adapter tests and by `test_qa.py`. These criteria
are about the spine; retrieval quality is M2's subject.

The close-out run reported **543 passed with zero skips**, so every service gate was live
and no criterion was quietly skipped.

---

## ⚠️ What M1 does **not** establish

Stated first, because the most common way a walking skeleton lies is by looking finished.

**Retrieval quality is unmeasured.** There is no Recall@k, no nDCG, no labelled query
set. The QA path retrieves *something* and the citations that come back are real — that
is the whole claim. Whether they are the *right* chunks is M2's question, and M2 owns it
deliberately so hybrid and reranking have to earn their place against a number instead of
an intuition (D56).

**M1's QA path is the ablation baseline, not the adopted design** (D56). Dense-only,
child-only context, relevance-descending order. Doc 6 adopts parent expansion and
sandwich placement; both are listed in M2's ablation table, and an axis whose control is
never built cannot be measured, only argued. The cost is stated rather than discovered
later: M1's answers are thinner than the final design's, and small-to-big is not
exercised end to end until M2.

**Citation coverage is unmeasured** — see §3. M1 guarantees that every citation the model
emits is real; it makes no claim that the model emits one per claim.

**No abstention threshold** (D60). τ is fitted on a labelled set in M2, and picking a
number now would mean defending it rather than fitting it. What ships instead needs no
calibration: an empty workspace raises, and a retrieval that finds nothing refuses
without spending a generation call.

**Latency is not profiled.** NFR-P2/P6 have M0 component numbers but no end-to-end
percentiles; that is M2 deliverable 6.

---

## 1. Scale

| | |
|---|---|
| Commits | 10 (`680a4f8` … `efc4179`), one per phase plus pre-work and a docs pass |
| Source | 67 modules · 9,598 lines |
| Tests | 27 modules · 8,860 lines · **543 tests** |
| Tests needing **no** backing service | **400 of 543** — the other 143 are marked `integration` and skip themselves when Postgres, Qdrant, GROBID, Ollama or the gitignored corpus is absent, so a bare checkout makes zero network calls (NFR-T3) |
| Alembic migrations | 4 (`0001_initial_schema` … `0004_message_ordinal`) |
| `import-linter` contracts | 3, enforced from commit one |
| Decisions recorded | **31** (D36–D66), of which 4 overturn an earlier position — two of them M0's own conclusions |

### Version stamps in force (NFR-M5)

| Artifact | Stamp |
|---|---|
| Parser | `grobid-crf-0.8.2.1/tei/1.1.0` |
| Chunker | `structure-aware/1.2.0` |
| Embedder | `BAAI/bge-base-en-v1.5/fp16/1.0.0` |
| Answer prompt | `grounded-answer/1.0.0` |

Both chunker and parser were bumped *during* M1 by defects found in the FR-I5 round trip
(§5) — which is the stamps doing their job.

---

## 2. Requirements

### Delivered, as doc 8 scopes M1

| ID | Requirement | Proven by |
|---|---|---|
| FR-I2 | Upload a PDF directly | `test_upload.py` (14 tests) |
| FR-I4 | Parsing extracts document structure, not a flat blob | `test_every_fixture_yields_usable_structure` over the 20-paper corpus |
| FR-I5 | Every chunk retains provenance | 5 files; the round trip `text[start:end] == block.text` is enforced at construction (D45) |
| FR-I6 | Chunking is structure-aware | `test_ingestion_pipeline.py`; tables lead with their caption (D41) |
| FR-Q1 | Question scoped to a workspace | `test_answering.py`, `test_retrieval.py` |
| **FR-Q2** | **Citation *validity*** — every emitted citation resolves; a label resolving to nothing is stripped | Acceptance criterion 3; layer 2 of §6.11. **Narrowed in M1 — see §3** |
| FR-Q5 | Answers streamed token-by-token | `test_answering.py` |
| FR-W1 | Workspace CRUD | `test_api.py` |
| FR-W2 | Add/remove papers, removal leaves no orphans | `test_api.py`, `test_vector_index.py`; synchronous in M1 (D62) |
| NFR-M1 | Layered architecture with dependency inversion | The `lint-imports` gate — 3 contracts — plus acceptance criterion 6 |
| NFR-M3 | Typed everywhere, static analysis in CI | The `mypy` gate — 67 source files, zero issues |

> **Traceability gap, recorded rather than fixed:** FR-I4, NFR-M1 and NFR-M3 are the only
> three with no test naming the ID in a comment. All three are genuinely proven — the
> first by the corpus test, the latter two by gates rather than tests, which is the
> correct place for them. The convention of citing IDs in comments simply was not applied
> there. Cosmetic; noted so it is not rediscovered as a hole.

### Scope taken on purpose, beyond doc 8's M1 list

Each of these was a judgement that the alternative left a structure dead through a whole
milestone, or left a requirement's stated test unpassable.

| ID | Why it came early |
|---|---|
| FR-Q6, FR-Q9 | A QA turn is persisted in full, accounting included (D59). The alternative leaves three tables and nine columns dead while §5.4.3's flow ends with "PERSIST (message + evidence set + timings + token cost)". M2's latency profiling needs per-stage timings recorded from the first turn, not retrofitted. |
| FR-A2 | 404-never-403 for another tenant's workspace. A 403 confirms the id is real — the existence leak §5.8 layer 1 exists to prevent. Free to do correctly at the boundary; expensive to retrofit. |
| FR-A4 | Two tenants, byte-identical PDFs, two paper rows, each retrieving only its own — with the *mechanism* asserted, not just the outcome: both tenants' vectors provably share one Qdrant collection, so the paper-id filter is what separates them. Recorded as debt in phase 4; closed in phase 6. |
| NFR-SEC5 | Evidence delimiters neutralised at render time. Described honestly as **partial** — it closes the trivial escape and does nothing about a passage that merely reads as an instruction. |
| NFR-U1, NFR-U2 | Tenant-scoped job status; RFC 9457 problem documents with `FailureClass.user_message` as `detail` (D65). |

---

## 3. The finding that changed a requirement

**This is M1's most interview-relevant result, and it is a negative one.**

The phase 7 end-to-end run used real Postgres, real Qdrant, real GROBID and a real local
Ollama 3B against *Attention Is All You Need*.

| Question | Citations | Factually correct |
|---|---|---|
| "What architecture do the authors propose?" (broad) | **zero** | **No** — claimed learned positional embeddings *replace* sinusoidal, the reverse of the paper |
| Targeted question (encoder depth) | ✅ cited | ✅ N = 6 encoder layers |
| Targeted question (BLEU) | ✅ cited | ✅ 28.4 |

**The pipeline was correct throughout.** Nothing was fabricated by the system, nothing
hidden, no citation invented. Retrieval returned real chunks; validation had nothing to
strip. The 3B model simply did not cite, and got a broad synthesis question wrong.

### Why that forced a requirements change rather than a fix

FR-Q2 read: *"Every non-trivial claim in the answer carries an inline citation resolving
to a specific chunk, with paper, section, and page."* That is **two requirements under
one ID with one test**, and its own *Testable via* column tested only the second —
*"every **emitted** citation ID exists"* — which is silent about a claim that emits
nothing. M1's deliverable said "**syntactic** citation validation" and §6.11 attributed
FR-Q2 to **layer 2**. Only the prose promised more.

So FR-Q2 was **split** (D66):

| | Question it answers | ID | Milestone | Kind |
|---|---|---|---|---|
| **Citation validity** | "Does every citation the answer emits resolve to real retrieved context?" | FR-Q2 | **M1** | Guaranteed, deterministically — a label resolving to nothing is stripped whatever the model does |
| **Citation coverage** | "Did each non-trivial claim receive a citation at all?" | **FR-Q11** | **M5** | Measured, not guaranteed — a property of generation |

§3.6's rule is **satisfied, not bent**: each ID now maps to exactly one milestone, where
one previously spanned two. The original wording is preserved verbatim in doc 3 rather
than deleted.

### Coverage is not entailment

The distinction is load-bearing and easy to collapse by accident. ML-12's NLI measures
**citation support precision**, defined in doc 9 §9.6 as *"of **valid** citations,
fraction where the chunk actually supports the claim"* — it is scoped to citations that
exist.

| Metric | Blind to |
|---|---|
| Citation **validity** (FR-Q2, M1) | A claim that was never cited |
| Citation **support precision** (ML-12, M5, NLI) | A claim that was never cited |
| Citation **coverage** (FR-Q11, M5) | — this is the one that sees it |

**An answer citing nothing scores perfectly on validity *and* on support precision.**
Only coverage catches the run above. Doc 9 gained the metric explicitly, with a note that
it is **not** the table's existing "Completeness" row, which measures coverage of
*reference-answer facts* by LLM judge — two different ideas that one word would have
merged.

**No code, prompt, model or threshold was changed to produce this.**
`ANSWER_PROMPT_VERSION` is still `grounded-answer/1.0.0` and the six constraints are
byte-identical. What changed was three docstrings and three sentences of user-facing copy
that had claimed the guarantee the observation disproves.

---

## 4. Decisions — 31 recorded (D36–D66)

Full text in `docs/00-decisions-log.md`. The load-bearing ones:

### Reversals — an earlier conclusion measured and overturned

| # | Was | Is | Why |
|---|---|---|---|
| D36 | Vector filter on `workspace_id` | **`paper_id` IN (…)** | §5.6.4 and §5.8 contradicted each other. A public paper is one shared row, so a workspace filter would duplicate every vector per tenant. |
| D39 | "TEI lacks page/char offsets" (D19, M0) | **TEI carries them** | The M0 run never sent `teiCoordinates`. With it: **10,621 of 10,621 sentences carry coordinates** (100%, 20/20 papers) at **1.5% latency cost** (2.73 s vs 2.69 s/paper). TEI supplies *more* than PyMuPDF — bounding boxes, not just page numbers. |
| D42 / D45 | Document-relative chunk offsets | **Section-relative**, enforced at construction | D42 declared it; nothing checked it, so D45 made `ParsedSection` carry its canonical text and assert the round trip. An invariant that lives only in docs is not an invariant. |
| D61 | "vanilla HTML + `EventSource`" (D37) | **POST + `fetch`** | `EventSource` can only issue GET, which would put a research question — possibly quoting an unpublished manuscript — into access logs, browser history and a URL length cap. The SSE *wire format* is kept; only the browser API changed, and D37's substance ("not React, no build step") is intact. |

### Structural

| # | Decision |
|---|---|
| D47 | M1's queue is Postgres `FOR UPDATE SKIP LOCKED` with a visibility lease — the job row *is* the enqueue. No broker. |
| D50 | The worker **claims as system, then runs as tenant** — `SystemContext` demands a stated reason, so the bypass stays greppable. |
| D53 | Postgres before vectors; chunks stamped in two phases. A vector whose row is gone is a findable orphan; a row whose vector is gone is silently unretrievable. |
| D57 | The answer prompt lives in `application/`, not `infrastructure/llm/` — §6.10's six constraints are business rules, the same whichever vendor serves the tokens. |
| D26 + D58 | Ollama first, Groq second, **inverting §5.11's hosted-then-local reading** — Groq's free tier is 12,000 tokens/min ≈ ~2 RAG requests/minute (M0), unusable as a default; the local 3B runs at 125 tok/s fully GPU-resident with no quota. D58 keeps M1's chain to ordered failover and nothing more: the token bucket, circuit breaker and `Retry-After` accounting are M3's, because a breaker with no measured failure rate to tune against is furniture. |
| D62 | FR-W2's cleanup is synchronous; only FR-W3's unbounded workspace cascade waits for M3. |
| D63 | The worker runs in the API process behind `worker_in_process`, so M3's split is a config change, not a restructuring. |
| D64 | The layering gate is tested in a **copy** of the tree — planting a violation in the real `src/` is one interrupted test away from breaking the gate for everyone. |
| D65 | RFC 9457 problems under relative `/problems/{slug}`; one dependency is the entire auth seam, so M3 replaces JWT verification in one function. |
| D66 | The FR-Q2 split (§3). |

---

## 5. Defects found — and what found them

The pattern worth noting: **a green suite found none of these.** Every one surfaced from
writing something new — a test that failed, a mutation that survived, or a review of code
that was already passing.

| Defect | Found by | Phase |
|---|---|---|
| Migration 0003 bound a string against a `uuid` column — asyncpg binds VARCHAR, Postgres refuses the implicit cast | Running it against real Postgres | 5 |
| Chunker duplicated overlap text — a chunk carried 1,480 chars inside a 1,319-char span | The FR-I5 round-trip assertion | 5 |
| Section paths not unique — only **28.9% of GROBID heads carry numbering**, so duplicate paths mis-attributed chunks | Real corpus, not fixtures | 5 |
| `SKIP LOCKED` could be removed with nothing failing | **Mutation testing** | 5 |
| Section persistence made insert-only with nothing failing — the test passed because the retry *crashed* and rolled back | **Mutation testing** | 5 |
| Conversation turns returned in random order — `created_at` defaults to `now()`, which is *transaction* time, so question and answer got identical timestamps | The acceptance test | 6 |
| The evidence fence could be closed from inside it by a PDF containing the literal delimiter | Self-review of a control the docstring called "raises an attacker's cost" | 6 |
| An abandoned SSE stream discarded the whole turn — answer, evidence, timings, quota already spent | Self-review | 6 |
| Validation ran after `200 OK` — `stream()` was an async generator whose body does not run until first `__anext__()` | Self-review; phase 7 could never have rendered the problem document | 6 |
| **A tenancy hole**: the paper router did a raw `session.get(Workspace, id)` with no owner filter — §5.8 layer 2 bypassed in the layer whose job is not to do that | Caught while writing it | 7 |
| A blank question (`"   "`) passed `min_length=1`, reached the domain, and became a 500 | Self-review | 7 |
| The layering-gate test **passed while proving nothing** — it invoked `python -m importlinter.cli` with a wiped environment, exited 0 with empty stdout, and the assertion never ran | Verification pass | 7 |
| The table acceptance test asserted something a mid-table fragment also satisfies | Verification pass | 7 |
| The documented entrypoint did not exist — `settings.py` and `.env.example` named `scholiast.api.app:app`, which has no module-level `app` | Verification pass | 7 |
| README and UI copy claimed "every claim is cited" — the guarantee D66 had just disproved | Grep before committing phase 7 | 7 |

`import-linter` also rejected the obvious phase 7 implementation — an API lifespan
constructing an `IngestionWorker`. The rule was right: an HTTP layer that imports a
background worker cannot be deployed without one. That is what produced `scholiast.main`
as a composition root, and it is the clearest case in the project of a machine-checked
contract catching a design smell rather than a typo.

### Mutation testing

Run against the safety-critical paths rather than the whole tree.

**Killed:** `audit_citations` trusting every label (5 tests) · delimiter neutralisation
removed (3) · the `finally` persistence cleanup removed (2) · the duplicate-write guard
removed (3) · `stream()` reverted to an async generator (3) · a planted domain→SQLAlchemy
import (the layering gate, exit 1).

**Survived — and the survival is the result:** dropping Qdrant's `paper_id` filter leaked
nothing through the QA path, because the tenant-scoped chunk repository discarded the
foreign chunks anyway. §5.8 claims three layers *"each independently sufficient to prevent
a leak"* — that is now **measured rather than asserted**. Layer 3 alone is pinned by phase
4's adapter tests, which the same mutation does kill.

---

## 6. Debt carried out of M1

Recorded, not silently re-litigated. Nothing here is a correctness bug in M1's scope.

| To | Item |
|---|---|
| M2 | Retrieval **quality** is asserted nowhere — deliberately; that is M2's labelled set and ablation table |
| M2 | §5.7 batch sizing: 32 is ~420 ms, not the ~50–100 ms the priority semaphore needs |
| M2 | `Embedder` port lacks `precision` though `Reranker` has it |
| M2 | Two concurrent workers would race on section replacement; M1 runs one and nothing enforces it |
| M2 | `ChunkRepository.add_many` is unscoped — pre-existing, inconsistent with the now-scoped delete |
| M2 | `_paper_titles` is an N+1; fine at ≤8 evidence items, wants a batch read |
| M2 | `GenerationChunk.done` and `TokenUsage.reported` are set by adapters and read by nothing — **verified: zero call sites** |
| M2 | A refusal records a sub-millisecond `generation_ms` for a turn where no generation happened |
| M3 | The DB session is held open across generation — a concurrency ceiling, not a correctness bug |
| M3 | FR-I10 OOM halve-and-retry: `GpuBudgetExceeded` exists but nothing raises it |
| M3 | The lease (300 s) equals the GROBID client timeout (300 s); unreachable with one sequential worker, real once Celery brings a fleet |
| M3 | Queue `fail`/`complete`/`heartbeat` are unscoped by job id; only `claim` is guarded. **Re-checked at M1 close:** phase 7 exposes `GET /jobs/{job_id}`, but it is a tenant-scoped *read* (joined through the job's workspace), and the three mutating methods have no HTTP surface — worker-only. Still not reachable; the phase 5 note's trigger condition has partially arrived. |
| M3 | Groq's `Retry-After` is recorded in the error detail but nothing consumes it |
| M3 | Re-chunking orphans vectors for chunk ids that no longer exist — the §5.6.4 sweeper's job |
| M3 | A DB-level failure inside the pipeline poisons the session, so the classified detail is replaced by the rollback error. It recovers; it is lossy |
| M3 | `OllamaProvider.placement()` has no caller on the request path |
| M5 | **FR-Q11 has a requirement, an acceptance criterion and a metric definition, and no code.** Deliberate: the labelled answer set it measures over is doc 9 §9.6's, which M5 owns |

---

## 7. Gates at close (`efc4179`)

```
ruff format --check src tests    94 files already formatted
ruff check src tests             All checks passed!
mypy src                         Success: no issues found in 67 source files
lint-imports                     Contracts: 3 kept, 0 broken
pytest                           543 passed, 1 warning in 288.06s
```

The one warning is a Starlette deprecation about `httpx` raised on import of FastAPI's
test client — upstream, in a dev dependency, not Scholiast code.

| | Phase 5 | Phase 6 | Phase 7 |
|---|---|---|---|
| Tests | 416 | 496 | **543** |
| Without services | 315 | 382 | **400** |
| Coverage, as the gate computes it | 93% | 93% | **91%** |

NFR-T1's floor is 75% on core packages, so the gate passes with room. The drop from 93%
is phase 7's API surface and `main.py` (48% — the composition root's worker branch only
runs under a real process, not under the ASGI lifespan the tests drive).

### ⚠️ A measurement defect in the coverage number itself, found at close

`[tool.coverage.run]` sets `source`, `branch` and `omit`, but not `concurrency`.
SQLAlchemy's asyncio layer runs its work inside greenlets (`greenlet_spawn`), and
coverage.py's default `thread` tracer does not follow a greenlet switch — so lines that
execute underneath an `await` on the database can be recorded as never executed.

**Demonstrated on an isolated, deterministic case.** `test_removal_is_idempotent` passes
and asserts `removed is False`, which is only reachable through the early return in
`RemovePaperUseCase.execute`. Running that one test, changing nothing but the tracer:

| Tracer | `remove_paper.py` | Lines reported missing |
|---|---|---|
| `thread` (as configured) | 72% | `64-65, 69-72, 74->80, 78, 99, 102->exit` |
| `greenlet, thread` | **86%** | `65, 78, 99, 102->exit` |

Both figures reproduced exactly across repeat runs. The branch the test demonstrably
executes is reported as dead under the configured tracer.

**Across the full suite**, on an identical 3,031-statement basis, missed statements fall
from **199 to 180** — 91% → 92%. That one-point figure should be read as approximate:
full-suite coverage carries run-to-run noise from timing-dependent branches in the
polling worker (`workers/ingestion.py` moved 83% → 80% between two runs of the same
tests). **The isolated case above is the proof; the full-suite delta is only an
indication of size.**

Nothing about the tests changed and NFR-T1 is met on either number. It is recorded
because the project's claim is that its numbers are real, and this one had an artifact in
it — three commit messages now carry a coverage figure that understates.

**Deliberately not fixed here.** Adding `concurrency` to `pyproject.toml` changes what CI
computes, and that is a decision to take on purpose with the before/after recorded, not a
line to slip into a milestone-log commit. Measured, flagged, left for M2.

---

## 8. Ready for M2?

**Yes, and M2 is the project's thesis.** M1 built the spine: one PDF in, one cited answer
out, through real layering with the contracts machine-checked. What it deliberately did
not build is any way to say whether the answer was *good*.

M2 has what it needs: per-stage timings persisted from the first turn (D59), parent chunk
rows already written so parent expansion is a constructor argument rather than a re-chunk
(D56), version stamps on every derived artifact so "was it the reranker or the re-chunk?"
is answerable (NFR-M5), and an ablation baseline that exists as running code rather than
as a paragraph.

The one thing M1 hands M2 that was not planned is **FR-Q11** — a requirement that exists
because a real 3B model, on a real paper, answered a real question without citing
anything, and the requirement as written could not tell the difference.
