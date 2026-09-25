# M2 — Evaluation Harness & Retrieval Quality: Milestone Log

**Opened:** 2026-09-15 · **Closed:** 2026-09-25 · **8 commits — one per phase**
**Machine:** HP OMEN, RTX 5060 Laptop (8 GB), 24 GB RAM, Windows 11, Python 3.12.9

> **Objective (doc 8 §M2):** Stop guessing. Build the labeled dataset and metric pipeline,
> establish a baseline, then earn every retrieval improvement with a number.
>
> **This milestone is the project's thesis. If the timeline slips, everything else gets
> cut before this does.**

This log lives under `docs/milestone-logs/`, alongside the design documents it refers to.
Those documents are gitignored (D12); this one is not (D67).

Every number below was measured on the commit it is reported against, on the machine
above. Nothing here is projected. **Three of M2's headline results are negative**, and
they are reported in the same detail as the positive one — which is the point of building
the harness before forming an opinion.

---

## Status — 8/8 acceptance criteria

| # | Acceptance criterion | Status | What proves it |
|---|---|---|---|
| 1 | `eval` command runs end-to-end and emits a reproducible report | ✅ **PASS** | `uv run eval` and five comparison modes, each archiving a date-stamped JSON with a full config snapshot (§9.12) |
| 2 | Two consecutive runs agree within a stated tolerance | ✅ **PASS** | **Exactly**, for every metric: a phase 5 re-run reproduced phase 4's artifact at **124 of 124 quality values** under float equality, and the smoke gate reports **+0.000 on all ten metrics**. The tolerance is stated in §3 — it is not a band, it is equality, and latency is excluded by name |
| 3 | Retrieval ablation table populated with real numbers **on D-RET-CORE**, with bootstrap intervals | ✅ **PASS** | Four arms, five candidate depths and two fusion rules, every number an interval (§2). See the caveat below: it is three artifacts, not one table |
| 4 | **D-RET-FULL reported beside CORE**, in its own block, without graded nDCG — and disagreement stated (D69) | ✅ **PASS** | Every block carries both; FULL omits nDCG in the report builder, not the printer. Two disagreements are reported rather than reconciled (§2) |
| 5 | Hybrid + rerank compared against dense-only, **result reported honestly either way** | ✅ **PASS** | Reranking is a **null result** against hybrid and is published as one (§2), with the diagnostic that explains it |
| 6 | p50/p95/p99 latency reported per stage | ✅ **PASS** | Per stage (embed / search / rerank / total) on an idle system, and under concurrent ingestion with the GPU gate on and off (§2) |
| 7 | Train/eval split verified leak-free (split by paper, not by query) | ✅ **PASS** | `verify_split` raises rather than warns, runs in every eval invocation and in the suite: 207 train papers, 69 eval papers, **0 overlap** |
| 8 | CI smoke eval wired and gating | ✅ **PASS** | A `smoke` job; **exit 1 demonstrated on a planted regression**, exit 0 clean (§4) |

**Criterion 3's caveat, stated rather than glossed.** Doc 8 calls the ablation table "the
headline artifact". What exists is three archived artifacts — `arms-*`, `depths-*`,
`fusion-*` — each internally complete with intervals and paired deltas, rather than one
rendered table. Assembling them into a single published view is a reporting job (§9.13),
not a measurement one, and it is listed as debt.

---

## ⚠️ What M2 does **not** establish

Stated first, because an evaluation milestone's most convincing failure mode is a number
that looks more load-bearing than it is.

**Grade 1 is still unpopulated; every judgment is 2/0** (D69). nDCG@10 is valid and is the
least discriminating form of the metric doc 9 designates as the headline ranking number.
Every within-noise result in this milestone is therefore weaker evidence than the same
result would be on the intended 3-point scale, and the hand review that would fix it did
not happen.

**The corpus is LaTeX-derived, and the gap to production is unmeasured** (T12). Phase 7
set out to measure it and found that it cannot be measured this way: **40.6% of QASPER's
evidence strings carry a `BIBREF<n>` citation placeholder** the typeset PDF never
contains, so judgments do not transfer to a GROBID build by string matching — only **15 of
112** survive. The question D68 posed is open.

**Adoption decisions are not made.** FR-Q3 asks whether hybrid+rerank beats dense-only on
the eval set; the numbers exist, and production is still dense-only. Changing it is a
decision with a latency argument attached, and this milestone deliberately supplies the
evidence rather than the verdict.

**Concurrent query load is unmeasured** (NFR-S4's 20 users), as is end-to-end ingestion
(NFR-P6 covers parse + chunk + embed + write; only the embed contention was measured).

**Answer quality, groundedness and abstention are untouched** — doc 9 §9.6's subject and
M5's milestone. M2 measured retrieval, which is the ceiling on all three.

---

## 1. Scale

| | |
|---|---|
| Commits | 8 (`eb65e0f` … `f345874`), one per phase |
| Source | 80 modules · 14,437 lines (M1 closed at 67 · 9,598) |
| Tests | 36 modules · 12,206 lines · **796 tests** (M1 closed at 543) |
| The eval package | 11 modules · 4,130 lines — a real package, not a script folder |
| Decisions recorded | **15** (D68–D82), of which 3 re-plan a phase boundary and 2 correct an earlier measurement |
| `import-linter` contracts | 3, unchanged and green throughout |
| Archived eval artifacts | 16 runs across 7 kinds (`retrieval`, `compare`, `arms`, `depths`, `fusion`, `contention`, `pdf-check`) |

### The dataset (D-RET, `d-ret/1.0.0`)

| | |
|---|---|
| Corpus | 281 QASPER validation papers · **7,854 indexed chunks** |
| Queries | 888 total — **629 train**, 259 eval |
| D-RET-CORE | **53 queries over 41 papers**, the primary benchmark (D69) |
| Judgments | 1,876, all grade 2 |
| Evidence accounting | 2,808 strings: 2,349 exact · 77 whitespace-normalised · 253 float references · 129 section headings · **0 unresolved** |
| Split | by paper, 207 train / 69 eval, **0 overlap**, verified not asserted |

### Version stamps in force (NFR-M5)

| Artifact | Stamp |
|---|---|
| Dataset | `d-ret/1.0.0` |
| Corpus builder | `qasper-json/1.0.0` |
| Chunker | `structure-aware/1.2.0` |
| Embedder | `BAAI/bge-base-en-v1.5/fp16/1.0.0` |
| Reranker | `BAAI/bge-reranker-base/fp16/1.0.0` |
| Parser (PDF check only) | `grobid-crf-0.8.2.1/tei/1.1.0` |

The reranker stamp was **missing from the first phase 4 artifact** and added before that
commit: a block that reports a cross-encoder's ordering while naming only the embedder
cannot answer "was it the reranker or the re-chunk?", which is the question NFR-M5 exists
for.

---

## 2. The measurements

### Hybrid beats dense-only — the one clear positive

Paired deltas against the M1 dense baseline, D-RET-CORE (n=53), same queries and corpus:

```
recall@10  +0.088 [+0.004, +0.177]  significant
recall@20  +0.090 [+0.034, +0.165]  significant
nDCG@10    +0.061 [+0.020, +0.108]  significant
MRR        +0.059 [+0.012, +0.116]  significant
recall@50  +0.023 [-0.057, +0.115]  within noise
```

RSK-8 recorded the honest possibility that hybrid would not win. It did. The mechanism is
visible rather than inferred: the share of the top 10 drawn from the query's own paper
rises dense **0.228** → lexical **0.266** → hybrid **0.275**, which is exact-identifier
matching attacking the dominant failure mode phase 2 found — **paper confusion**, with
only 2.28 of the top 10 from the right paper.

### Reranking is a null result, and the reason is a starved candidate set

Against hybrid, on identical fused shortlists, every delta straddles zero on both splits —
and CORE and FULL disagree in sign on recall@10 and recall@20. Phase 4 published that
without being able to say why. Phase 5's candidate-depth sweep says why:

| Candidate depth | CORE ceiling | FULL ceiling | rerank p50 | stage p95 |
|---|---|---|---|---|
| 10 | 0.262 | 0.236 | 35 ms | 69 ms |
| 20 | 0.319 | 0.307 | 70 ms | 109 ms |
| 30 | 0.329 | 0.335 | 109 ms | 157 ms |
| **50** (adopted budget) | **0.342** | **0.375** | 159 ms | 221 ms |
| 100 (diagnostic, D77) | **0.417** | **0.449** | 312 ms | 411 ms |

At the adopted top-50 budget **roughly two-thirds of relevant chunks never reach the
cross-encoder**, and the ceiling is still climbing at 100. Doc 9 §9.3 calls pre-rerank
recall "the diagnostic most people omit" precisely because it separates *"the reranker is
weak"* from *"the candidate set never had the answer"*. It was the second.

Depth 100 buys recall@50 (CORE **+0.052** [+0.004, +0.113], FULL **+0.041** [+0.018,
+0.067]) and nothing in the top 10, at double the rerank cost — a recall gain with no
ranking gain. **Not a production recommendation**: doc 6 §6.6 fixes the budget at top-50.

### Two more nulls, each measured the expensive way

**Embedding model (ML-4).** bge-base vs bge-small, one invocation, one dataset: all nine
paired deltas straddle zero. bge-small indexed **2.1× faster** (24.7 s vs 52.2 s) at half
the vector memory (12.1 vs 24.1 MB) with **no per-query latency difference** (embed p50
7.59 vs 7.56 ms at batch 1). Resource advantages are real and separate; the quality
conclusion stayed at the strength the data supports, and production did not change (D71).

**Weighted fusion (ML-7).** Weight fitted on the **train split** — never the eval queries
— landed at 0.6 on dense, and every delta against RRF is within noise on both splits. The
fit is the more honest artifact: recall@10 is **0.2375 / 0.2371 / 0.2379** at weights
0.4 / 0.5 / 0.6, so the top three are separated by less than 0.001 against bootstrap
half-widths of ±0.05, and MRR would have chosen 0.5. The curve is archived whole. RRF
stays the default, now measured rather than assumed.

### The GPU priority scheme works, bounded by the batch

Doc 8 deliverable 6's number. Four conditions, back to back, only the sharing rule varying:

| Condition | rerank p95 | × idle | stage p95 | ingest chunks/s |
|---|---|---|---|---|
| idle | 196 ms | — | 221 ms | — |
| contended, ungated | 372 ms | 1.89× | 521 ms | 209.3 |
| contended, **gated** | **289 ms** | 1.47× | 438 ms | 173.1 |
| contended, gated, batch 16 *(diagnostic)* | 232 ms | 1.18× | 318 ms | 119.3 |

The gate removes **22% of the rerank tail for 17% of ingestion throughput** — affordable
against NFR-P6, which allows 90 s/paper where this costs 0.16 s. **NFR-P2 holds in every
condition**, so the gate's contribution is margin rather than rescue. §5.7 pairs the gate
with batches bounded to ~50–100 ms; D74 measured the current default at p50 103 / p95 135
ms, above that target, and the batch-16 cell shows what the bound costs: with phase 4's
batch-64 measurement the contention penalty scales **1.18× / 1.89× / 3.2×** at 16 / 32 /
64. The default stays 32; this is the number the deferred tuning decision needs.

One nuance the aggregate hides: the *query embed* gets slightly worse under the gate at
batch 32 (95 → 101 ms), because a small embed waits for a running batch instead of
time-slicing beside it. The gate helps the burst it was built for; the small call pays.

---

## 3. Reproducibility — the tolerance, stated

Criterion 2 asks for agreement "within a stated tolerance". Measured, it is stronger than
a tolerance:

| What | Agreement |
|---|---|
| Metric values, run to run, same machine | **Exact.** 124 of 124 quality values identical between the phase 4 artifact and a phase 5 re-run, under float equality — points, CI bounds, paired deltas, noise flags |
| Smoke subset, run to run | **Exact.** +0.000 on all ten metrics |
| Retrieval metrics across four contention conditions | **Exact**, and asserted by the harness: only timing may vary between conditions, so a difference would be a defect rather than noise |
| Latency | **Not reproducible, and the artifact says so.** Idle rerank p50 moved 156 → 228 ms between two runs an hour apart on the same tree — machine state, not code |

That last row is why every latency comparison in this milestone is **within-run**. One
contention run was discarded for exactly this reason (§5).

The seeds that make it exact: bootstrap `20260915`, D-RET split and CORE by hash of the
id rather than by sampling, fusion-weight grid fixed, sample seeds recorded in every
artifact that draws one.

---

## 4. The CI gate (FR-E6)

A `smoke` job runs a seeded 10-paper subset (360 chunks, 27 queries) on **CPU at fp32**,
dense and hybrid, and compares every metric with a committed baseline:

```
clean tree              PASS, exit 0 — every delta +0.000
planted 0.20 regression FAIL, exit 1 — "1 regression(s), 0 missing"
wall clock              59 s, inside doc 10 §10.6's ~2 minutes
```

**The failure path was executed, not assumed.** The first attempt to verify the exit code
read `grep`'s status through a pipe and would have "confirmed" a gate that passed every
build forever.

Regression-only semantics at 0.05 (D82): a drop fails, a rise is reported and passes. The
baseline is committed, regenerates only on an explicit flag, and states in its own
`purpose` field that these are subset numbers at CPU/fp32 and **not** the published
results. Reranking is absent from the smoke because on CPU it costs minutes — the whole
job budget.

**One risk carried into M3:** the baseline was generated on Windows and is verified on
Linux. The first CI run is what validates the tolerance. If it trips on drift rather than
on a real change, the fix is to revisit the tolerance or the baseline's provenance — not
to widen it quietly.

---

## 5. Defects found — and what found them

The M1 pattern held: **a green suite found none of these.**

| Defect | Found by | Phase |
|---|---|---|
| A conditional-body test that asserted nothing, and a CORE-stability test that called the wrong function | **Mutation testing** (mutation B2 survived) | 1 |
| nDCG normalised against an untruncated ideal | Writing the test against a hand-computed value | 2 |
| CUDA init inside the first timed query — CORE p99 **13,595 ms** against a p50 of 16 ms, and the larger FULL set hid it by diluting the outlier | The p99 column | 2 |
| `AsyncQdrantClient.set_vectors` does not exist | Running it | 3 |
| `_assert_dimensions` silently checked nothing on a named-vector collection — the exact failure its own docstring warned about | Self-review | 3 |
| Hybrid timing mislabelled: the GPU embed sat inside `search_ms`, reporting `embed p50 0.02 ms` | Reading the number and disbelieving it | 3 |
| Two mypy errors hidden while Smart App Control had gate 3 down | The gate coming back | 3 |
| The embedder took INGESTION priority for *query* embeds — the gate would have deprioritised the user-facing path | Self-review | 4 |
| A warm-up condition (`== "hybrid"`) that would silently skip the new arm | Writing the test that names the mistake | 4 |
| A vacuous gate test: the direct-handoff test asserted ordering and passed with handoff removed | **Mutation testing** | 4 |
| A docstring claiming a tiebreak prevented nondeterminism, when `sorted` already did | **Mutation testing** — deleting it changed nothing | 4 |
| The first archived rerank artifact named no reranker, and its header described a three-arm comparison it had not run | Pre-commit verification | 4 |
| Model loading charged *inside* the contention timing window — would have published "the gate costs 32% of ingestion" when it costs 17% | Noticing the gated arm's wall clock did not match its per-query sum | 6 |
| A machine-state outlier reported as a result: idle rerank p50 228 ms against 156 ms the same morning, the whole machine ~1.4× slower | Comparing the idle row against an earlier artifact | 6 |
| `--update-baseline` without `--smoke` silently did nothing | Self-review | 8 |
| An exit code verified through a pipe, reading `grep`'s status | Re-checking the one thing the gate depends on | 8 |

**Mutation testing** was run against the paths where a silent failure would be worst: the
GPU gate's handoff, the rerank ordering contract, the reranker version stamp, the terminal
report's reranker line, the ceiling diagnostic, and the fitting-set isolation. Each
mutation was killed by the test written for it, and every mutated file was restored and
verified byte-identical.

---

## 6. Debt carried out of M2

| To | Item |
|---|---|
| M3 | **The GPU gate is in-process only.** An asyncio primitive coordinates one process; M3's Celery split breaks that and needs a cross-process mechanism (D74) |
| M3 | Starvation is not addressed: continuous high-priority reranks could starve ingestion. Accepted, not solved |
| M3 | The smoke baseline's cross-platform tolerance is validated by the first CI run, not before it |
| M5 | **Grade 1 hand review** — CORE's queries pair with **1,526** chunks in their own papers, of which 100 already carry grade 2, leaving **1,426** to read. Until that happens, nDCG@10 is the least discriminating form of the headline metric (D69) |
| M5 | Candidate depth is now a live production question (REV-4), with the cost side measured: depth 100 costs p95 411 ms of NFR-P2's 800 ms idle, before contention |
| M5 | Embedding batch size stays 32; at fp16 it holds the GPU p50 103 / p95 135 ms against §5.7's 50–100 ms target, and 16 would fit (D74, D79) |
| M5 | FR-Q3's adoption decision — production is still dense-only, with hybrid measured as better |
| M5 | D68's question is open: whether the headline numbers carry to production needs hand-labelled PDF-side evidence or a non-string alignment (T12) |
| M5 | Concurrent query load (NFR-S4's 20 users) and end-to-end ingestion (NFR-P6) unmeasured |
| M6 | The ablation table exists as three artifacts; §9.13's single published view is a reporting job |
| M6 | **`BIBREF` contamination**: 47.8% of indexed chunks carry citation placeholders the lexical arm tokenises as terms (`bibref0`…). It qualifies the phase 3 lexical result and nothing else in the report mentions it (T12) |
| — | Two early `arms-*` artifacts carry a stale `latency_measured` description; superseded, not deleted |

---

## 7. Gates at close (`f345874`)

```
ruff format --check src tests    117 files already formatted
ruff check src tests             All checks passed!
mypy src                         Success: no issues found in 80 source files
lint-imports                     Contracts: 3 kept, 0 broken
pytest                           796 passed, 0 skipped, 0 failed
```

| | P2 | P3 | P4 | P5 | P6 | P7 | P8 |
|---|---|---|---|---|---|---|---|
| Tests | 618 | 649 | 687 | 736 | 754 | 779 | **796** |

Every phase's gate run was fingerprinted — a SHA-256 over the tracked diff plus the
untracked files — and the fingerprint compared before and after, so "the gates ran on the
tree that was committed" is checked rather than assumed.

### Coverage at close: 87% (M1 closed at 91%)

4,604 statements, 501 missed. The drop is the eval package's orchestration, and it sits
exactly where a reader should expect it:

| Module | Cover | What it is |
|---|---|---|
| `fusion.py`, `smoke.py` | **100%** | the weight fit; the gate's decision logic |
| `qasper.py`, `contention.py`, `pdfcheck.py` | 95–99% | corpus building, contention bookkeeping, mapping diagnosis |
| `dret.py`, `harness.py` | 89–92% | dataset assembly, retrieval runs, metrics plumbing |
| `metrics.py` | 72% | error branches on malformed qrels |
| **`eval/cli.py`** | **44%** | `run_contention`, `run_pdf_check`, `run_smoke`, `_run` |

`cli.py` is the orchestration whose only honest exercise is a real run against a GPU, a
live Qdrant, GROBID and arXiv — and **those runs are archived as artifacts rather than
asserted as tests**, which is the right trade for code whose output is a measurement. What
is tested is every decision those runners delegate: the plan and its validation, the
fusion fit, the mapping diagnosis, the contention guards, the gate's comparison. That is
why the modules holding the decisions sit at 89–100% while the module holding the wiring
does not.

NFR-T1's floor is 75% on core packages, so the gate passes with room — and the figure
carries the same greenlet-tracer caveat the M1 log recorded in its §7, unchanged here and
still unfixed on purpose, so 87% is if anything an understatement.

### Smart App Control, and what M2 learned about it

M1 closed all-green on 2026-09-14. Two days later, during phase 2, Windows Smart App
Control began blocking `grimp._rustgrimp` and mypy's `__mypyc` — taking gates 3 and 4 with
them — with no change on this side (D72). Phase 2 was committed with the blocker
documented rather than routed around. The block **lifted on its own** before phase 3,
recurred once during phase 6's commit preparation, and cleared again minutes later with
the files byte-identical. Two claims D72 made were then corrected by measurement:

- *"There is no module-level route around it"* — **false in general.** During the phase 6
  recurrence the `lint-imports` launcher was blocked while `grimp._rustgrimp` imported
  fine in-process, and the same three contracts ran that way.
- *"Persistent vs intermittent"* was a timescale artifact. Every block is time-varying;
  the installed `_rustgrimp` still matches its wheel's RECORD hash, so identical bytes
  were blocked and then allowed.

The rule that replaced them: **"five gates green" is a claim about a run, not about the
machine** — make it only when all five actually ran and passed in that session.

---

## 8. Ready for M3?

**Yes, and M2 delivered what it promised: numbers instead of opinions.** Hybrid beats
dense-only and the win is significant on four of five CORE metrics. Reranking does not
beat hybrid, and the milestone can say *why* rather than shrugging — two-thirds of the
relevant chunks never reach it. The GPU priority scheme works and the measurement says
what it costs. Weighted fusion and the smaller encoder earned nothing, measured the
expensive way, and changed nothing in production.

**The most valuable thing in this milestone is the three negative results**, because each
one came with a diagnosis. The most valuable habit is narrower: every number that looked
surprising was chased to its cause before it was published — the 13,595 ms p99, the
`embed p50 0.02 ms`, the gated arm that seemed to cost 32% of ingestion, the idle
reference that moved 1.4× overnight. Three of those would have shipped as findings.

What M3 inherits is a harness that measures configurations rather than arguing about them,
a CI gate that fails on regression, and one open question M2 could not close: the corpus
is LaTeX-derived, and whether these numbers carry to real PDFs is measured only as *"not
by string matching"*.
