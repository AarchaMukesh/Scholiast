# M3 — Production Shape: Milestone Log

**Opened:** 2026-09-26 · **Closed:** 2026-09-30 · **8 phase commits, one per phase**
**Machine:** HP OMEN, RTX 5060 Laptop (8 GB), 24 GB RAM, Windows 11, Python 3.12.9

> **Objective (doc 8 §M3):** Convert the skeleton into a system that is multi-tenant,
> asynchronous, observable, and tested. **Swap adapters — do not rewrite business logic.**
> If M1's layering was right, this is mostly infrastructure work.

This log lives under `docs/milestone-logs/`, alongside the design documents it refers to.
Those documents are gitignored (D12); this one is not (D67).

**The objective's premise held.** No use case was rewritten to accommodate Celery, the
circuit breaker, the cascade or the metrics. The three `import-linter` contracts never
broke, and the two times the layering refused something it was catching a deployment fact
rather than being pedantic — D96 (the API cannot import the task it dispatches) is D63's
argument applied a second time.

**Seven of the eight acceptance criteria pass as written. One was changed — by two
decisions — and the reasons are the most interesting findings in the milestone**: both are
cases of a document describing something the code could not do. Stated in the table and
argued in §3.

---

## Status — 7/8 as written, 1 amended with cause

| # | Acceptance criterion | Status | What proves it |
|---|---|---|---|
| 1 | Two users cannot see each other's data — proven by the negative suite, not by inspection | ✅ **PASS** | `tests/integration/test_api_isolation.py`: 11 endpoints × 4 assertions, and a `SpyVectorIndex` proving a refused request never reached retrieval — so "the request died before retrieval" is the property under test rather than an inference from a 404 (D91) |
| 2 | Killing a worker mid-ingestion loses no accepted job | ✅ **PASS** | Needed **no new proof**, and that is the finding: M1's lease tests already establish it, and D95 keeps the job in Postgres, so Celery cannot weaken a guarantee it does not hold |
| 3 | Re-adding the same paper produces **zero** duplicate chunks | ✅ **PASS** | M1's FR-I7 idempotency, unchanged and still covered — deterministic chunk ids plus `uuid5` point ids (D46, D51) |
| 4 | Each failure class produces the correct state, retry behaviour, and message | ✅ **PASS** | `tests/unit/test_failure_taxonomy.py`, whose `SPEC` table is **transcribed from §4.4.1 rather than derived from the code**, so a change to the code fails a test instead of silently redefining the spec |
| 5 | Deleting a workspace leaves zero residue in Postgres, Qdrant, and object storage | ✅ **PASS** | `tests/integration/test_cascade_deletion.py` counts rows in each store rather than trusting a return value, and asserts the whole object-store prefix is empty — not just the keys we thought to name |
| 6 | Vector index rebuildable from scratch by one documented command | ✅ **PASS** | `uv run scholiast-reindex` (D101). Verified against the real database: `--dry-run` reports **67 chunks**, which is exactly what the live collection holds |
| 7 | Every degradation mode demoable and disclosed in the response | ⚠️ **AMENDED (D105, D106)** | Restated as *every mode the system **has*** — three of them, all disclosed and all tested end to end, including through an open circuit breaker. Two documented rungs were vacuous and one described the happy path; see §3 |
| 8 | Coverage target met (≥75% on core packages); CI fully green | ✅ **PASS** | See §7. Coverage is above target on every core package, with the §5.12 caveat about async DB code stated again |

---

## ⚠️ What M3 does **not** establish

Stated first, because an infrastructure milestone's most convincing failure mode is a
capability that is present in the tree and absent from the request path.

**FR-O4 distributed tracing is not implemented** (D109). Priority S, no OpenTelemetry
dependency, and no collector in the dev topology. What §5.12 says the trace is *for* — the
QA path making the latency breakdown self-evident — is emitted as per-stage timings tied
across processes by a correlation id, so the questions are answered and the visualisation
is not. Doc 3's stated test for it is "trace presence test", which a span exported nowhere
would satisfy; that is the version this milestone declined to ship.

**The SSRF allowlist is not built** (D115). Doc 8 lists it under M3's deliverables and the
surface audit found nothing to protect: no endpoint accepts a URL, uploads are multipart,
and every outbound call goes to a configured host. It becomes real in **M4**, where
discovery dereferences URLs that arrive in third-party API responses, and it is recorded
as that milestone's precondition rather than built with no caller.

**The rate limiter is per-process** (D112). Real today, because there is exactly one API
process and generation happens only in it. It weakens the moment a deployment scales out,
and nothing in the code will say so — this entry is where a reader finds out.

**`/metrics` and `/health/ready` are open unless a token is configured** (D114). The
mechanism exists and is off by default. A static token is a shared secret with no rotation
story that travels on every scrape; it is a network-policy substitute, not a replacement.

**Three §5.12 metrics have no producer and were dropped** (D108): cost per request (D14 —
no paid inference), GPU utilization and wait time (D24 — no cross-process VRAM accounting,
and D97 left the gate unwired), cache hit rate (there is no application cache). A gauge
that always reads zero is worse than an absent one, because a reader trusts it.

**Embedding throughput is not exposed.** It is the one §5.12 metric with a real producer
that this milestone does not surface: it is genuinely worker-local and has no durable row
to read at scrape time (D111).

**Two declared queues carry nothing yet.** `backfill` exists for M4's discovery flow and
`dead-letter` is a parking space; §5.7.4's priority queues are therefore structurally
present and unproven.

**The browser UI is still unauthenticated and therefore broken** (D88). `static/app.js`
sends no `Authorization` header, so every call from it returns 401. The API is complete and
driven over HTTP by the integration suite; the deferral costs a browser session, not a
capability. It is owned by the phase that next touches the front end.

**Concurrent load is still unmeasured** (NFR-S4's 20 users), as it was at M2 close.
Nothing in M3 measured throughput under concurrency; the rate limits are judgement calls
sized against NFR-S4, not measurements.

---

## 1. Scale

| | M2 close | M3 close |
|---|---|---|
| Tests | 796 | **1218** |
| Source files (mypy) | — | 109 |
| Decisions | D1–D83 | **D1–D115** (32 added) |
| Migrations | 0005 | **0006** |
| `import-linter` contracts | 3 kept | **3 kept** |

**Nine commits: eight phases plus one `docs:` commit** that records an environment trap
found while closing phase 4 (§6).

---

## 2. What each phase established

| Phase | Commit | The claim it earned |
|---|---|---|
| 1 auth | `50354eb` | Argon2id + HS256, stateless access tokens and rotated single-use refresh tokens with family revocation (D84). The **uniformity test** collapses 12 failure causes into one shape set and asserts `len(shapes) == 1`, so FR-A5's no-enumeration-oracle is a property rather than a promise. `M1_USER_ID` no longer appears in `src/` |
| 2 tenancy | `1fe20cf` | §5.8's three layers, with layer 1 asserted by a 52-case negative suite. D92 split **ownership** from **visibility** in the queue — the cascade needs a scope that includes soft-deleted workspaces, and using visibility would have turned a deletion into an ingestion loop |
| 3 Celery | `c9cf6a5` | Celery as **transport**, Postgres as source of truth (D95). Dispatch by name, never by import (D96). Retry counting stays in the database, because two budgets would dead-letter at a number neither states |
| 4 taxonomy | `7d92b9c` | §4.4.1's six classes all reachable. `RESOURCE_EXHAUSTED` had **zero producers**; FR-I10's ladder now halves in-adapter and escalates only when a single item will not fit (D98). A classification survives the session that could not record it |
| 5 cascade | `cb892c2` | §5.6.4's mark-then-sweep completed, a reconciliation sweeper behind it, and NFR-R6's rebuild command (D99–D102). The sweeper's three targets include **orphaned papers**, which no document names |
| 6 providers | `886f3b1` | `Retry-After` consumed rather than discarded (D103); a circuit breaker per rung, in process, whose open window is the server's stated one (D104). What it buys is latency on the ladder, not fewer failures |
| 7 observability | `a1d55b2` | JSON logs with redaction at the boundary, a correlation id on the **job row** rather than the message (D107), a readiness probe that checks, and `/metrics` (D108, D110, D111) |
| 8 hardening | `354b934` | NFR-SEC7's rate limiter (D112, D113), D110's decision closed with an optional scrape token (D114), and an audit finding seven of nine NFR-SEC requirements already satisfied |

---

## 3. The amended criterion, argued

Criterion 7 is the only one not met as written, and two decisions amend it. Both are cases
of a design document describing behaviour the code could not perform. In each case the
document was corrected and the reasoning recorded, rather than the code
grown branches that cannot execute — which is **D97's precedent**: there, a GPU gate had
been built and measured and turned out never to have been wired to production, and the
wiring was removed rather than left to advertise a capability that was not connected.

### `reranker_unavailable` and `dense_retrieval_unavailable` are vacuous (D105)

`AnswerQuestionUseCase.__init__` takes **no reranker**. The `Reranker` port and
`LocalReranker` adapter exist and only the eval harness builds one, because reranking
measured as a **null against hybrid** in M2 and FR-Q3's adoption decision is deliberately
open. A rung that degrades from a component not in the request path is a documented failure
mode that cannot occur.

`dense_retrieval_unavailable` says "lexical-only retrieval". D73 built the sparse arm and
production is dense-only — no composition root passes `sparse=True` — so there is no
lexical index to serve from, and losing Qdrant loses both arms at once because both live in
it.

**Rejected: wire both into production so the rungs become real.** That adopts reranking,
which M2 measured as a null and FR-Q3 leaves open, and switches on the lexical arm,
changing indexed output and invalidating every M2 comparison. Two adoption decisions
smuggled in under a degradation phase.

### `local_generation` now describes the happy path (D106)

§5.11 predates D26 and reads hosted-first with local as the last resort. **D26 inverted
that on a measurement**: Groq's free tier is ~2 RAG requests per minute, so hosted-first
makes the development default unusable after two questions, and `build_provider_chain` has
put Ollama first ever since.

Keeping the rung would mark nearly every response `degraded: local_generation`, and a flag
that fires on almost every request carries no information. That inverts NFR-U4 and AP-7: a
system where "normal" is spelled "degraded" launders reduced quality in the opposite
direction, by making the warning ordinary. What §5.11 wanted disclosed is not lost — the
response carries `provider` and `model` on every answer, because the identity of the
generator is a fact about a response rather than a warning about it.

**The same principle caught a bug in phase 7.** Logging a failed Celery dispatch would have
warned on *every* upload in the default single-process topology, because `NullDispatcher`
returns False by design (D63). `JobDispatcher.publishes` exists to keep that log line
meaningful.

---

## 4. Amendments to earlier decisions

Three decisions in this log were amended after they were written, each because writing the
next phase surfaced something the entry had assumed.

| | What it said | What was true |
|---|---|---|
| **D94** | "the box restarted" | Only Docker Desktop restarted; OS boot was 2026-09-24. The flake's explanation changed with it |
| **D97** | A cross-process GPU gate was needed because the worker was moving out of process | The gate was **never wired to production at all** — only `eval/cli.py` builds one, so D79's 372→289 ms was a *harness* measurement and the request path has always run ungated. The mechanism ships; at the owner's decision it is not enabled, and the dead wiring was removed |
| **D100** | A `beat` service in docker-compose | Compose **cannot** host one: there is no Dockerfile and no `build:` stanza, so it holds backing services plus Flower's third-party image and nothing containing Scholiast. Beat runs natively beside the worker; containerising is M6's |

---

## 5. Defects found — and what found them

The point of this section is the second column.

| Defect | Found by |
|---|---|
| **`GET /v1/jobs/{id}`'s 404 named the job's own workspace** — an identifier disclosure *and* an existence oracle, in a response whose status code was already correct (D93) | Enumerating all fourteen raise sites to check whether it was the only one |
| **A vacuous completeness test.** `app.routes` yields `_IncludedRouter` objects with no `.path`, so the routed set was empty and the assertion passed against anything | A mutation that deleted an endpoint and survived |
| **A toothless isolation test**: with a non-retryable class, scoped and unscoped reads agree, so the test could not fail | Mutation |
| **A TTL test that asserted Redis, not the gate.** It planted a lock with its own expiry and watched it expire — proving Redis expires keys | Mutation removing the gate's TTL entirely |
| **`RESOURCE_EXHAUSTED` had zero producers.** §4.4.1 documented it and `GpuBudgetExceeded` was raised nowhere: a failure mode the spec promised and the code could not reach | Reading the taxonomy table against the code |
| **A classification rewritten by a database hiccup.** A DB failure earlier in the pipeline leaves the session aborted, so the write recording *why* a paper failed also raised — and the crash handler reclassified everything as TRANSIENT | The M1 log recorded it as debt; phase 4 closed it |
| **The blob purge leaked every non-current TEI generation.** M1 deleted `source_key` plus the `tei_key` the row records, and D49 keeps several generations on purpose — so "zero residue" meant "zero residue among the keys we named" | Reading D49 against the purge while writing the cascade |
| **A race no ordering can fix.** Two workspaces sharing a paper, both deleted, each seeing the other's membership still uncommitted, so both skip the purge and the paper survives with no members — unreachable by every scoped query. It predates the phase: two concurrent `RemovePaperUseCase` calls have it today | Reasoning through the cascade's concurrency; closed by making orphaned papers a sweep target (D102) |
| **The rebuild would have indexed parent windows** — 111 vectors where ingestion wrote 67 — making a parent a matchable hit competing with the child inside it. Nothing would have failed; every retrieval number in M2 would have moved | `scholiast-reindex --dry-run` against the real database, whose count disagreed with the live collection |
| **`python -m importlinter.cli` is a no-op that exits 0.** No `__main__` guard, so `-m` imports the module and exits — accepting any arguments and checking nothing. It reported `imports=0` for a gate that never ran | Noticing the banner was missing, then running `--help` and `total nonsense --bogus` and getting the same silence |
| **Phase 7's logging bound `sys.stdout` at configuration time.** `create_app` configures logging, so the first test to build an app bound the stream for the whole process and later writes went to a closed object | An unrelated 500 in phase 8's rate-limit suite — an order-dependent failure |
| **The rate limiter had dead code and a latent bug.** An eager `_prune` deleted expired entries before `check` looked, making the window-roll branch unreachable; worse, it applied the current call's window to *every other bucket's* entries. Invisible because all three windows are 60 s | A mutation that disabled the roll branch and broke nothing |
| **A module-level rate limiter** would have made test results order-dependent, one test's requests counting against another's | Writing the second test against it |
| **`app.state.dispatcher` regression.** Making `enqueue` record a dispatch meant every custom lifespan had to set it; the isolation suite's Postgres-only lifespan did not, so uploads 500'd. Fixed by defaulting it in `create_app` rather than patching the test | Running the existing suite after the change |
| **An invented Celery setting** (`task_queues_late_ack_visibility`) that does not exist | Self-review of the diff before commit |

### Four of these were my own tests being wrong

Worth separating, because a test that passes for the wrong reason is the failure mode this
project's mutation discipline exists to catch — and it caught all four.

- Phase 4: a unit test drove `_abandon` directly and never the site that *creates* the
  carrier, so deleting the wrap left it green.
- Phase 6: the failed-probe test gave the probe a `Retry-After`, so the breaker re-opened
  through the *header* branch and the test passed with the probe logic deleted.
- Phase 7: the unreported-token test passed `TokenUsage(0, 0, reported=False)` —
  incrementing by zero changes nothing either way.
- Phase 8: the configured-allowance test used the *peer*-keyed dependency, leaving the
  user-keyed path (QA and uploads, the expensive ones) uncovered.

### Mutation runs

| Phase | Mutations | Survivors after correction |
|---|---|---|
| 3 GPU gate | 7 | 0 |
| 4 taxonomy | 14 | 0 |
| 5 cascade | 17 | 0 |
| 6 breaker | 18 | 0 |
| 7 observability | 20 | 0 |
| 8 rate limiting | 20 | 0, **plus 2 retired as behaviourally equivalent** |

The two retired mutants are named rather than quietly dropped. Incrementing a rate-limit
count on a *refusal* changes nothing observable — `started` is untouched, so the window
rolls at the same instant and the count is only compared against a limit it has already
passed. And `==` versus `compare_digest` differs only in timing; a timing assertion on a
32-byte compare is a flaky test pretending to be evidence, so the constant-time compare is
justified by reasoning recorded in the code rather than by a test that would pass either
way.

---

## 6. Environment traps added to CLAUDE.md

One, and it cost a false "five gates green" claim:

**`python -m importlinter.cli` exits 0 without checking anything.** In a CI log that is
indistinguishable from a passing gate. It is the pipe-exit-code trap in different clothes —
a green signal produced by a command that did no work — and it is *not* the Smart App
Control workaround recorded above it in the table, which is the `lint_imports_command`
*function* called in-process. Conflating the two is what produced the dead gate. The tell
is recorded: a real gate 4 always prints `Contracts: 3 kept, 0 broken`, so a silent gate 4
is a failed one.

A second environment fact was found and **not** added as a trap, because it is a test
defect rather than a machine one: `test_m1_acceptance.py::test_a_domain_module_importing_sqlalchemy_breaks_the_build`
fails under PowerShell and passes under Bash, because PowerShell sets
`PYTHONIOENCODING=utf-8:surrogateescape` — the child `lint-imports` then emits UTF-8 while
the parent's `subprocess.run(text=True)` decodes cp1252, so `stdout` comes back `None`. It
is listed as debt.

---

## 7. Gates at close (`354b934`)

All five ran and passed in one session on the tree being committed. "Five gates green" is a
claim about a run, not the machine (CLAUDE.md), and gate 4 was run through the
`lint-imports` launcher — not the `-m` route that reports success without working.

```
ruff format --check src tests   151 files already formatted
ruff check src tests            All checks passed
mypy src                        no issues in 109 source files
lint-imports                    Contracts: 3 kept, 0 broken   (138 files, 593 dependencies)
pytest                          1218 passed, 0 failed, 0 skipped
```

### Coverage at close

**88% overall** (M2 closed at 87%), 1218 tests, measured on this commit.

| Package | Coverage | |
|---|---|---|
| `domain` | **98%** | stdlib only, no I/O — the easiest to cover and the one where a gap would matter most |
| `infrastructure` | **92%** | |
| `application` | **91%** | |
| `api` | **91%** | |
| `eval` | 77% | M2's harness, unchanged this milestone |
| `workers` | **65%** | and the number is honest rather than flattering — see below |

**Every core package is above the ≥75% target. `workers` is not a core package, and its 65%
is two specific gaps rather than thin testing:**

- `reindex.py` — **0%**. NFR-R6's CLI. `RebuildIndexUseCase` behind it is covered at the
  unit level and the command was exercised by hand against the real database (`--dry-run`
  reported 67 chunks, matching the live collection exactly, which is how the parent-window
  defect in §5 was found) — but the argument parsing and adapter wiring have no automated
  test. Stated rather than averaged away.
- `tasks.py` — **56%**. The Celery task bodies need a worker process to execute. The work
  they delegate to (`IngestionWorker`, `MaintenanceWorker`) is covered at 87% and 100%, and
  a test pins that the task *names* and queue routes match what the API dispatches (D96) —
  which is the failure that would otherwise be silent.


The §5.12 caveat still applies and is still unfixed on purpose: `[tool.coverage.run]` has
no `concurrency`, and SQLAlchemy's asyncio layer runs inside greenlets, so lines executing
under an `await` on the database read as dead. **Do not read a low number on a repository
or a use case as a missing test** until it is re-run with `concurrency = greenlet, thread`.
Changing the setting changes what CI computes, which is why it has not been changed.

---

## 8. Debt carried out of M3

| Debt | Owner | Why it is not done |
|---|---|---|
| **FR-O4 tracing** | M6 | Priority S; needs a collector, which needs a deployment (D109) |
| **SSRF allowlist** | **M4** | Nothing dereferences a user-influenced URL until discovery does (D115). The requirements are written down in D115 so M4 inherits them |
| **The browser UI is 401 on every call** | The phase that next touches the front end | D88: a login screen plus token storage plus 401-refresh-retry is front-end work whose review has nothing to do with Argon2id |
| **Rate limiting is per-process** | M6 | Correct at one replica; a deployment that scales out must replace it (D112) |
| **`X-Forwarded-For` is not trusted** | M6 | Correct with nothing in front; a deployment adding a proxy must add trusted-proxy handling in the same change, or every anonymous caller shares one bucket (D113) |
| **`/metrics` and `/health/ready` open by default** | M6 | The token mechanism exists and is off; network policy is the real control (D114) |
| **Embedding throughput not exposed** | M6/M7 | Worker-local with no durable row; needs either a worker metrics endpoint or a durable counter (D111) |
| **`backfill` and `dead-letter` queues carry nothing** | M4 | Declared now because a queue added after the workers are deployed is a queue nothing consumes until every worker restarts |
| **The GPU gate is not enabled in production** | A phase with its own before/after measurement | A gated embedder re-batches and perturbs fp16 vectors (D74), so switching it on changes indexed output. Doing that as a side effect of another change would make two changes inseparable (D97) |
| **The M1 layering test fails under PowerShell** | Any phase | One line: `encoding="utf-8", errors="replace"` on that `subprocess.run`. §6 |
| **Concurrent query load unmeasured** (NFR-S4) | M5/M6 | Carried from M2 close unchanged |
| **Coverage `concurrency` setting** | — | Deliberately unfixed: changing it changes what CI computes (M1 log §7) |

---

## 9. Ready for M4?

**Yes, and the preconditions are named rather than assumed.**

M4 is the discovery engine: academic source clients, parallel fan-out with partial results,
deduplication, and multi-signal ranking. What it needs from M3 and now has: durable
asynchronous work with bounded retries and a failure taxonomy that can classify a source
outage; a provider chain with a circuit breaker whose pattern the source clients should
reuse; correlation ids that survive the API-to-worker boundary; and a queue (`backfill`)
already declared for bulk work.

**Two things M4 must do before it ships, both recorded:**

1. **Build the SSRF allowlist** (D115). Discovery dereferences URLs from third-party API
   responses — attacker-influenced the moment a malicious record is in OpenAlex or a
   redirect points at link-local metadata. Scheme allowlist, host allowlist, private and
   link-local ranges checked *after* DNS resolution, redirects re-validated at every hop
   rather than followed, and a response size cap.
2. **Produce work on the `backfill` queue**, which exists and is empty, so §5.7.4's
   priority separation stops being structural and starts being exercised.

**One thing M4 should resist.** FR-Q3's adoption decision is still open and production is
still dense-only. Enabling hybrid, the reranker or the lexical arm makes two of §5.11's
degradation rungs implementable (D105) — and that is not a reason to do it. Each is an
adoption decision with a measurement attached, and M2 built the harness precisely so those
are earned rather than assumed.
