# M4 — Discovery Engine: Milestone Log

**Opened:** 2026-09-30 · **Closed:** 2026-10-09 · **13 phase commits, one per phase**
**Machine:** HP OMEN, RTX 5060 Laptop (8 GB), 24 GB RAM, Windows 11, Python 3.12.9

> **Objective (doc 8 §M4):** Close the loop — topic in, ranked and deduplicated candidate
> papers out. The feature that stops this being a PDF chatbot.

This log lives under `docs/milestone-logs/`, alongside the design documents it refers to.
Those documents are gitignored (D12); this one is not (D67).

**The loop is closed.** A topic goes to OpenAlex and arXiv at once under one deadline;
the records come back as one entry per paper, ranked by fitted weights, each score
explained in the UI; a paper can be added by its identifier, and the worker fetches its
PDF through an SSRF guard and indexes it. Every number that chose something was measured
against a bar written down before the measurement, and four live passes against the
sources were pre-registered and hash-locked before their first request.

**Six of the seven acceptance criteria pass as written. The seventh, criterion 1
(NFR-P1), is not met as written, by the rule fixed before the run that tested it**
(D158, D160): a cached search meets its 400 ms budget with room to spare (p95 42 ms),
and a cold search does not meet 3.0 s at p95, because 4 of 29 lost both sources and
OpenAlex answered inside the 2.0 s deadline on only 15 of 29. The cause is measured to
the request (§3). This log puts that first, because it is the result most easily lost
behind a closed loop.

---

## Status — 6/7 as written, 1 not met with cause

| # | Acceptance criterion | Status | What proves it |
|---|---|---|---|
| 1 | Topic query returns ranked candidates within NFR-P1 (≤3 s cold, ≤400 ms cached) | ❌ **NOT MET AS WRITTEN** (D158, D160) | Addendum A4, live, the real app over 6 fresh starts: **cached p95 42 ms ✅**, zero outbound requests on every cached repeat ✅; **cold p95 is a failed search ❌** (4 of 29 returned 503), the 25 that answered p50 2.28 s, max 2.44 s; **completion ❌** OpenAlex 15/29, arXiv 22/29 against 24. §3 |
| 2 | arXiv preprint and camera-ready of the same paper collapse into one entry | ✅ **PASS** | `test_dedup.py::test_a_preprint_and_its_camera_ready_are_one_work` (an arXiv id against an EMNLP DOI, joined by title) and `test_a_journal_version_two_years_on_is_one_work` (two years apart, which exact-year blocking would never join); live, DPR "found in openalex and arxiv" in D155's walkthrough |
| 3 | Dedup precision/recall reported on the fixture set | ✅ **PASS** | `eval --dedup` on 379 labelled pairs frozen before the matcher existed (D146): held out, **precision 1.000 [0.930, 1.000], recall 0.927 [0.827, 0.971]**, no false merge; identifiers alone reach 0.291 (D145, D148) |
| 4 | Score breakdown visible in the UI ("why did this rank here") | ✅ **PASS** | "Why here?" lists each part, its weight and what it added, imputed parts marked; walked through live (D155) and seen again in A4's UI check |
| 5 | Killing one source still returns partial results with per-source status | ✅ **PASS** | `test_discovery_fanout.py`: the real adapters under fault injection, 11 cases across the two sources (a server error, the source's own throttle with `Retry-After`, a malformed body, a read timeout, a refused connection), each leaving the other source intact; live in A4, 5 of 6 kill searches partial with the right statuses, the sixth a 503 because the *other* source also missed the deadline on that search; in the UI, "openalex did not answer in time; showing what the others found. 50 papers." |
| 6 | Second identical query issues zero outbound calls | ✅ **PASS** | `test_a_users_second_identical_search_sends_nothing` against real Redis, and `test_queries_that_normalise_alike_share_an_entry` (phase 4); live in A4, **0 outbound requests during each of 12 cached repeats**, counted in the app process |
| 7 | Ranking weights ablated, not hand-tuned by intuition | ✅ **PASS** | A 0.1 simplex fitted on 10 training searches of 1,029 blind graded judgments; held out, scored once: nDCG@10 **+0.053 [+0.003, +0.100]** over the sources' own order (D149, D151, D153) |

---

## ⚠️ What M4 does **not** establish

Stated before the findings, as in every milestone log.

**NFR-P1 for a cold search.** Criterion 1 is not met (D160). The live run was one machine,
one network, one afternoon and one user, and p95 over 29 searches sits next to the
maximum; but the rule was fixed before the run and says "not met", and the measured cause
(§3) is the sources' time on a fresh connection, which no part of this milestone changed.

**Why OpenAlex's body takes 0.8 s.** Measured: a near-constant 0.80 s from headers to last
byte on a fresh connection per search. *Not* measured: whether a pooled connection, which
pays the handshakes and TCP slow start once per process, removes it. That is the obvious
next experiment, and it is a decision against D119 and D122 that needs its own
pre-registered run.

**A fresh-process effect.** On the first search after a start a source sent no headers
at all on 4 of 12 requests, against 3 of 46 later; but on all 6 kill starts the surviving
source answered. Suggestive, not established.

**Ranking quality beyond 10 held-out searches.** The judged pool is two-thirds on-topic
(D149), so a ranker's room is mostly ordering 2s above 1s; the judge wrote the ranker; the
recency weight shows no measurable effect over dense relevance alone (−0.008 [−0.027,
+0.010]). Citation impact is weighted 0 because blind topical judgments cannot credit it,
not because it was shown useless.

**Dedup beyond the candidate set.** Recall is within pairs that share an identifier or a
title overlap of at least 0.55; the labeller saw the held-out pairs while labelling, so
only the thresholds were fitted blind to them; Wilson intervals assume independent pairs,
and pairs share records.

**Two sources, not three, and no paper encoder.** Semantic Scholar answered 429 to 10 of
10 keyless requests and was never measured (D118). Doc 8's SPECTER2-class encoder for
dedup was not built: every miss is lexical, and an identifier or title rule already holds
precision at 1.000 (D145).

**Anchors on the live day.** D126 asked for the 20 anchors re-checked live. Only 5 topics'
keyword searches kept their result lists (the first part's were lost when the harness
crashed, D160), and in 4 of those one source missed the deadline, so today's 3 of 10 is
confounded and is not a ranking measurement.

**Load and contention.** One search at a time throughout. GPU contention between
discovery's dense ranking and ingestion is unmeasured; concurrent load (NFR-S4) is still
unmeasured, as at M2 and M3 close.

**The UI's behaviour beyond one walkthrough.** Verified by hand once (D155) and once more
in A4's UI check; a later change to `app.js` is protected only by the pytest guards.

**Discovered papers are private to whoever added them** (D131). Sharing licence-permitted
papers is the named upgrade path, gated on three shared-path defects phase 6 mapped.

---

## 1. Scale

| | M3 close | M4 close |
|---|---|---|
| Tests | 1218 | **1929** |
| Source files (mypy) | 109 | **136** |
| Decisions | D1–D115 | **D1–D160** (45 added) |
| Migrations | 0006 | **0008** |
| `import-linter` contracts | 3 kept | **3 kept** |
| Live passes against the sources | 0 | **5** (spike P1–P5, A1, A2, A3, A4), every one pre-registered and hash-locked |

**Thirteen commits, one per phase:** 1–5, 6a and 6b, 7, 8a and 8b, 9, 10a and 10b. Phase
6 split into adding and fetching a discovered paper, then queue priority and backfill;
phase 8 so the judged set was frozen in history before any ranker existed; phase 10 so
A4's rule was committed before its data.

---

## 2. What each phase established

| Phase | Commit | The claim it earned |
|---|---|---|
| 1 spike | `71b9a77` | A pre-registered live spike before any discovery code (D116): **ASM-1 fails** (no source meets latency, completeness, limits and relevance together), **ASM-2 passes** (83.9% of 180 works fetch a PDF). 354 requests, zero budget, pacing, address or redirect violations, checked by a log checker rather than asserted |
| 2 adapters | `2015305` | One port, OpenAlex and arXiv behind it, stdlib-only normalisers shared by cache, dedup and adapters (D119). Addendum A1 kept arXiv's AND construction against OR, by a rule written before the requests (D120). Two sources, not three (D118) |
| 3 fan-out | `c9005b6` | One 2.0 s deadline for every source, a fixed status set, 503 rather than an empty list when none answers (D122). The circuit breaker extracted from M3 into one core, which **exposed a latent M3 defect**. A2 chose OpenAlex's page size (25) and found its ranking changed under us (D125, D126) |
| 4 cache | `b073612` | Each source's page cached per user for a day, failures never stored, in its own capped Redis so a full cache costs hits, never ingestion (D127, D128). A cached search through the endpoint: p95 3.5 ms |
| 5 SSRF guard | `107825f` | D115 built: every hop checked, **every** resolved address public, the connection pinned to the checked IP with TLS verified against the real host, redirects by hand, a byte cap while streaming (D129). The allowlist is the measured minimum: three hosts, 74.4% of the spike's papers (D130) |
| 6a add | `c5625fb` | A discovered paper added by `{source, source_id}` alone, made from the source's record, never the client's; a paper may exist before its PDF (migration 0007); FETCHING through the guard (D131–D138). Live: arXiv 2004.04906 to INDEXED in 16.5 s |
| 6b backfill | `768fa46` | Priority lives in the claim, uploads before discovered papers (D139, migration 0008); **M3 precondition 2 met**: backfill is produced and consumed; a beat sweep finds what no enqueue announces (D141). Live with a real Celery worker and beat, the project's first |
| 7 dedup | `d80f720` | A rules-first matcher fitted on train topics only and scored once held out: precision 1.000, recall 0.927 (D145–D148). A search result is a work, not a record (D147) |
| 8a judgments | `b4813f4` | A3 re-recorded what production would rank, and 1,029 works were graded blind against a guideline written first, then frozen by hash before any ranker code (D149, D150) |
| 8b ranking | `66af30a` | Dense relevance 0.8, recency 0.2, citation 0.0, **fitted**, held out +0.053 nDCG@10 (D151–D153). Every score is exactly the weighted sum of what is shown. Work vectors cached, so a cached search embeds only its query: p95 25 ms |
| 9 UI | `050ed32` | The UI logs in again (**closes D88**), a Discover view with "why here?", third-party text rendered inert under a strict CSP (D154), and pytest guards for what no Python test would see (D155) |
| 10a warm start | `59aca00` | The first search after a start went from 16.2 s to 0.59–0.82 s: the encoder is warmed before the API accepts a request (D156) and loads from the local copy, not after 32 requests to the model hub (D157). A4 written and committed before its first request (D158, D159) |
| 10b live run | *this commit* | A4 run under one deviation (DEV-A4-1): criterion 1 not met as written, the cause measured per request; criterion 5 shown live in the API and the UI (D160) |

---

## 3. The criterion not met, argued

D158 fixed the rule before the run: both p95 bounds, **and** each source `ok` on at least
80% of the cold searches, because D122 had named the way to fake this criterion (a tight
deadline that drops the slow source returns fast partial results).

| | Bar | A4, 2026-10-09 | |
|---|---|---|---|
| Cold p95 (29 valid searches) | ≤ 3.0 s | a failed search (4 of 29 were 503) | ❌ |
| Cold searches that answered (25) | — | p50 **2.28 s**, max **2.44 s** | (all under 3.0 s) |
| Cached p95 (12 repeats of complete searches) | ≤ 0.4 s | **42 ms** (p50 30 ms) | ✅ |
| Outbound requests during a cached repeat | 0 | **0** on all 12 | ✅ |
| OpenAlex `ok` | ≥ 24 of 29 | **15** (52%, [34%, 69%]) | ❌ |
| arXiv `ok` | ≥ 24 of 29 | **22** (76%, [58%, 88%]) | ❌ |

**Where a cold search's time goes**, timed inside the app process: the fan-out is
deadline-bound (p50 1.98 s); dense ranking of works not yet embedded adds p50 0.29 s, p95
0.42 s; everything outside the use case, p50 4 ms. So when the sources answer, a cold
search lands near 2.3 s, inside 3 s. The budget split of D122 holds; the sources do not
fill their 2.0 s reliably.

**Why OpenAlex misses**, from headers and last byte timed separately for every request:

- When it finished, its headers came at p50 **0.99 s** and its body took a near-constant
  **0.80 s** more (p50 0.80, p95 0.81) for 290–330 KB decoded.
- **11 of its 14 misses already had headers** (1.19–1.88 s) and ran out of deadline
  mid-body. OpenAlex fits 2.0 s only when its headers arrive by about 1.2 s.
- A3 (the day before) had it inside 2.0 s on 18 of 20. This day was slower, which is the
  variability D126 warned about, and the run measures the day it ran.

A body time that constant on a fresh connection per search (D119) is what TCP slow start
over a long round trip looks like, more than a server's generation time. **That is a
hypothesis.** It names the next experiment: one pooled connection per source per process,
measured with its own pre-registered run, against the alternatives of a smaller OpenAlex
page or a longer deadline. Each trades against NFR-P1's split (D122), which is why none
was tried inside the run that was measuring the system as shipped.

**The deviation, stated.** A4 stopped part-way: the laptop entered modern standby 2 s into
the 20th cold search (Kernel-Power 506), despite a keep-awake hold that covers idle sleep
only. DEV-A4-1, approved by the owner before anything more ran, voided that one search,
resumed in a new start, raised the request maximum (failed searches re-ask their failed
source, D127, more than planned), **and recorded that criterion 1 was already decided**
before the remainder ran: 3 failed searches already put the p95 on a failure, and OpenAlex
could no longer reach 24. The remainder ran for what A4 still owed: criterion 5, the UI
check, the first-after-start sample and the cached numbers.

**What was not done, on purpose.** No retry, no longer deadline, no rerun until it passed.
D158 says a source that is slow for everyone on the day fails the bar for a reason the code
cannot fix, and the log says so rather than choosing a better afternoon.

### Addendum (2026-10-12, after M4 closed): the cause investigated, and the criterion closed

Written after this log was closed; the run above and its verdict are unchanged. The
decisions are D163, D169 and D170.

**Measured since.** Each run timed the sources alone, through the production adapters, on
this machine and network:

| Run | Date | OpenAlex @25 inside 2.0 s | arXiv @50 inside 2.0 s |
|---|---|---|---|
| A5, new connection per request (D163) | 2026-10-10 | 29/30 | 23/30 (at a 15 s gap) |
| A6-multiday day 1 (D169) | 2026-10-10 | 28/30 | 26/29 |
| A6-multiday day 2 | 2026-10-11 | 28/30 | **0/29** (27 HTTP 429s, 2 timeouts) |
| A6-multiday day 3 | 2026-10-12 | 25/30 | 28/29 |

- **The slow-start hypothesis above is refuted** (A5). OpenAlex's TCP connect took p50
  6 ms and its TLS handshake p50 9 ms, and its body took p50 0.93 s on a new connection
  against 0.92 s on a reused one. The body is OpenAlex streaming its response, not our
  connection handling. Pooling was not adopted, by its own pre-registered rule.
- **A6-multiday tested four configurations**, OpenAlex at 25 or 15 results × a 2.0 or
  2.5 s deadline. None cleared the 80% bar on all three days: all four failed on day 2
  alone, when arXiv refused every request, and passed on days 1 and 3.

**Concluded.** The cold misses are the sources' own latency and availability: OpenAlex on
slow days (A4), arXiv on a day it refused every request and on days it ran at about 76–77%
against the bar's 80% (A4, A5). D158 counts a source that is slow or down for everyone as
a miss, on purpose, so the criterion records the product's dependence on them. The levers
our code holds are spent against them: pooling is refuted, and a smaller page or a longer
deadline was unnecessary for OpenAlex at the shipped settings on A6-multiday's three days
and does nothing against a refusal.

**Not concluded.** That NFR-P1's cold bound cannot be met. The evidence covers two
sources, one machine and network, five days and the shipped configuration. A5 and
A6-multiday timed the sources alone and do not re-judge criterion 1; only a new
pre-registered end-to-end run can.

**Closed (D170).** Criterion 1 stays NOT MET AS WRITTEN, now closed under the tested
conditions. The requirement is not changed after the fact, and no further latency levers
or live runs are pursued for it now. A requirement change or a different source mix, fixed
in advance and judged by a new end-to-end run, remains the owner's option.

---

## 4. Amendments to earlier decisions and documents

| | What it said | What was true |
|---|---|---|
| **Doc 8 §M4** | Semantic Scholar / OpenAlex primary, arXiv secondary | **OpenAlex and arXiv.** S2 keyless answered 429 to every request and was never measured (D118) |
| **Doc 8 §M4** | Dedup with a SPECTER2-class encoder | **Rules only.** Every miss is a title at overlap 0.78–0.82; an encoder answers a question the data did not ask (D145) |
| **Doc 5 §5.4.1** | Cache the response, keyed on the normalised query | **Each source's page, per user** — a whole-response cache would re-ask every source after a partial search (D127) |
| **D88** (M3) | The UI is 401 on every call, owned by the next front-end phase | **Closed** by D154 |
| **D95** (M3) | The in-process poller defaults to false | The code defaults to the poller; amended by D141 |
| **D115** (M3) | SSRF allowlist deferred to M4 | **Built** (D129, D130) |
| **D118** | OpenAlex primary, on 70% anchor recall | OpenAlex changed its ranking three days later (14 anchors → 6); D118 stood, ranking moved to phase 8's judged set (D126) |
| **D122** | The adapters warm a shared TLS context; a fresh connection per search | Still true, and A4 measured what the fresh connection costs (§3) |
| **D123** | A search could wait for arXiv's slot before its deadline | Amended before coding: a search never waits; arXiv is reported `rate_limited` at once |
| **D142** | A throttled host's wait is taken in the worker | Amended after self-review: the host is deferred and the serial worker is never held |
| **D156** | The encoder load was the suspect for the search that lost both sources | Refuted by the walkthrough's own server log: the load had finished 14 s earlier, and arXiv was down for everyone |
| **A4** (D159) | Kill a source by refusing its connection; kill it as "down" with a 503 | Amended twice before approval, by the dry run: Windows takes 2.04 s to refuse, so "refused" reads as `timeout`; and the arXiv adapter reads 503 as throttling, so "down" answers 500 |
| **FR-I1, FR-I7** | Assigned to M3 | Impossible without discovery; moved to M4 with a criterion, and met (D136) |

---

## 5. Defects found — and what found them

| Defect | Found by |
|---|---|
| **A half-open breaker probe that ended without a verdict never released its slot**, so the breaker refused every later call until restart. Latent in M3's LLM breaker; discovery's deadline would have made it routine | Extracting the breaker into a shared core for sources (phase 3); two tests failed on M3's code first |
| **Each search built a TLS context**: 183 ms of blocked event loop per source, outside what A2 timed | A test's run time (phase 3) |
| **The first search after a start took 16.2 s**: 32 model-hub requests and the model load inside the request | D155's live walkthrough; split by measurement in phase 10a |
| **Five defects in the spike's own scripts** before the first live call: every refused URL crashed the probe, queued requests could overrun a cap or reach a stopped host, two URL shapes raised, S2-only URLs would have reached a tracked file, a restart forgot host state | Review of the harness, each shown by an offline test that failed first (phase 1) |
| **A stop file checked before the pacing wait**, letting one more request through | A1's own incident (phase 2) |
| **Code that should not exist**: unwrapping Teredo could only make some Teredo addresses acceptable | Mutation (phase 5) |
| **An `Optional` narrowed through `bool()`** that tests could not see and a hand check missed | mypy, once Smart App Control let it run (phase 6a) |
| **The worker waited out a host's Retry-After**, holding the serial worker for a host's time | Self-review (phase 6b) |
| **"Retrying automatically" was shown exactly when it was false** | Reading the message against the state machine (phase 6b, D143) |
| **Celery's `queue_order_strategy="priority"` ordered queues by string hash**, differently in every process | Phase 6b's priority work |
| **A real encoder fell back to lexical under a mocked network** — the model load asked the hub and respx refused it | The test with the real encoder, before shipping (phase 8b) |
| **Four UI defects**: a stale "n/m ready", "PDF · PDF" link labels, a doubled parenthesis, and a browser running a cached `app.js` (hence `Cache-Control: no-cache`) | The live walkthrough (phase 9) |
| **The A4 harness kept its records in memory**, so a crash lost the first part's result lists | The crash (phase 10b); records are now written as they happen |
| **After a 503 the UI keeps "Searching OpenAlex and arXiv…" beside the error** | A4's UI check; **not fixed** (§8) |

### Tests that were wrong, all found by mutation

- Phase 3: a login test disabled rate limiting and so proved nothing, because the limiter
  resolves the user anyway; a pacer test hung instead of failing under a broken pacer.
- Phase 4: an "older format" cache entry was refused for missing fields, not its format;
  the endpoint tests used one user, so a cache shared across users passed.
- Phase 5: no body was shorter than the PDF magic; a concurrency test's mock never yielded,
  so the per-host lock was never needed.
- Phase 6a: attaching a paper that has its PDF was tested only with a record carrying no
  links; the merge test used an empty workspace, where "any member" and "this member" agree.

### Mutation runs

| Phase | Mutations | Result |
|---|---|---|
| 1 spike harness | 11 (+3 in phase 3) | 14 caught |
| 2 adapters | 21 | 21 caught |
| 3 fan-out | 37 | 37 caught |
| 4 cache | 24 | 24 caught |
| 5 SSRF guard | 33 | 28 caught, **5 kept as equivalent on this Python** |
| 6a add | 60 | 60 caught, 2 after strengthening their tests |
| 6b backfill | 34 | 34 caught |
| 7 dedup | 35 | 35 caught |
| 8b ranking | 38 | 38 caught |
| 9 UI | 15 | 15 caught |
| 10a warm start | 8 | 8 caught |
| **Total on production code** | **305** | **300 caught, 5 equivalent** |

The five equivalent mutants are the guard's explicit loopback, link-local, IPv4-mapped,
Teredo and local-use NAT64 checks: on Python 3.12.9 `is_global` already refuses every
address they target. They stay, because `ipaddress`'s tables have been wrong before
(CVE-2024-4032), and in an SSRF guard a table error is a hole.

---

## 6. Environment traps added to CLAUDE.md

**Modern standby stops a long run, whatever the keep-awake hold says.** The app's hold
prevents idle sleep only; the laptop entered standby twice in 14 minutes on mains power
during A4, once 2 s into a search. A long measurement that shows a 120 s client timeout or
a gap should be checked against Kernel-Power events 506/507 before anything is debugged.

**A refused connection takes Windows 2.04 s to report.** It retries a closed port before
giving up, so under discovery's 2.0 s deadline a refused source is indistinguishable from
a hung one and reads `timeout`. On Linux a refusal is immediate. A test or demo that
expects `error` from a refused connection on Windows is wrong.

---

## 7. Gates at close

All five ran and passed in one session on the tree being committed, gate 4 through the
`lint-imports` launcher. Smart App Control (D72) blocked nothing this time; it blocked mypy
for about two hours during phase 6a and the `lint-imports` launcher during phase 8a, where
the in-process `lint_imports_command` route ran the same three contracts.

```
ruff format --check src tests   211 files already formatted
ruff check src tests            All checks passed
mypy src                        no issues in 136 source files
lint-imports                    Contracts: 3 kept, 0 broken   (182 files, 907 dependencies)
pytest                          1929 passed, 0 failed, 0 skipped
```

### Coverage at close

**90% overall** (M3 closed at 88%), branch coverage, 1929 tests, measured on this tree.

| Package | Coverage | |
|---|---|---|
| `domain` | **98.6%** | stdlib only; discovery's normalisers, dedup and ranking live here |
| `api` | **94.5%** | |
| `infrastructure` | **93.7%** | the SSRF guard, both adapters, the cache and the breaker core |
| `application` | **93.2%** | |
| `eval` | 82.1% | `--dedup` and `--ranking` added this milestone |
| `workers` | 72.5% | the Celery task bodies need a worker process, as at M3 close |
| `main.py` | 55.9% | the composition root; no test boots it end to end, which is what the live runs did |

Every core package is above NFR-T1's 75%. The §5.12 caveat from the M3 log still applies:
`[tool.coverage.run]` has no `concurrency`, so lines under an `await` on the database read
as dead, and a low number on a repository or use case is not by itself a missing test.

The live numbers in §3 are not a gate. They are `spike/results/20_discovery_live.json`,
produced under protocol hash `1da1a928...` by `spike/scripts/20_discovery_live.py`.

---

## 8. Debt carried out of M4

| Debt | Owner | Why it is not done |
|---|---|---|
| **NFR-P1 cold** (criterion 1) | Before M6, its own change | The measured cause is the sources on a fresh connection (§3). A pooled connection per source is the hypothesis; it, a smaller OpenAlex page and a longer deadline each trade against D122, and each needs a pre-registered measurement. **Update, 2026-10-12:** all three were measured in pre-registered runs. Pooling was refuted (A5, D163), and the three-day measurement of page size and deadline found no configuration that passed every day (A6-multiday, D169). The debt is closed as not met under the tested conditions, not shown impossible to meet (D170, §3's addendum) |
| **The UI's status line after a 503** | The next front-end change | Found by A4's UI check; the 503 path does not clear `#search-status` |
| **arXiv's 503 reads as rate limiting** | Any phase touching the arXiv adapter | arXiv throttles with 503 and `Retry-After`, so phase 2 maps every 503 to `rate_limited`; a real outage that answers 503 inside the deadline is shown to the user as throttling |
| **A fresh-process effect** on the first search after a start | With the pooling measurement | Suggestive at 4/12 against 3/46, contradicted by 0/6 on kill starts; not established. **Update, 2026-10-12:** the pooling measurement (A5, D163) timed the adapters, not fresh app processes, so it did not test this; it stays not established |
| **The reranker still asks the model hub on every load** | Whichever phase enables reranking | `CrossEncoder` is constructed without D157's local-first rule; production does not rerank today, so only the eval pays it |
| **A containerised API needs the model in its image** | M6 | D156 loads the encoder at boot; a container without it fails to start, or downloads 419 MB |
| **A second API process needs a shared arXiv pacer** | M6 | D123's pacer is per process; two processes would double arXiv's rate |
| **Semantic Scholar** | When a key exists | Never measured keyless (D118) |
| **Sharing licence-permitted discovered papers** | A later milestone | D131 keeps them private; three shared-path defects are mapped |
| **Two concurrent adds of the same new work by one user is a 500** | Any phase | The per-owner unique index; an upload's content hash has the same race |
| **A discovered paper whose fetch failed stays FAILED in that workspace** | Any phase | D136's remedy route is recorded, not built |
| **Reconciliation can delete a fresh upload's PDF before its row commits** | Any phase | Needs object timestamps in the store port (D144) |
| **GPU contention between discovery ranking and ingestion** | M5/M6 | Unmeasured; phase 8b measured on an idle GPU |
| **`get_sentence_embedding_dimension` is deprecated** in sentence-transformers | In progress separately | A FutureWarning, not a failure |
| Carried from M3: FR-O4 tracing, per-process rate limiting, `X-Forwarded-For`, open `/metrics`, embedding throughput, the unwired GPU gate, the PowerShell layering test, concurrent load (NFR-S4), coverage `concurrency` | As in the M3 log §8 | Unchanged by M4 |

---

## 9. Ready for M5?

**Yes, with criterion 1's cold half named as open rather than closed.**

M5 is grounding and comparison: citation coverage (FR-Q11), comparison across papers, and
the measurement of whether answers stay grounded. What it needs from M4 and now has: papers
that arrive by discovery as well as upload, each with a source identity and a version-
stamped path through the same ingestion; a UI that logs in; and a discovery ranker whose
every score is explainable, which is the same discipline M5's grounding needs.

**What M5 should not inherit silently.** NFR-P1's cold half is not met, and the reason is
measured. The pooling experiment is small and self-contained; whether it runs before M5 or
alongside it is the owner's call, but it should run as its own pre-registered change, not
as a side effect of M5's work, so that its before and after stay separable.

**Update, 2026-10-12.** The pooling experiment ran before M5's work, as its own
pre-registered change, and was refuted (A5, D163); a three-day measurement of the other
levers followed (A6-multiday, D169). Criterion 1's cold half, named above as open, is now
closed as NOT MET under the tested conditions (D170, §3's addendum). That is not a finding
that it cannot be met.

**One thing M5 should resist**, unchanged from M3: FR-Q3's adoption decision is still open
and production is still dense-only for QA. Hybrid and the reranker are adoption decisions
with measurements attached, not defaults to switch on because M5 touches retrieval.
