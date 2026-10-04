# M4 — Discovery Spike Findings (phase 1)

**Run:** 2026-10-01, one home connection, one Windows 11 machine. Session 1 05:15–05:23 UTC ·
PDF probe 05:24–05:54 · download sample 05:54–06:03 · session 2 12:12–12:13, 6 h 49 min after
session 1's last request (the protocol requires at least 6 h).
**Protocol:** `spike/m4-protocol.json`, approved at canonical SHA-256 `6a280902…` (D116),
amended to `6f46de9a…` by DEV-2 after the first 15 requests (D117), and given addendum A1 as
`b23f235e…` before A1's first request on 2026-10-02 (D120). Every run line records the hash it
ran under. The spike's frozen pools and `18_oa_probe.json` carry `6f46de9a…`; the results files
regenerated after A1, and A1's pools, carry `b23f235e…`.

Purpose (D116): measure ASM-1 (each academic source is fast, complete and permissive enough
for a 3 s discovery search) and ASM-2 (how often a discovered paper has a fetchable
open-access PDF) **before** any M4 discovery code exists. Scripts under `spike/scripts/`
(`17`–`19`, `m4_*.py`). Every verdict and every pre-registered measure below is in
`spike/results/19_discovery_analysis.json`. The diagnoses (why anchors were missed, latency
split into headers and body, per-hop A2, PDF sizes, standby times) were recomputed from the
raw run logs under `var/spike/run/`, which stay local.

---

## What this does not establish

Read this before the numbers. The protocol pre-registered the first eight; the rest are what
this particular run adds.

- **Latency is one home connection, one machine and two time windows.** Not a production
  number, and not reproducible. At n=20, p95 is the second-worst value.
- **Relevance is a known-item check** on 20 well-known papers chosen in advance, biased toward
  highly cited work. It is a sanity bar, not a precision measurement; phase 8's judged set is.
- **Fetching a PDF's first 2 KiB is not ingestion.** The download sample is 30 PDFs and was
  checked with `validate_upload` and `preflight`, not parsed by GROBID.
- **Licence fields are mostly missing**, and nothing here settles what arXiv's terms permit.
  The spike only counts.
- **Sources change.** The pools are a snapshot as of 2026-10-01. The topics are ML/NLP only (T6).
- **Semantic Scholar was not measured.** Keyless S2 returned 429 to its first five requests
  in each session, 7 hours apart, and the pre-registered host-stop rule stopped it both
  times. The S2 result is "unusable keyless from this connection on this day", not a
  statement about S2's latency, completeness or relevance, and it says nothing about S2 with
  a key. Keyless behaviour depends on everyone else's traffic at that moment.
- **Session 1 ran in two processes.** It was stopped by hand after 15 requests for DEV-2 and
  resumed, so each source's session-1 latency set holds two connection-cold requests, not
  one; the warm-only figure removes only the first.
- **The five DEV-2 repeats ran alone**, not started together with the other two sources, and
  stand in for the requests OpenAlex refused.
- **The latency verdicts are for 50 results per page with abstracts and every location.**
  A smaller page or a slimmer field list is probably faster; that was not measured.
- **arXiv's relevance result is about the protocol's query construction as much as about
  arXiv.** The diagnosis of why anchors were missed is approximate: it checks words in the
  title and abstract, not arXiv's full field set or its stemming.
- **Unique works are linked by identifier only, as pre-registered.** 51 same-title pairs
  across sources were not linked, so some papers are counted twice in the ASM-2 denominator.
- **12 works were not probed** because the arxiv.org cap was reached. They count as neither
  success nor failure.
- **The machine went into Modern Standby twice during the run.** Kernel-Power events 506/507
  place it in standby from 05:55 to 06:00, during the download sample, and from 09:42 to
  12:10, which is why session 2 started 49 minutes after its earliest permitted time.
  Neither window overlaps session 1, the probe or session 2. The download sample reports
  bytes and checks, not timings.

---

## Verdicts

| Assumption | Verdict | Why |
|---|---|---|
| **ASM-1** — sources fast, complete and permissive enough | **FAIL** | The bar is two sources (one of them OpenAlex or S2) meeting all of 1a–1d. **None does.** OpenAlex misses only 1a; arXiv misses only 1d; S2 could not be measured keyless |
| **ASM-2** — share of discovered papers with a fetchable OA PDF | **PASS** | L2 under A3 = **151/180 = 83.9%** [77.8%, 88.5%] against a 60% bar |

Both verdicts are against bars fixed before the first call. The diagnoses below explain them;
they do not change them.

| Source | 1a latency | 1b completeness | 1c limits | 1d relevance | All four |
|---|---|---|---|---|---|
| OpenAlex (keyed) | ❌ p95 2.61 s: "usable only truncated" | ✅ | ✅ 0 × 429 in 41 | ✅ 70% in top 50 | ❌ |
| arXiv | ✅ p95 1.60 s | ✅ | ✅ 0 × 429/503 in 35 | ❌ 35% in top 50 | ❌ |
| S2 (keyless) | — no response | — | ❌ 10/10 × 429 | — | ❌ |

---

## 1. ASM-1, per source (session 1)

### 1a. Latency: request issued to last body byte, 2xx only

| | n | p50 | p90 | p95 | max | Done ≤ 2.0 s | Done ≤ 3.0 s |
|---|---|---|---|---|---|---|---|
| OpenAlex search | 20 | 1.94 s | 2.47 s | **2.61 s** | 2.88 s | 12/20 (60%) | 20/20 |
| arXiv search | 20 | 0.88 s | 1.32 s | 1.60 s | **16.34 s** | 19/20 | 19/20 |
| S2 search | 0 | — | — | — | — | 0/20 | 0/20 |
| All three for one query | 20 | — | — | — | — | 0/20 | 0/20 |
| OpenAlex single record | 10 | 0.60 s | 0.65 s | 0.66 s | 0.66 s | | |
| arXiv single record | 4 | 0.31 s | 0.36 s | 0.36 s | 0.36 s | | |

**OpenAlex's time is mostly the body, not the server.** Time to response headers is p50
0.71 s and p95 1.34 s; the median search body is **678 KB**: 50 works with inverted-index
abstracts and every location. Transfer accounts for more than half the total. That points
phase 2 at page size and `select` before anything else, but it is a hypothesis this run did
not test.

**arXiv's tail is one request.** `P1/t01-keyword` took 16.3 s, inside the 20 s ceiling; the
other 19 finished within 1.6 s.

**Session 2**, reported beside these numbers and never pooled with them: the 10
reversed-keyword queries, 7 hours later.

| | n | p50 | p90 | p95 | max | Done ≤ 2.0 s | Done ≤ 3.0 s |
|---|---|---|---|---|---|---|---|
| OpenAlex search | 10 | 1.98 s | 2.35 s | 2.37 s | 2.37 s | 6/10 | 10/10 |
| arXiv search | 10 | 1.03 s | 1.28 s | 1.51 s | 1.51 s | 10/10 | 10/10 |
| S2 search | 0 | — | — | — | — | 0/5 sent | 0/5 sent |

OpenAlex's median barely moved (1.94 s → 1.98 s), and 4 of its 10 session-2 searches still
took over 2 s. At n=10, p95 is the slowest value.

### 1b. Completeness (all session-1 records, 50 per query)

| Field | OpenAlex (n=1000) | arXiv (n=900) |
|---|---|---|
| title · authors · year | 100% · 99.9% · 100% | 100% · 100% · 100% |
| abstract | 92.1% | 100% |
| citation count | 100% | structural gap: arXiv carries none |
| venue | 71.7% | 6.9% (`journal_ref`) |
| DOI (not an arXiv DOI) | 82.5% | 8.6% |
| arXiv id | 35.0% | 100% |
| OA link | 93.9% | 100% |
| full date | 100% | 100% |

Both meet the bars (title, authors and year ≥ 95%; abstract ≥ 70%; citations ≥ 95% where
the source has them). arXiv returned 900 rather than 1000 records because some
natural-language queries matched fewer than 50 papers.

### 1c. Limits

- **OpenAlex, keyed:** 0 × 429 across 41 requests (31 in session 1, 10 in session 2). Its
  headers state the price:
  **$0.001 per search** (10 credits), $0.0001 for the filtered anchor-check list, **$0 for a
  single-record lookup**, and $0 for the five refused requests, against a **$1/day**
  allowance. Session 1 cost **$0.0201**. At this price $1 buys about 1,000 searches a day,
  which matches the documented figure D116 refused to trust without reading the headers.
- **arXiv:** 0 × 429 or 503 across 35 requests at the terms' pace (3 s from the previous
  response's end, one connection).
- **S2, keyless:** 429 on **10 of 10** [72%, 100%], five in each session. Within a session
  they were about four seconds apart. The body says "Too Many Requests… apply for a key".
  Five in a row is the host-stop trigger, so the rest were logged as not sent (host
  stopped): 16 requests in session 1, 5 in session 2.

### 1d. Relevance: the 20 anchors in each source's results

| | keyword top 10 | keyword top 50 | natural language top 50 |
|---|---|---|---|
| OpenAlex | 60% | **70%** [48%, 85%] | 55% |
| arXiv | 15% | **35%** [18%, 57%] | 25% |

All 20 anchor ids were confirmed against arXiv's own titles (0 mismatches), and arXiv matches
are by exact id. **arXiv's 35% is real, and most of it is our query construction.** The
protocol ANDs every content word (`all:knowledge AND all:distillation AND …`). Of the 13
anchors arXiv missed, judged by the words in each one's title and abstract:

- **7 lack a query word in any form**, so the AND excludes them whatever arXiv's stemming
  does. DistilBERT's abstract, for example, has no form of "compression".
- **2 carry a query word only in another form.** Hinton's paper says "distilling", not
  "distillation", and DDPM says "generative", not "generation". Whether the AND excluded
  these depends on arXiv's stemming, which was not measured.
- **4 contain every word but rank below 50.** arXiv's relevance has no popularity signal,
  and recent papers with the same words crowd out the canonical one.

OpenAlex's `relevance_score` does carry citation weighting (D116), which is what a
known-item check on famous papers rewards. The anchor check found 17 of the 20 anchors in
OpenAlex under their arXiv DOI; a miss there means "not under that DOI", not "absent".

---

## 2. ASM-2: the PDF funnel

192 unique works from session 1's keyword results (each source's top 10 per topic). 12 works
whose only link was an arxiv.org PDF went unprobed when the arxiv.org cap of 120 requests was
reached: 9 came from arXiv's results and 3 from OpenAlex records carrying an arXiv id. That
leaves **180** in the denominator.

| Level | Works | Share [95% CI] |
|---|---|---|
| L0: has at least one OA link | 170 | 94.4% [90.1%, 97.0%] |
| L1: a link usable over https (as given or upgraded) | 170 | 94.4% [90.1%, 97.0%] |
| **L2: first bytes `%PDF-` within 5 redirects, under A3 (any safe host)** | **151** | **83.9% [77.8%, 88.5%]** |
| L2 under A1 (arxiv.org only) | 119 | 66.1% [58.9%, 72.6%] |
| L2 under A2, as pre-registered (final hosts arxiv.org + aclanthology.org: the smallest set of final hosts covering ≥ 90% of L2) | 136 | 75.6% [68.8%, 81.3%] |
| L3 (download sample): passed `validate_upload` and `preflight` | 30 of 30 | 100% [88.6%, 100%] |

**A2 counts the host that served the PDF, as the protocol defines it, and that flatters it.**
2 of its 136 works reached aclanthology.org through a redirect from `www.aclweb.org`, and
neither had another link. An allowlist of exactly those two hosts, checked on every hop as
section 3 says it must be, fetches **134: 74.4% [67.6%, 80.3%]**. That is 88.7% of L2, below
the 90% A2 was defined to reach. A1 is unaffected: all 119 arxiv.org fetches had no redirects.

By source: works with an arXiv record fetched **90 of 91**; works with an OpenAlex record
fetched 65 of 93 (69.9%). The download sample's PDFs were p50 1.6 MB (max 20.0 MB) and p50
17 pages (max 581).

**Why 19 works failed.** There were 24 failed attempts in all. 22 belong to the 19 failed
works. The other 2 belong to works later fetched from another link: the nature.com redirect
loop (fetched from Kent's repository instead) and one AAAI legacy-site 403 (fetched from
ojs.aaai.org).

- **14 × 403**: ACM (4), MIT Press (2), Oxford University Press (2), and 1 each from PNAS,
  ACS, ScienceDirect, SSRN, AAAI's legacy site and Illinois' repository. The spike sent one
  script User-Agent and never tried another, so what triggered the refusals is not
  established.
- **6 HTML pages instead of a PDF.** 4 came from a URL OpenAlex gave as a `pdf_url`: PubMed
  Central (2), Penn's repository, and a Digital Commons page. The other 2 came from
  OpenAlex's `oa_url` fallback and were landing pages, never offered as PDFs: nature.com,
  reached through doi.org, and KU Leuven's repository.
- **1 redirect loop.** nature.com sent the PDF URL through the same cookieless
  `idp.nature.com` handshake that its 5 delivered PDFs passed through, at 3 redirects each.
  This time it answered with a 303 to the article's HTML page, which started the handshake
  again, and the chain went over the 5-redirect limit. The other five also had no cookies,
  so missing cookies alone do not explain it.
- **1 × 503** (nature.com), and **2 × 404**: IEEE Xplore, and one from arxiv.org. The arXiv
  API returned `2507.20128v2` at rank 5, and its own comment marks the entry as a duplicate
  submission superseded by `2603.00576`. The unversioned URL the protocol builds,
  `arxiv.org/pdf/2507.20128`, returned 404; the API's own versioned link was not tried.

**One fetched arXiv PDF is 97 MiB**, nearly twice `MAX_PDF_BYTES` (NFR-P8's 50 MiB), so
production would refuse it at upload validation. It is the only one of the 148 fetched PDFs
whose size is known (145 from the probe's Content-Range, 3 from the download sample) to exceed
the cap, and the next largest is 43 MiB. 3 ojs.aaai.org PDFs were served with no length, so
their size is unknown.

---

## 3. Evidence without a bar

**Allowlist** (D115's open question).

- Initial candidate hosts: arxiv.org 132, aclanthology.org 16, doi.org 8, nature.com 8, then
  a long tail of 27 hosts with 1–6 URLs each.
- Hosts that served a fetched PDF: arxiv.org 119, aclanthology.org 17, ojs.aaai.org 6,
  nature.com 5, link.springer.com 2, and one each from Stanford and Kent.
- Redirects before a fetched PDF: 0 for 142, 1 for 2, 3 for 6, 5 for 1.
- **An allowlist of final hosts is not enough.** 9 fetched PDFs came through chains with
  other hosts in the middle: `idp.nature.com` and `idp.springer.com` (cookie handshakes),
  `www.aclweb.org` and `bmcbioinformatics.biomedcentral.com`. An allowlist must cover each
  hop or those fetches fail.
- Only 2 of 209 candidate URLs were given as http; the one that was upgraded and sent fetched.

**Licence**, for the 151 fetched works:

- 86 have no OpenAlex record (arXiv-only, and arXiv's API reports no licence);
- 36 have an OpenAlex record with no licence reported;
- 26 are cc-by, 2 public domain, 1 cc-by-nc-nd.

S2 licences were unavailable. Phases 5–6 decide what may be stored and served with this
count in hand. It shows mostly missing licence data, not mostly permissive licences.

**Dedup** (per topic, all 50 ranks of session-1 keyword results).

- OpenAlex records carry an arXiv id 35% of the time and a DOI 82.8%; arXiv records carry a
  DOI 9%.
- **32** cross-source works were linked by identifier, 30 of them by arXiv id. **51**
  same-title pairs across sources were *not* linked.
- **Identifier-only matching misses more cross-source duplicates than it finds.** The typical
  miss is OpenAlex having merged the preprint into its published version: a publisher DOI and
  no arXiv location. The chain-of-thought paper is one example.
- Of linked works, 28 agree on the year and 5 differ by one.

Phase 2's dedup needs title (and probably year) matching as well as identifiers.

**Phrasing.** No source returned zero results for any query, in either phrasing. arXiv's
natural-language queries matched as few as 3 papers in total (mean 40 returned, against 50
for keyword); OpenAlex returned 50 for every query. Anchor recall falls from keyword to
natural language for both sources (OpenAlex 70% → 55%, arXiv 35% → 25%).

**Stability** (top-10 overlap, session-1 keyword vs session-2 reversed keyword, 7 hours
apart): OpenAlex **0.97** mean over 10 topics (lowest 0.9), arXiv **0.99** (lowest 0.9).
Neither source cares about word order, and neither reshuffled its top 10 within the day. A
discovery cache keyed on the normalised query would rarely serve stale results over hours.
Whether that holds over days was not measured.

---

## 4. Deviations and incidents

- **DEV-1 was not needed.** OpenAlex accepted `relevance_score` in `select`, and all 20
  session-1 search bodies carry it.
- **DEV-2 (D117).** Every natural-language phrasing ends in `?`, and OpenAlex's default
  search refuses `?` and `*` as wildcards ("Wildcards (* or ?) require exact (no-stem)
  search"). The seeded order sent five natural-language queries first, so all five OpenAlex
  searches came back 400, at $0, with no query run. The run was stopped by hand.
  - With the owner's approval, the protocol was amended: OpenAlex queries replace `?` and `*`
    with spaces.
  - The five refused searches were repeated once each after P1; all five returned 200.
  - Nothing else was repeated.
  - **P0 could not have caught this.** It is offline by design, and the failure was in how a
    live service reads a character. **It is a finding for phase 2:** a user who types a
    question would get an error from the OpenAlex adapter unless it sanitises these characters.
- **S2 was stopped in both sessions**, by the pre-registered rule, as described in 1c.
- **Session 2 started late**, at 12:12 rather than 11:23 UTC, because the machine was in
  Modern Standby. That is later than permitted, never earlier, so the 6-hour rule holds.
- **Five defects were fixed before any live call**, by a review of the spike scripts. Each was
  shown by an offline test that failed on the old code and passes on the new.
  1. Every refused URL would have crashed the probe: a parameter-name collision.
  2. A cap reached, or a host stopped, while requests queued behind that host would have
     crashed the run or sent requests it should not have.
  3. Two kinds of malformed third-party URL raised exceptions nothing caught.
  4. Download records would have written S2-only URLs to a tracked file.
  5. A restart forgot each host's stop and Retry-After state.

  Mutation testing in P0: **11 of 11 mutants caught**. Six break checker rules, two remove the
  pacer's admission re-checks, one disables the fetch guard's address check, and two turn off
  S2-URL hashing.

---

## 5. What this feeds (decisions not taken here)

Each of these is decided in the phase that implements it, with these numbers (D116).

| Decision | What the spike says |
|---|---|
| Which sources to build | OpenAlex met three of four bars, and its fourth looks like a page-size question. arXiv also met three; its fourth is mostly our AND query construction, and OR is worse (section 7), so the fix is not a different join. S2 needs a key decision: keyless was 10/10 × 429 across two windows, and a keyed pass would be its own addendum |
| OpenAlex query handling | Strip or escape `?` and `*`; the default stemmed search refuses them |
| Page size and `select` vs NFR-P1's 3 s | Transfer dominates OpenAlex latency; measure a smaller page through the real adapter before fixing a timeout |
| arXiv query construction | AND-of-every-word excludes canonical papers, and OR was measured and is worse (section 7, D120): AND stays. What remains untested is a construction between the two, such as AND over the rarest few words; that needs its own measurement |
| Allowlist (D115) | A1 66%, A2 76% (74% when enforced on every hop), A3 84% of works. Any allowlist must cover redirect hosts, not only the hosts serving the PDF |
| Dedup | Title (and probably year) matching on top of identifiers |
| Licence | Mostly unreported; 26 cc-by of 151 fetched |
| `MAX_PDF_BYTES` | One fetched arXiv PDF was 97 MiB, the only one over the cap among the 148 fetched PDFs whose size is known |
| Cost | OpenAlex costs $0.001 per search and $0 per single record, so a discovery cache protects a $1/day allowance of about 1,000 searches |

---

## 6. Acceptance (D116)

| # | Criterion | Status |
|---|---|---|
| 1 | Protocol approved before the first call; every results file records its hash | ✅ D116 approval before any call. The DEV-2 amendment was approved after 15 requests (D117), and the run logs record both hashes |
| 2 | P0 passed, including the checker mutant | ✅ before the first live call. 11/11 mutants caught |
| 3 | The log checker confirms budgets, pacing, address refusals and no automatic redirects | ✅ for D116's run: 354 requests, 0 violations. With A1's 40 the log holds 394, and the checker reports one 3 s gap it cannot verify, after the request cut off by hand (section 7) |
| 4 | Every planned request has a logged outcome | ✅ for D116's run: 0 planned requests missing, 0 sent without an outcome. A1 has one sent with no outcome, the same request (section 7) |
| 5 | ASM-1 verdict per source, with n and Wilson intervals | ✅ FAIL, above |
| 6 | ASM-2 verdict with the funnel under A1, A2 and A3 | ✅ PASS, above |
| 7 | Allowlist, dedup, licence, stability and phrasing evidence exists | ✅ section 3 |
| 8 | Pools reload with SHA-256 verified; no S2 content, PDF or key in any tracked file | ✅ `finalize` reloaded 81 entries with the hash verified. `leakcheck` is clean across 17 files, and a grep of the working tree outside `var/` and `spike/.env` finds no key value. **The S2 checks were vacuous:** S2 returned no data, so there were no S2 ids or URLs to leak |
| 9 | All five gates green, gate 4 printing "Contracts: 3 kept, 0 broken" | ✅ run on the final tree after session 2: format, ruff and mypy clean; `Contracts: 3 kept, 0 broken.`; `1218 passed`. The spike changes nothing under `src/` or `tests/` |
| 10 | "Does not establish" above the numbers; decisions listed, not taken | ✅ |

---

## 7. Addendum A1: arXiv's query construction (D120)

Run on 2026-10-02 after the spike, under a protocol addendum approved before its first
request: the 20 session-1 queries against arXiv only, each joined with OR and, as a
same-day control, with AND. **Pre-registered rule:** adopt OR only if, for the keyword
phrasing, its top-50 anchor recall is at least 50% and strictly higher than the same-day
AND's. Its responses are in `data/eval/discovery-pools-arxiv-a1.json.gz`; the spike's own
pools are unchanged.

**What A1 does not establish.**

- **Anchor recall is blind to precision.** It counts 20 famous papers found, not how many
  off-topic papers came with them. Phase 8's judged set measures that.
- **One alternative was tested.** A1 says OR is worse than AND, not that AND is the best
  construction available.
- **ASM-1 is not recomputed.** A construction chosen after the verdict cannot pass the bar
  retroactively, so ASM-1 stays FAIL.
- **arXiv was degraded for part of the run**, and 8 of the 40 requests failed, all before
  the final resume. Under the no-repeats rule they stay failures. The latency numbers mix
  that morning with a healthy window.

**Result: AND stays.**

| Keyword phrasing, n = 20 anchors | Same-day AND | OR |
|---|---|---|
| Queries answered | 9 of 10 | 9 of 10 |
| Anchors in the top 10 | 2 | 1 |
| **Anchors in the top 50** | **6 (30%)** [14.5%, 51.9%] | **4 (20%)** [8.1%, 41.6%] |
| Found only under this construction | 2 (LoRA, Mamba) | 0 |
| Matches arXiv reported per query | 271 – 2,307 | 228,612 – 1,956,075 |

OR met neither condition. Under OR, every anchor both constructions found ranked lower
(6 → 25, 12 → 21, 3 → 6, 25 → 34), and LoRA and Mamba dropped out of the top 50. The
match counts suggest why: OR over common words matches up to two million papers, and
arXiv's relevance ranking, which has no popularity signal, buries the canonical ones among
them.

**The natural-language phrasing is not a like-for-like comparison.** Each construction
answered only 7 of its 10 queries, and the failed sets differ (AND lost t08, t09 and t10; OR
lost t02, t08 and t10). Each put 2 of 20 anchors in its top 50. On the 6 topics both answered,
AND found 2 and OR 1, and DPR fell from rank 1 to 7 under OR. AND's losses include t09 and
t10, which held 3 of the spike's 5 natural-language anchors in its top 50.

**The same-day control reproduced the spike's anchor ranks exactly.** On every topic that
answered both days, from identical request URLs, AND ranked each anchor where the spike's AND
did (15, 6, 29, 12, 3, 25). arXiv's index had moved slightly between the days: its reported
totals rose by 1–6 on six of the nine topics, and four of the nine top-50 lists differ at one
to four positions, in t07 by one replaced paper. No anchor moved, so the comparison is not
confounded by the drift.

**The lost query cannot have changed the result.** t03's keyword query failed under both
constructions (a timeout under AND, a 429 under OR). Even crediting both of its anchors to
OR and neither to AND gives 6 against 6, which is not strictly greater.

**Incidents.** arXiv was degraded on the morning of 2026-10-02: 13–20 s latencies,
timeouts, and 429s at the 3 s pace, while its status page showed no incident. A1 was paused
twice and resumed at 06:26 UTC.

- **The first pause was by hand.** Killing the process cut `A1/t08-natural_language/OR` off
  mid-flight. It is logged as sent with no outcome, the one acceptance-4 exception in the
  whole spike. With no end time, the log checker also cannot verify the 3 s gap after it,
  the one acceptance-3 exception. Both were caused by the person running it.
- **The second pause was by a timeout watcher.** It let one more request through because
  the stop file was checked before the pacing wait instead of after it; that was corrected.
- **The final resume ran under a watcher fixed beforehand.** It paused on any timeout, 429,
  5xx, or success over 10 s, and judged only status and latency. A request the decision
  does not use answered in 1.3 s first. All 27 requests after the resume were healthy,
  slowest 8.1 s.
- **Nothing was repeated.** Over the whole run, OR's p50 latency was 2.1 s against AND's
  1.1 s.

---

## Reproduce

```bash
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py selftest   # P0, offline
.venv\Scripts\python.exe spike\scripts\17_discovery_search.py session1     # P1, DEV-2, P2
.venv\Scripts\python.exe spike\scripts\18_oa_probe.py probe                # P3
.venv\Scripts\python.exe spike\scripts\18_oa_probe.py download             # P4
.venv\Scripts\python.exe spike\scripts\17_discovery_search.py session2 --wait
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py check
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py analyse
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py finalize
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py leakcheck
.venv\Scripts\python.exe spike\scripts\17_discovery_search.py addendum-a1   # A1 (D120)
.venv\Scripts\python.exe spike\scripts\19_discovery_analysis.py finalize-a1
```

The live passes need `OPENALEX_API_KEY` in `spike/.env`, and none of them repeats a request
already in the run log. The run logs, S2 responses and PDFs stay under `var/spike/`
(gitignored); the tracked outputs are S2-free.
