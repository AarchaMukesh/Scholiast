"""M4 discovery spike: P0 checks, the log checker, P6 analysis, tracked outputs (D116).

    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py selftest   # P0, no network
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py check      # the real logs
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py analyse
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py finalize
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py finalize-a1
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py finalize-a2
    .venv\\Scripts\\python.exe spike\\scripts\\19_discovery_analysis.py leakcheck

`selftest` is P0: the address classifier against the protocol's table, URL checks,
normalisation, the pacer and the budget on a scaled clock, the fetch path end to
end against a mock transport (queued requests re-judged, every refusal path), and
the log checker against a synthetic log with one planted violation per rule. It
then mutates the checker one rule at a time, the pacer's admission re-checks, and
the fetch guard's address check, and requires a test to fail under every mutant.
For the arXiv spacing rule it must be the planted test that fails (D116, P0).

`check` runs the same checker over the real run logs. `analyse` computes every
number the findings quote. `finalize` writes the tracked outputs, S2-free, and the
frozen pools. `leakcheck` proves no key, no S2 identifier and no PDF is tracked.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gzip
import hashlib
import inspect
import json
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx
from m4_common import (
    A2_SIZES,
    MAX_REDIRECTS,
    POOLS_MANIFEST_PATH,
    POOLS_PATH,
    RESULTS_DIR,
    ROOT,
    RUN_DIR,
    Budget,
    HostHealth,
    Pacer,
    RunLog,
    Secrets,
    address_allowed,
    arxiv_from_url,
    arxiv_query,
    check_url,
    iso,
    load_env,
    load_protocol,
    nearest_rank,
    norm_arxiv,
    norm_title,
    openalex_query,
    probe_budget_keys,
    read_jsonl,
    s2_hash,
    s2_query,
    wall,
    wilson,
)
from m4_http import Chain, Ctx, build_ctx, fetch_chain
from m4_parse import Rec, parse
from m4_works import build_works, p1_keyword_records, record_keys, search_records

from scholiast.infrastructure.discovery.openalex import ADAPTER_VERSION, SELECT, OpenAlexSource

SOURCES = ("openalex", "semantic_scholar", "arxiv")
OUT_REQUESTS = RESULTS_DIR / "17_discovery_requests.json"
OUT_PROBE = RESULTS_DIR / "18_oa_probe.json"
OUT_ANALYSIS = RESULTS_DIR / "19_discovery_analysis.json"
FINDINGS = ROOT / "SPIKE-FINDINGS-M4.md"
POOLS_A1_PATH = ROOT / "data" / "eval" / "discovery-pools-arxiv-a1.json.gz"
POOLS_A1_MANIFEST_PATH = ROOT / "data" / "eval" / "discovery-pools-arxiv-a1.manifest.json"
POOLS_A2_PATH = ROOT / "data" / "eval" / "discovery-pools-openalex-a2.json.gz"
POOLS_A2_MANIFEST_PATH = ROOT / "data" / "eval" / "discovery-pools-openalex-a2.manifest.json"
CURVE_S = (1.0, 1.5, 2.0, 2.5, 3.0)


# ============================================================ the log checker


def merged(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every request that was sent, joined to its outcome by request id."""
    outcomes = {r["rid"]: r for r in records if r.get("event") == "outcome"}
    out = []
    for r in records:
        if r.get("event") != "sending":
            continue
        o = outcomes.get(r["rid"], {})
        out.append(
            {
                "rid": r["rid"],
                "host": r["host"],
                "t_start": r["t_start"],
                "t_end": o.get("t_end"),
                "budget_keys": r.get("budget_keys", []),
                "addresses": r.get("addresses"),
                "status": o.get("status"),
                "outcome": o.get("outcome"),
                "client_redirects": o.get("client_redirects", 0),
            }
        )
    return out


def decided(records: list[dict[str, Any]]) -> set[str]:
    return {r["rid"] for r in records if r.get("event") in ("sending", "not_sent")}


def unfinished(records: list[dict[str, Any]]) -> list[str]:
    """Requests logged as sent with no outcome: a crash mid-request (acceptance 4)."""
    ended = {r["rid"] for r in records if r.get("event") == "outcome"}
    return sorted(r["rid"] for r in records if r.get("event") == "sending" and r["rid"] not in ended)


def check_log(
    requests: list[dict[str, Any]],
    protocol: dict[str, Any],
    decided_rids: set[str],
    api_addresses: dict[str, list[str]],
) -> list[str]:
    """Every rule of acceptance 3, applied to sent requests. Returns violations."""
    violations: list[str] = []
    budget = Budget(protocol)
    used: Counter[tuple[str, ...]] = Counter()
    for req in requests:
        for key in req["budget_keys"]:
            used[tuple(key)] += 1
    for key, count in sorted(used.items()):
        if count > budget.cap(key):  # MUTATION-POINT: budget
            violations.append(f"budget: {'/'.join(key)} used {count} > cap {budget.cap(key)}")

    pacing = protocol["pacing"]
    by_host: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for req in requests:
        by_host[req["host"]].append(req)
    for host, reqs in by_host.items():
        reqs.sort(key=lambda r: r["t_start"])
        rule = pacing.get(host)  # MUTATION-POINT: arxiv-spacing
        if rule is None:
            rule = pacing["any_other_host"]
        gap = float(rule["min_gap_s"])
        from_end = rule.get("measured_from") == "previous response end"
        for prev, cur in zip(reqs, reqs[1:], strict=False):
            reference = prev["t_end"] if from_end else prev["t_start"]
            if reference is None:
                violations.append(f"pacing: {host} {prev['rid']} has no end time; gap unverifiable")
                continue
            if cur["t_start"] - reference < gap:
                violations.append(
                    f"pacing: {host} {cur['rid']} started {cur['t_start'] - reference:.3f}s after "
                    f"{prev['rid']} (minimum {gap}s from its {'end' if from_end else 'start'})"
                )
            if prev["t_end"] is not None and cur["t_start"] < prev["t_end"]:  # MUTATION-POINT: overlap
                violations.append(f"overlap: {host} {cur['rid']} started before {prev['rid']} ended")

    for req in requests:
        addresses = req["addresses"] if req["addresses"] is not None else api_addresses.get(req["host"])
        if not addresses:
            violations.append(f"address: {req['rid']} has no resolved-address record")
        elif not all(address_allowed(a) for a in addresses):  # MUTATION-POINT: address
            violations.append(f"address: {req['rid']} was sent to a non-global address {addresses}")
        if req["client_redirects"]:  # MUTATION-POINT: auto-redirect
            violations.append(f"redirect: {req['rid']} followed a redirect automatically")
        if req["outcome"] == "redirect":
            match = re.match(r"^(.*)/h(\d+)$", req["rid"])
            if match and int(match.group(2)) < MAX_REDIRECTS:
                following = f"{match.group(1)}/h{int(match.group(2)) + 1}"
                if following not in decided_rids:  # MUTATION-POINT: missing-hop
                    violations.append(f"redirect: {req['rid']} has no next hop recorded")
    return violations


# ============================================================ P0: selftest


def _synthetic() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """A clean log that every rule accepts."""
    reqs: list[dict[str, Any]] = []

    def add(rid: str, host: str, start: float, end: float, keys: list[list[str]],
            addresses: list[str] | None = None, outcome: str = "ok") -> None:
        reqs.append({"rid": rid, "host": host, "t_start": start, "t_end": end,
                     "budget_keys": keys, "addresses": addresses,
                     "status": 302 if outcome == "redirect" else 200,
                     "outcome": outcome, "client_redirects": 0})

    api = [["api", "arxiv_api", "search"]]
    add("A/1", "export.arxiv.org", 1000.0, 1001.0, api)
    add("A/2", "export.arxiv.org", 1004.5, 1005.5, api)
    probe = [["probe_host", "arxiv.org"], ["probe_total"]]
    add("P/x/u0/h0", "arxiv.org", 1000.0, 1000.5, probe, ["151.101.1.1"])
    add("P/y/u0/h0", "arxiv.org", 1015.5, 1016.0, probe, ["151.101.1.1"])
    other = [["probe_host", "example.org"], ["probe_total"]]
    add("P/z/u0/h0", "example.org", 1000.0, 1000.2, other, ["93.184.216.34"], "redirect")
    add("P/z/u0/h1", "example.org", 1005.5, 1005.7, other, ["93.184.216.34"])
    oa = [["api", "openalex", "search"]]
    add("O/1", "api.openalex.org", 1000.0, 1000.8, oa)
    add("O/2", "api.openalex.org", 1001.3, 1002.0, oa)
    return reqs, {"export.arxiv.org": ["128.84.21.199"], "api.openalex.org": ["104.26.0.1"]}


def _planted() -> dict[str, tuple[list[dict[str, Any]], set[str], dict[str, list[str]], str]]:
    """One synthetic log per rule, each breaking exactly that rule."""
    cases = {}

    def base() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
        reqs, api = _synthetic()
        return copy.deepcopy(reqs), api

    def by(reqs: list[dict[str, Any]], rid: str) -> dict[str, Any]:
        return next(r for r in reqs if r["rid"] == rid)

    # 2.9 s after the previous END but 5.9 s after its START: only the arXiv rule
    # (3 s from end) catches it; the any-other-host fallback (5 s from start) does not.
    reqs, api = base()
    by(reqs, "A/1")["t_end"] = 1003.0
    by(reqs, "A/2").update(t_start=1005.9, t_end=1006.5)
    cases["arxiv-spacing"] = (reqs, {r["rid"] for r in reqs}, api, "pacing: export.arxiv.org")

    reqs, api = base()
    by(reqs, "P/y/u0/h0").update(t_start=1014.0, t_end=1014.5)
    cases["arxiv-org-spacing"] = (reqs, {r["rid"] for r in reqs}, api, "pacing: arxiv.org")

    reqs, api = base()
    by(reqs, "P/z/u0/h1").update(t_start=1004.0, t_end=1004.2)
    cases["other-host-spacing"] = (reqs, {r["rid"] for r in reqs}, api, "pacing: example.org")

    reqs, api = base()
    for n in range(40):
        start = 1003.0 + n * 1.3
        reqs.append({"rid": f"O/x{n}", "host": "api.openalex.org", "t_start": start,
                     "t_end": start + 0.5, "budget_keys": [["api", "openalex", "search"]],
                     "addresses": None, "status": 200, "outcome": "ok", "client_redirects": 0})
    cases["budget"] = (reqs, {r["rid"] for r in reqs}, api, "budget: api/openalex/search")

    reqs, api = base()
    by(reqs, "P/x/u0/h0")["addresses"] = ["10.0.0.5"]
    cases["address"] = (reqs, {r["rid"] for r in reqs}, api, "address:")

    reqs, api = base()
    by(reqs, "O/1")["client_redirects"] = 1
    cases["auto-redirect"] = (reqs, {r["rid"] for r in reqs}, api, "redirect: O/1 followed")

    reqs, api = base()
    reqs = [r for r in reqs if r["rid"] != "P/z/u0/h1"]
    cases["missing-hop"] = (reqs, {r["rid"] for r in reqs}, api, "redirect: P/z/u0/h0 has no next hop")

    # A 20 s download on arxiv.org, the next request 15.5 s after its START: pacing
    # (15 s from start) is satisfied, so only the overlap rule can catch it.
    reqs, api = base()
    by(reqs, "P/x/u0/h0")["t_end"] = 1020.0
    cases["overlap"] = (reqs, {r["rid"] for r in reqs}, api, "overlap: arxiv.org")
    return cases


MUTANTS = {
    "arxiv-spacing": 'rule = None if host == "export.arxiv.org" else pacing.get(host)',
    "budget": "if False:",
    "overlap": "if False:",
    "address": "elif False:",
    "auto-redirect": "if False:",
    "missing-hop": "if False:",
}
#: A2's rule (D125), broken the ways a careless edit would break it.
A2_MUTANTS = {
    "a2-deadline": "fits = p95 is not None and p95 < A2_DEADLINE_S",
    "a2-anchors": "if fits and 2 * row[\"keyword_anchors\"] > n_anchors:",
    "a2-failures": 'all_s = [o["latency_s"] if o is not None else float("inf")',
}


def _mutated(target: Any, name: str, replacement: str) -> Any:
    """`target` (a function or class) with the line at one MUTATION-POINT replaced."""
    source = textwrap.dedent(inspect.getsource(target))
    lines = source.splitlines()
    marker = f"# MUTATION-POINT: {name}"
    hits = [i for i, line in enumerate(lines) if marker in line]
    if len(hits) != 1:
        raise RuntimeError(f"mutation point {name!r} found {len(hits)} times")
    i = hits[0]
    indent = lines[i][: len(lines[i]) - len(lines[i].lstrip())]
    lines[i] = indent + replacement
    namespace = dict(vars(sys.modules[target.__module__]))
    exec("\n".join(lines), namespace)  # noqa: S102 - a mutant of the spike's own code
    return namespace[target.__name__]


def _mutant(name: str) -> Any:
    """`check_log` with one rule broken the way a careless refactor would break it."""
    return _mutated(check_log, name, MUTANTS[name])


def _scaled_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    scaled = copy.deepcopy(protocol)
    scaled["pacing"]["export.arxiv.org"]["min_gap_s"] = 0.3
    scaled["pacing"]["arxiv.org"]["min_gap_s"] = 0.4
    return scaled


async def _pacer_test(protocol: dict[str, Any]) -> list[str]:
    """The pacer honours from-end and from-start gaps and Retry-After, on a scaled clock."""
    failures = []
    pacer = Pacer(_scaled_protocol(protocol))
    spans: dict[str, list[tuple[float, float]]] = defaultdict(list)

    async def use(host: str, work: float) -> None:
        async with pacer.slot(host) as start:
            await asyncio.sleep(work)
        spans[host].append((start, time.time()))

    await asyncio.gather(*(use("export.arxiv.org", 0.2) for _ in range(3)),
                         *(use("arxiv.org", 0.05) for _ in range(3)))
    for (_s0, e0), (s1, _) in zip(spans["export.arxiv.org"], spans["export.arxiv.org"][1:], strict=False):
        if s1 - e0 < 0.3:
            failures.append(f"pacer: arXiv API gap from end {s1 - e0:.3f}s < 0.3s")
        if s1 < e0:
            failures.append("pacer: arXiv API requests overlapped")
    for (s0, _), (s1, _) in zip(spans["arxiv.org"], spans["arxiv.org"][1:], strict=False):
        if s1 - s0 < 0.4:
            failures.append(f"pacer: arxiv.org gap from start {s1 - s0:.3f}s < 0.4s")
    pacer.defer("example.org", 0.5)
    before = time.time()
    async with pacer.slot("example.org"):
        pass
    if time.time() - before < 0.45:
        failures.append("pacer: Retry-After deferral not honoured")
    return failures


#: The pacer's admission re-checks, each removed in turn (P0, run beside MUTANTS).
PACER_MUTANTS = {"admit-before-wait": "pass", "admit-after-wait": "pass"}
#: The fetch guard's own address check: the SSRF defence itself, not the checker of it.
FETCH_MUTANTS = {"guard-address": "if False:"}


async def _admission_test(
    protocol: dict[str, Any], pacer_cls: Any = Pacer, fetch: Any = fetch_chain
) -> dict[str, Any]:
    """Requests queued behind a host are judged again when their turn comes.

    Offline: every host is an IP literal, which resolves without DNS, and a mock
    transport answers. Each case is a failure the one-shot run must not have: a cap
    reached, or a host stopped, while a request waited its turn gives 'not sent'
    (never a crash, never a request); a URL that httpx or the resolver rejects gives
    an outcome (never a crash); and every request logged as sent has an outcome.
    """
    scaled = copy.deepcopy(protocol)
    scaled["pacing"]["any_other_host"]["min_gap_s"] = 0.05
    reached: list[httpx.URL] = []

    async def answer(request: httpx.Request) -> httpx.Response:
        reached.append(request.url)
        await asyncio.sleep(0.01)
        if request.url.host == "93.184.216.35":
            return httpx.Response(429, headers={"Retry-After": "1"})
        if request.url.host == "93.184.216.39":
            return httpx.Response(302, headers={"Location": "https://127.0.0.1/x.pdf"})
        return httpx.Response(206, content=b"%PDF-1.7\n" + b"0" * 4096)

    async def run(name: str, urls: list[str], per_host: int = 25, total: int = 300,
                  seed_host: str | None = None) -> tuple[list[str], list[dict[str, Any]]]:
        p = copy.deepcopy(scaled)
        p["budget"]["other_pdf_hosts"]["max"]["probe_requests_per_host"] = per_host
        p["budget"]["pdf_probe_requests_total_max"] = total
        with tempfile.TemporaryDirectory() as tmp:
            log = RunLog(name, Secrets([]), directory=Path(tmp))
            ctx = Ctx(p, pacer_cls(p), Budget(p), HostHealth(), log, set())
            if seed_host:
                ctx.pacer.seed(seed_host, wall(), wall())
            transport = httpx.MockTransport(answer)
            async with httpx.AsyncClient(transport=transport, follow_redirects=False) as client:
                got = await asyncio.gather(
                    *(fetch(ctx, client, rid_prefix=f"T/{name}/{i}", url=u, mode="probe",
                            budget_keys_for=probe_budget_keys, meta={})
                      for i, u in enumerate(urls)),
                    return_exceptions=True,
                )
            records = log.records()
        return [c.outcome if isinstance(c, Chain) else f"raised {type(c).__name__}" for c in got], records

    def sent(records: list[dict[str, Any]]) -> int:
        return sum(1 for r in records if r.get("event") == "sending")

    failures: list[str] = []
    cases: dict[str, Any] = {}

    # A per-host cap reached while two requests queue for the host.
    got, records = await run("cap", ["https://93.184.216.34/a.pdf"] * 3, per_host=2)
    cases["per_host_cap_while_queued"] = {"outcomes": got, "sent": sent(records)}
    if sorted(got) != ["fetched", "fetched", "not_sent_budget"] or sent(records) != 2:
        failures.append(f"admission: per-host cap reached while queued gave {got}")
    unfinished_all = unfinished(records)

    # A host stopped (5 consecutive 429s, each with Retry-After: 1) while two more queue:
    # they are refused, and refused at once rather than after waiting out the Retry-After.
    got, records = await run("stop", ["https://93.184.216.35/a.pdf"] * 7)
    last_end = max((r["t_end"] for r in records if r.get("event") == "outcome"), default=0.0)
    refused_after = [datetime.fromisoformat(r["at"]).timestamp() - last_end
                     for r in records if r.get("event") == "not_sent"]
    cases["host_stopped_while_queued"] = {"outcomes": got, "sent": sent(records),
                                          "refused_after_s": [round(x, 3) for x in refused_after]}
    if sorted(got) != ["not_sent_host_stopped"] * 2 + ["throttled"] * 5 or sent(records) != 5:
        failures.append(f"admission: host stopped while queued gave {got}")
    elif max(refused_after) > 0.5:
        failures.append(f"admission: a stopped host's queued requests waited {max(refused_after):.2f}s "
                        "for its Retry-After before being refused")
    unfinished_all += unfinished(records)

    # The shared probe cap filled by other hosts while one request slept out its gap.
    urls = ["https://93.184.216.36/a.pdf", "https://93.184.216.37/a.pdf", "https://93.184.216.38/a.pdf"]
    got, records = await run("total", urls, total=2, seed_host="93.184.216.36")
    cases["total_cap_while_waiting"] = {"outcomes": got, "sent": sent(records)}
    if got != ["not_sent_budget", "fetched", "fetched"] or sent(records) != 2:
        failures.append(f"admission: shared cap filled while waiting gave {got}")
    unfinished_all += unfinished(records)

    # Every refusal path, end to end: each must give an outcome and send nothing it
    # refuses. A NUL that httpx rejects, an empty label the resolver rejects, a
    # scheme, a private address, a redirect to loopback, and an http URL upgraded.
    urls = ["https://93.184.216.34/a\x00b.pdf", "https://a..b.com/x.pdf", "ftp://93.184.216.34/a.pdf",
            "https://10.0.0.1/a.pdf", "https://93.184.216.39/moved.pdf", "http://93.184.216.40/a.pdf"]
    got, records = await run("refusals", urls)
    reasons = [r.get("reason") for r in records if r.get("event") == "not_sent"]
    cases["refusals"] = {"outcomes": got, "sent": sent(records), "reasons": reasons}
    if got != ["refused", "dns_error", "refused", "refused", "refused", "fetched"] or sent(records) != 2:
        failures.append(f"refusals: gave {got} with {sent(records)} sent")
    elif not {"malformed:InvalidURL", "scheme:ftp", "address"} <= set(reasons):
        failures.append(f"refusals: reasons {reasons}")
    unfinished_all += unfinished(records)

    bad = [str(u) for u in reached if u.scheme != "https" or not address_allowed(u.host)]
    cases["transport_saw"] = {"requests": len(reached), "non_https_or_non_global": bad}
    if bad:
        failures.append(f"containment: the transport received {bad}")
    if unfinished_all:
        failures.append(f"admission: sent with no outcome logged: {unfinished_all}")
    return {"cases": cases, "failures": failures}


#: The S2-only decision in the tracked probe document, forced off in turn.
DOCUMENT_MUTANTS = {"probe-s2-only": "only = False", "download-s2-only": "only = False"}


def _resume_test(protocol: dict[str, Any]) -> list[str]:
    """A rerun after a crash obeys what the crashed run had learnt about each host.

    A host stopped in session 1 stays stopped when session 1 resumes, and not in
    session 2; a Retry-After still in force is still waited out; a request logged
    as sent with no end holds back the host's next request from now.
    """
    failures = []
    now = wall()
    records: list[dict[str, Any]] = []
    for n in range(5):
        t = now - 60 + n
        records += [{"event": "sending", "rid": f"S/{n}", "host": "api.semanticscholar.org", "t_start": t,
                     "budget_keys": [["api", "semantic_scholar", "search"]], "session": 1},
                    {"event": "outcome", "rid": f"S/{n}", "host": "api.semanticscholar.org", "t_start": t,
                     "t_end": t + 0.5, "status": 429, "headers": {}, "session": 1}]
    records += [{"event": "sending", "rid": "O/1", "host": "api.openalex.org", "t_start": now - 1,
                 "budget_keys": [["api", "openalex", "search"]], "session": 1},
                {"event": "outcome", "rid": "O/1", "host": "api.openalex.org", "t_start": now - 1,
                 "t_end": now - 0.5, "status": 429, "headers": {"retry-after": "120"}, "session": 1},
                {"event": "sending", "rid": "A/1", "host": "export.arxiv.org", "t_start": now - 30,
                 "budget_keys": [["api", "arxiv_api", "search"]], "session": 1}]
    with tempfile.TemporaryDirectory() as tmp:
        log = RunLog("resume", Secrets([]), directory=Path(tmp))
        for r in records:
            log.append(r)
        same = build_ctx(protocol, log, session=1)
        other = build_ctx(protocol, log, session=2)
    if "api.semanticscholar.org" not in same.health.stopped:
        failures.append("resume: a host stopped in session 1 is not stopped when session 1 resumes")
    if "api.semanticscholar.org" in other.health.stopped:
        failures.append("resume: a host stopped in session 1 is stopped in session 2")
    if same.pacer.earliest("api.openalex.org") < now + 119:
        failures.append("resume: a Retry-After still in force was forgotten")
    if same.pacer.earliest("export.arxiv.org") < now + 3.0:
        failures.append("resume: a request with no logged end did not hold back the next one")
    if same.budget.used[("api", "semantic_scholar", "search")] != 5:
        failures.append("resume: budget usage not replayed")
    return failures


def _containment_test(document: Any = None) -> list[str]:
    """A work fetched and downloaded through a URL only S2 supplied leaves no trace of it.

    The URL and the redirect target its chain met must be absent from the tracked
    probe document, in the probe attempts and in the downloads; and leakcheck's
    needle search must find the URL when it is planted in a JSON text.
    """
    document = document or probe_document
    s2_url, cdn_url = "https://pub.example/oa.pdf", "https://cdn.example/f/oa.pdf"

    def hops(prefix: str) -> list[dict[str, Any]]:
        return [{"rid": f"{prefix}/h0", "hop": 0, "url": s2_url, "sent": True, "host": "pub.example",
                 "status": 302, "location": cdn_url},
                {"rid": f"{prefix}/h1", "hop": 1, "url": cdn_url, "sent": True, "host": "cdn.example",
                 "status": 206, "location": None}]

    chain = {"outcome": "fetched", "hosts": ["pub.example", "cdn.example"], "final_host": "cdn.example",
             "upgraded": False}
    work = {"work": "doi:10.1/x", "topics": ["t01"], "sources": ["semantic_scholar"], "result": "fetched",
            "candidates": [{"url": s2_url, "kind": "s2_pdf", "supplied_by": ["semantic_scholar"],
                            "licences": {"semantic_scholar": "CC-BY"}, "url_check": "ok"}],
            "attempts": [{"url_index": 0, **chain, "hops": hops("P3/doi:10.1/x/u0")}]}
    download = {"work": "doi:10.1/x", "lane": "other", "url_index": 0, **chain, "outcome": "downloaded",
                "hops": hops("P4/doi:10.1/x"), "bytes": 1, "sha256": "0" * 64, "check": "ok"}
    run = {"api": [], "bodies": [], "pdf": [], "probe_works": [work], "downloads": [download]}

    failures = []
    needles = s2_only_urls(run)
    if needles != {s2_url, cdn_url}:
        failures.append(f"containment: S2-only needles {sorted(needles)}")
    text = json.dumps(document(run, "digest", set(), {}), ensure_ascii=False)
    hits = leaked_urls(text, {s2_url, cdn_url}, quoted=True)
    if hits:
        failures.append(f"containment: the tracked probe document carries S2-only URL(s) {hits}")
    if leaked_urls(json.dumps({"url": s2_url}), needles, quoted=True) != [s2_url]:
        failures.append("containment: leakcheck does not find a planted S2-only URL")
    return failures


def _a2_tests(protocol: dict[str, Any], choice: Any = None, analysis: Any = None) -> list[str]:
    """A2 (D125): the adapter builds the protocol's URLs, and the rule chooses what D125 says."""
    choice = choice or a2_choice
    analysis = analysis or addendum_a2
    failures = []
    spec = protocol["sources"]["openalex"]["search"]
    phrased = [(t, ph) for t in protocol["topics"] for ph in ("keyword", "natural_language")]
    for t, ph in phrased:
        for size in A2_SIZES:
            want = httpx.URL(spec["url"], params={**spec["params"], "search": openalex_query(t[ph]),
                                                  "per_page": size})
            got = OpenAlexSource("selftest").request(t[ph], limit=size).url
            if got != want:
                failures.append(f"A2: the adapter's URL for {t['id']}-{ph} at {size} is not the protocol's")

    def row(p95: float | None, anchors: int) -> dict[str, Any]:
        return {"p95_s": p95, "keyword_anchors": anchors}

    cases = [
        ("the deadline is inclusive", {50: row(2.3, 14), 25: row(2.0, 12), 15: row(1.4, 12)}, 25),
        ("the largest passing size wins", {50: row(1.9, 14), 25: row(1.5, 12), 15: row(1.2, 12)}, 50),
        ("a failure at the 95th rank fails the size", {50: row(None, 14), 25: row(None, 12), 15: row(1.6, 12)}, 15),
        ("half the anchors is enough", {50: row(2.4, 14), 25: row(1.8, 10), 15: row(1.5, 10)}, 25),
        ("under half is not", {50: row(2.4, 14), 25: row(1.8, 9), 15: row(1.5, 9)}, None),
    ]
    for name, per_size, want in cases:
        if (got_size := choice(per_size, 20)) != want:
            failures.append(f"A2 rule: {name}: chose {got_size}, want {want}")
    # Two failed requests of 20 put a failure at the 19th rank, which is nearest-rank p95.
    outcomes = [{"rid": f"A2/{t['id']}-{ph}/25", "pass": "A2", "outcome": "timeout" if i < 2 else "ok",
                 "latency_s": 1.0, "headers_s": 0.5, "bytes": 1000, "headers": {}}
                for i, (t, ph) in enumerate(phrased)]
    result = analysis(protocol, [], outcomes)
    if result is None or result["per_size"]["25"]["p95_s"] is not None:
        failures.append("A2 counts a failed request as finishing in time (D125)")
    return failures


def selftest() -> int:
    protocol, digest = load_protocol()
    results: dict[str, Any] = {"protocol_sha256": digest, "at": iso(wall())}
    failures: list[str] = []

    table = protocol["http_safety"]["self_test"]
    wrong = [a for a in table["must_refuse"] if address_allowed(a)] + [
        a for a in table["must_accept"] if not address_allowed(a)
    ]
    failures += [f"address classifier wrong for {a}" for a in wrong]
    results["address_table"] = {"refuse": len(table["must_refuse"]),
                                "accept": len(table["must_accept"]), "wrong": wrong}

    url_cases = [
        ("http://arxiv.org/pdf/2005.11401", True, True, None),
        ("https://arxiv.org/pdf/2005.11401", True, False, None),
        ("https://example.org:8443/a.pdf", False, False, "port:8443"),
        ("ftp://example.org/a.pdf", False, False, "scheme:ftp"),
        ("https://user:pw@example.org/a.pdf", False, False, "userinfo"),
        ("javascript:alert(1)", False, False, "scheme:javascript"),
        ("https:///nohost", False, False, "no-host"),
    ]
    for url, ok, upgraded, reason in url_cases:
        got = check_url(url)
        if (got.ok, got.upgraded if ok else upgraded, got.reason) != (ok, upgraded, reason):
            failures.append(f"check_url({url!r}) = {got}")
    results["url_cases"] = len(url_cases)

    norm_cases = {
        "http://arxiv.org/abs/2005.11401v4": "2005.11401",
        "https://arxiv.org/pdf/2106.09685v2.pdf": "2106.09685",
        "10.48550/arXiv.2201.11903": "2201.11903",
        "arXiv:1503.02531": "1503.02531",
        "hep-th/9901001v3": "hep-th/9901001",
        "not an id": None,
    }
    for raw, want in norm_cases.items():
        if norm_arxiv(raw) != want:
            failures.append(f"norm_arxiv({raw!r}) = {norm_arxiv(raw)!r}, want {want!r}")
    if arxiv_from_url("https://arxiv.org/abs/2305.14314") != "2305.14314":
        failures.append("arxiv_from_url failed on an abs URL")
    stop = protocol["query_construction"]["arxiv_stopwords"]
    expected_q = ("all:chain AND all:thought AND all:prompting AND all:improve AND "
                  "all:reasoning AND all:language AND all:models")
    t10 = next(t for t in protocol["topics"] if t["id"] == "t10")
    if arxiv_query(t10["natural_language"], stop) != expected_q:
        failures.append("arxiv_query differs from the protocol's worked example")
    if s2_query("Low-Rank  adaptation") != "low rank adaptation":
        failures.append("s2_query does not replace hyphens")
    if openalex_query("How do diffusion models generate images?") != "how do diffusion models generate images":
        failures.append("openalex_query keeps a wildcard character (DEV-2)")
    if openalex_query("a*b ? c") != "a b c":
        failures.append("openalex_query does not replace '*' and '?' with spaces (DEV-2)")
    if arxiv_query(t10["natural_language"], stop, "OR") != expected_q.replace(" AND ", " OR "):
        failures.append("arxiv_query's OR join differs from the AND construction in more than the join (A1)")
    if norm_title("SimCSE: Simple Contrastive") != "simcse simple contrastive":
        failures.append("norm_title")
    results["normalisation_cases"] = len(norm_cases) + 7

    budget = Budget(protocol)
    keys = [("api", "openalex", "search")]
    for _ in range(40):
        budget.take(keys)
    if budget.can(keys):
        failures.append("budget allows a 41st OpenAlex search")
    try:
        budget.take(keys)
        failures.append("budget.take did not refuse the 41st OpenAlex search")
    except RuntimeError:
        pass
    a1_budget = Budget(protocol)
    a1_keys = [("api", "arxiv_api", "a1")]
    for _ in range(protocol["addenda"]["A1"]["budget"]["max"]["queries"]):
        a1_budget.take(a1_keys)
    if a1_budget.can(a1_keys):
        failures.append("budget allows a 41st A1 request (D120)")
    if not a1_budget.can([("api", "arxiv_api", "search")]):
        failures.append("A1's budget spends the P1-P5 arXiv budget (D120)")
    a2_budget = Budget(protocol)
    a2_keys = [("api", "openalex", "a2")]
    for _ in range(protocol["addenda"]["A2"]["budget"]["max"]["searches"]):
        a2_budget.take(a2_keys)
    if a2_budget.can(a2_keys):
        failures.append("budget allows a 61st A2 request (D125)")
    if not a2_budget.can([("api", "openalex", "search")]):
        failures.append("A2's budget spends the P1-P5 OpenAlex budget (D125)")
    a2 = _a2_tests(protocol)
    failures += a2
    results["addendum_a2"] = {"failures": a2}
    replayed = Budget(protocol)
    replayed.replay([{"event": "sending", "budget_keys": [["probe_host", "arxiv.org"], ["probe_total"]]}] * 120)
    if replayed.can([("probe_host", "arxiv.org"), ("probe_total",)]):
        failures.append("budget replay does not restore arxiv.org usage")

    failures += asyncio.run(_pacer_test(protocol))
    admission = asyncio.run(_admission_test(protocol))
    failures += admission["failures"]
    results["admission"] = admission["cases"]
    containment = _containment_test()
    failures += containment
    results["containment"] = {"failures": containment}
    resume = _resume_test(protocol)
    failures += resume
    results["resume"] = {"failures": resume}

    clean, api = _synthetic()
    clean_violations = check_log(copy.deepcopy(clean), protocol, {r["rid"] for r in clean}, api)
    failures += [f"clean synthetic log flagged: {v}" for v in clean_violations]
    planted = _planted()
    planted_results = {}
    for name, (reqs, decided_rids, api_addr, expect) in planted.items():
        found = check_log(copy.deepcopy(reqs), protocol, decided_rids, api_addr)
        hit = any(v.startswith(expect) for v in found)
        planted_results[name] = hit
        if not hit:
            failures.append(f"planted {name} not flagged; got {found}")
    results["planted"] = planted_results

    # A mutant is caught when any test fails under it. The protocol additionally
    # requires, for the arXiv spacing rule, that it is the PLANTED test that fails
    # (D116, P0), so that is checked separately rather than implied.
    mutants = {}
    for name in MUTANTS:
        mutant = _mutant(name)
        reqs, decided_rids, api_addr, expect = planted[name]
        planted_fails = not any(
            v.startswith(expect) for v in mutant(copy.deepcopy(reqs), protocol, decided_rids, api_addr)
        )
        clean_fails = bool(mutant(copy.deepcopy(clean), protocol, {r["rid"] for r in clean}, api))
        caught_by = [t for t, failed in (("planted", planted_fails), ("clean", clean_fails)) if failed]
        mutants[name] = {"caught": bool(caught_by), "caught_by": caught_by}
        if not caught_by:
            failures.append(f"mutant {name} survived")
        if name == "arxiv-spacing" and not planted_fails:
            failures.append("D116: the arXiv spacing mutant did not fail the planted arXiv test")
    for name, replacement in PACER_MUTANTS.items():
        under = asyncio.run(_admission_test(protocol, _mutated(Pacer, name, replacement)))
        mutants[f"pacer:{name}"] = {"caught": bool(under["failures"]), "caught_by": under["failures"]}
        if not under["failures"]:
            failures.append(f"mutant pacer:{name} survived")
    for name, replacement in FETCH_MUTANTS.items():
        under = asyncio.run(_admission_test(protocol, fetch=_mutated(fetch_chain, name, replacement)))
        mutants[f"fetch:{name}"] = {"caught": bool(under["failures"]), "caught_by": under["failures"]}
        if not under["failures"]:
            failures.append(f"mutant fetch:{name} survived")
    for name, replacement in A2_MUTANTS.items():
        target = addendum_a2 if name == "a2-failures" else a2_choice
        mutated = _mutated(target, name, replacement)
        caught_by = (_a2_tests(protocol, analysis=mutated) if name == "a2-failures"
                     else _a2_tests(protocol, choice=mutated))
        mutants[name] = {"caught": bool(caught_by), "caught_by": caught_by}
        if not caught_by:
            failures.append(f"mutant {name} survived")
    for name, replacement in DOCUMENT_MUTANTS.items():
        caught_by = _containment_test(_mutated(probe_document, name, replacement))
        mutants[f"document:{name}"] = {"caught": bool(caught_by), "caught_by": caught_by}
        if not caught_by:
            failures.append(f"mutant document:{name} survived")
    results["mutants"] = mutants
    results["failures"] = failures
    results["passed"] = not failures

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "19_selftest.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k != "failures"}, indent=1))
    for f in failures:
        print("FAIL:", f)
    print("P0 selftest:", "PASSED" if not failures else f"FAILED ({len(failures)})")
    return 0 if not failures else 1


# ============================================================ loading the run


def load_run() -> dict[str, list[dict[str, Any]]]:
    return {name: read_jsonl(RUN_DIR / f"{name}.jsonl")
            for name in ("api", "bodies", "pdf", "probe_works", "downloads")}


def api_addresses(api: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for r in api:
        if r.get("event") == "preflight" and r.get("allowed"):
            out[r["host"]].update(r["addresses"])
    return {h: sorted(a) for h, a in out.items()}


def check() -> int:
    protocol, _ = load_protocol()
    run = load_run()
    records = run["api"] + run["pdf"]
    violations = check_log(merged(records), protocol, decided(records), api_addresses(run["api"]))
    for v in violations:
        print("VIOLATION:", v)
    open_rids = unfinished(records)
    for rid in open_rids:
        print("NO OUTCOME:", rid)
    sent = len(merged(records))
    print(f"log check: {sent} sent requests, {len(violations)} violations, "
          f"{len(open_rids)} sent with no outcome")
    return 0 if not violations and not open_rids else 1


# ============================================================ P6: analysis


REPEAT_SUFFIXES = ("/DEV-1", "/DEV-2")


def _outcomes(api: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """API outcomes, with a DEV-1 or DEV-2 repeat standing in for the request it repeated."""
    rows = [r for r in api if r.get("event") == "outcome"]
    replaced = {r["rid"].rsplit("/", 1)[0] for r in rows if r["rid"].endswith(REPEAT_SUFFIXES)}
    kept = []
    for r in rows:
        if r["rid"] in replaced:
            continue
        if r["rid"].endswith(REPEAT_SUFFIXES):
            r = {**r, "rid": r["rid"].rsplit("/", 1)[0], "stand_in": r["rid"].rsplit("/", 1)[1]}
        kept.append(r)
    return kept


def deviations(api: list[dict[str, Any]]) -> dict[str, Any]:
    """Each deviation, what it replaced, and what the repeat returned (D116 DEV-1, D117 DEV-2)."""
    rows = {r["rid"]: r for r in api if r.get("event") == "outcome"}
    out: dict[str, Any] = {"events": [r for r in api if r.get("event") == "deviation"]}
    for suffix in REPEAT_SUFFIXES:
        name = suffix[1:]
        pairs = [(rid[: -len(suffix)], rid) for rid in rows if rid.endswith(suffix)]
        out[name] = [{"rid": orig, "refused": {"status": rows.get(orig, {}).get("status"),
                                               "outcome": rows.get(orig, {}).get("outcome")},
                      "repeat": {"status": rows[rep].get("status"), "outcome": rows[rep].get("outcome"),
                                 "latency_s": rows[rep].get("latency_s")}}
                     for orig, rep in sorted(pairs)]
    return out


def _pct(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "p50": nearest_rank(values, 50),
        "p90": nearest_rank(values, 90),
        "p95": nearest_rank(values, 95),
        "max": max(values) if values else None,
    }


def latency(outcomes: list[dict[str, Any]], pass_id: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    per_query: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for source in SOURCES:
        reqs = sorted((o for o in outcomes if o.get("pass") == pass_id and o.get("kind") == "search"
                       and o.get("source") == source), key=lambda o: o["t_start"])
        for o in reqs:
            per_query[o["query_id"]][source] = o
        ok = [o for o in reqs if o["outcome"] == "ok"]
        first = reqs[0]["rid"] if reqs else None
        all_s = _pct([o["latency_s"] for o in ok])
        warm = _pct([o["latency_s"] for o in ok if o["rid"] != first])
        curve = {f"{t:.1f}": wilson(sum(1 for o in reqs if o["outcome"] == "ok" and o["latency_s"] <= t),
                                    len(reqs)) for t in CURVE_S}
        p95 = all_s["p95"]
        verdict = None if p95 is None else ("fits" if p95 <= 2.0 else "tight" if p95 <= 2.5 else "truncation_only")
        out[source] = {"requests": len(reqs), "ok": len(ok),
                       "outcomes": dict(Counter(o["outcome"] for o in reqs)),
                       # DEV-1/DEV-2 repeats ran alone, not started with the other sources.
                       "stand_in_repeats": sorted(o["rid"] for o in reqs if o.get("stand_in")),
                       "latency_s": all_s, "warm_only_s": warm, "completion": curve, "verdict": verdict,
                       "headers_s_p50": nearest_rank([o["headers_s"] for o in ok if o.get("headers_s")], 50)}
    together = {}
    queries = list(per_query.values())
    for t in CURVE_S:
        k = sum(1 for q in queries if len(q) == 3 and all(o["outcome"] == "ok" and o["latency_s"] <= t
                                                          for o in q.values()))
        together[f"{t:.1f}"] = wilson(k, len(queries))
    fan_out = [max(o["latency_s"] for o in q.values()) for q in queries
               if len(q) == 3 and all(o["outcome"] == "ok" for o in q.values())]
    out["all_three"] = {"completion": together, "fan_out_max_latency_s": _pct(fan_out)}
    return out


def _records(bodies: list[dict[str, Any]], pass_id: str, kind: str = "search") -> dict[str, list[tuple[dict[str, Any], Rec]]]:
    out: dict[str, list[tuple[dict[str, Any], Rec]]] = defaultdict(list)
    for entry in bodies:
        if entry.get("pass") == pass_id and entry.get("kind") == kind and entry.get("status") == 200:
            for rec in parse(entry["source"], entry["body"]).records:
                out[entry["source"]].append((entry, rec))
    return out


FIELDS = {
    "title": lambda r: bool(r.title),
    "authors": lambda r: r.n_authors >= 1,
    "year": lambda r: r.year is not None,
    "venue": lambda r: bool(r.venue),
    "abstract": lambda r: r.has_abstract,
    "citation_count": lambda r: r.citation_count is not None,
    "doi": lambda r: bool(r.doi) and not r.doi.startswith("10.48550/arxiv."),
    "arxiv_id": lambda r: bool(r.arxiv_id) or bool(r.doi and r.doi.startswith("10.48550/arxiv.")),
    "oa_link": lambda r: bool(r.oa),
    "full_date": lambda r: bool(r.full_date and len(r.full_date) >= 10),
}


def completeness(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    recs = _records(bodies, "P1")
    out = {}
    for source in SOURCES:
        rows = [r for _, r in recs.get(source, [])]
        shares = {f: wilson(sum(1 for r in rows if test(r)), len(rows)) for f, test in FIELDS.items()}
        bars = {"title": 0.95, "authors": 0.95, "year": 0.95, "abstract": 0.70}
        if source != "arxiv":
            bars["citation_count"] = 0.95
        failed = [f for f, bar in bars.items() if (shares[f]["rate"] or 0) < bar]
        out[source] = {"records": len(rows), "fields": shares, "failed_bars": failed,
                       "meets": bool(rows) and not failed,
                       "structural_gap": ["citation_count"] if source == "arxiv" else []}
    return out


#: The passes ASM-1 is measured on. A1 came after the verdict and does not change it (D120).
ASM1_PASSES = ("P1", "P2", "P5")


def limits(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for source in SOURCES:
        rows = [o for o in outcomes if o.get("source") == source and o.get("pass") in ASM1_PASSES]
        statuses = Counter(str(o.get("status")) for o in rows)
        throttled = sum(1 for o in rows if o.get("status") == 429)
        busy = sum(1 for o in rows if o.get("status") == 503)
        rate_headers: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for o in rows:
            for name, value in (o.get("headers") or {}).items():
                if name.startswith(("x-ratelimit", "ratelimit")):
                    bucket = rate_headers[f"{o.get('kind')}"][name]
                    if value not in bucket and len(bucket) < 12:
                        bucket.append(value)
        sizes = [o["bytes"] for o in rows if o.get("outcome") == "ok" and o.get("bytes") is not None]
        if source == "openalex":
            meets, rule = throttled == 0, "keyed: zero 429s"
        elif source == "semantic_scholar":
            meets, rule = (throttled / len(rows) <= 0.10) if rows else False, "keyless: 429s on <= 10% of requests"
        else:
            meets, rule = throttled == 0 and busy == 0, "zero 429s or 503s"
        out[source] = {"requests": len(rows), "statuses": dict(statuses), "throttled_429": throttled,
                       "share_429": wilson(throttled, len(rows)), "busy_503": busy,
                       "retry_after_values": sorted({(o.get("headers") or {}).get("retry-after")
                                                     for o in rows} - {None}),
                       "rate_limit_headers": {k: dict(v) for k, v in rate_headers.items()},
                       "response_bytes": _pct([float(s) for s in sizes]), "rule": rule, "meets": meets}
    return out


def _jaccard(a: str, b: str) -> float:
    x, y = set(a.split()), set(b.split())
    return len(x & y) / len(x | y) if x | y else 0.0


def _anchor_rank(recs: list[Rec], anchor: dict[str, Any], id_ok: bool) -> tuple[int | None, str | None, list[int]]:
    """definitions.anchor_match: (rank, rule, near-miss ranks) for one anchor in one result list."""
    want = norm_title(anchor["title"])
    near: list[int] = []
    for rec in recs:
        if id_ok and rec.arxiv_id == anchor["arxiv_id"]:
            return rec.rank, "arxiv_id", []
        if rec.ntitle and rec.ntitle == want:
            return rec.rank, "title", []
        if rec.ntitle and _jaccard(rec.ntitle, want) >= 0.8:
            near.append(rec.rank)
    return None, None, near


def _verified_anchor_ids(protocol: dict[str, Any], bodies: list[dict[str, Any]]) -> dict[str, bool]:
    """P2's arXiv check: an anchor id counts only if arXiv returned the anchor's title for it."""
    returned = {}
    for entry in bodies:
        if entry.get("rid") == "P2/anchor/arxiv" and entry.get("status") == 200:
            for rec in parse("arxiv", entry["body"]).records:
                returned[rec.arxiv_id] = rec.title
    return {a["arxiv_id"]: norm_title(returned.get(a["arxiv_id"])) == norm_title(a["title"])
            for t in protocol["topics"] for a in t["anchors"]}


def anchors(protocol: dict[str, Any], bodies: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_list = [(t["id"], a) for t in protocol["topics"] for a in t["anchors"]]
    # P2's arXiv check verifies every anchor id against its title.
    returned = {}
    presence: dict[str, set[str]] = {s: set() for s in SOURCES}
    for entry in bodies:
        if entry.get("pass") != "P2" or entry.get("status") != 200:
            continue
        if entry.get("rid") == "P2/anchor/arxiv":
            for rec in parse("arxiv", entry["body"]).records:
                returned[rec.arxiv_id] = rec.title
                presence["arxiv"].add(rec.arxiv_id)
        elif entry.get("rid") == "P2/anchor/semantic_scholar":
            payload = json.loads(entry["body"])
            ids = [a["arxiv_id"] for _, a in anchor_list]
            if isinstance(payload, list):
                for aid, item in zip(ids, payload, strict=False):
                    if isinstance(item, dict):
                        presence["semantic_scholar"].add(aid)
        elif entry.get("rid") == "P2/anchor/openalex":
            for rec in parse("openalex", entry["body"]).records:
                aid = norm_arxiv(rec.doi) if rec.doi else None
                if aid:
                    presence["openalex"].add(aid)
    verified = {a["arxiv_id"]: norm_title(returned.get(a["arxiv_id"])) == norm_title(a["title"])
                for _, a in anchor_list}

    by_query: dict[tuple[str, str, str], list[Rec]] = {}
    for entry in bodies:
        if entry.get("pass") == "P1" and entry.get("kind") == "search" and entry.get("status") == 200:
            by_query[(entry["source"], entry["topic"], entry["phrasing"])] = parse(entry["source"], entry["body"]).records
    out: dict[str, Any] = {"id_verified": verified,
                           "id_mismatches": [a for a, ok in verified.items() if not ok],
                           "present_in_source": {s: sorted(v) for s, v in presence.items()}}
    table = {}
    for source in SOURCES:
        per_phrasing = {}
        for phrasing in ("keyword", "natural_language"):
            hits10 = hits50 = 0
            rows = []
            for topic, anchor in anchor_list:
                recs = by_query.get((source, topic, phrasing))
                rank, rule, near = (_anchor_rank(recs, anchor, verified[anchor["arxiv_id"]])
                                    if recs is not None else (None, None, []))
                if rank is not None and rank <= 10:
                    hits10 += 1
                if rank is not None and rank <= 50:
                    hits50 += 1
                rows.append({"topic": topic, "anchor": anchor["title"], "rank": rank, "rule": rule,
                             "near_miss_ranks": near if rank is None else [],
                             "query_failed": recs is None,
                             "in_source": anchor["arxiv_id"] in presence[source]})
            per_phrasing[phrasing] = {"top10": wilson(hits10, len(anchor_list)),
                                      "top50": wilson(hits50, len(anchor_list)), "anchors": rows}
        top50 = per_phrasing["keyword"]["top50"]["rate"] or 0.0
        table[source] = {**per_phrasing, "meets": top50 >= 0.5}
    out["recall"] = table
    return out


def asm2(protocol: dict[str, Any], run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    works = run["probe_works"]
    finished = [w for w in works if w["result"] in ("fetched", "failed", "no_oa_link")]
    excluded = Counter(w["result"] for w in works if w["result"] not in ("fetched", "failed", "no_oa_link"))
    n = len(finished)
    l0 = sum(1 for w in finished if w["candidates"])
    l1 = sum(1 for w in finished if any(c["url_check"] == "ok" for c in w["candidates"]))
    fetched = [w for w in finished if w["result"] == "fetched"]

    def winning(w: dict[str, Any]) -> dict[str, Any]:
        return next(a for a in w["attempts"] if a.get("outcome") == "fetched")

    final_hosts = Counter(winning(w).get("final_host") for w in fetched)
    target = 0.9 * len(fetched)
    a2_hosts, covered = [], 0
    for host, count in sorted(final_hosts.items(), key=lambda kv: (-kv[1], kv[0] or "")):
        if covered >= target:
            break
        a2_hosts.append(host)
        covered += count
    a2 = sum(1 for w in fetched if winning(w).get("final_host") in a2_hosts)

    def arxiv_unknown(w: dict[str, Any]) -> bool:
        first = w["attempts"][0] if w["attempts"] else None
        return bool(w["candidates"] and w["candidates"][0]["kind"] == "arxiv_pdf" and first
                    and first.get("outcome") == "not_sent_budget")

    a1_pool = [w for w in finished if not arxiv_unknown(w)]
    a1 = sum(1 for w in a1_pool if w["result"] == "fetched"
             and set(winning(w).get("hosts") or []) == {"arxiv.org"})

    downloads = run["downloads"]
    dl_done = [d for d in downloads if d.get("outcome") == "downloaded"]
    l3_ok = sum(1 for d in dl_done if d.get("check") == "ok")
    per_source = {}
    for source in SOURCES:
        mine = [w for w in finished if source in w["sources"]]
        per_source[source] = {
            "works": len(mine),
            "fetched": wilson(sum(1 for w in mine if w["result"] == "fetched"), len(mine)),
            "has_link_from_this_source": wilson(
                sum(1 for w in mine if any(source in c["supplied_by"] for c in w["candidates"])), len(mine)),
        }
    return {
        "works_total": len(works),
        "denominator": n,
        "excluded_by_cap": dict(excluded),
        # Counted as failures, as pre-registered; shown so a stopped host is visible.
        "failed_with_a_stopped_host": sum(1 for w in finished if w["result"] == "failed"
                                          and w.get("host_stopped")),
        "L0_has_link": wilson(l0, n),
        "L1_https_usable": wilson(l1, n),
        "L2_fetched_A3": wilson(len(fetched), n),
        "A1_arxiv_only": {**wilson(a1, len(a1_pool)), "unknown_excluded": n - len(a1_pool)},
        "A2": {**wilson(a2, n), "hosts": a2_hosts},
        "L3_sample": {"chosen": len(downloads), "downloaded": len(dl_done), "passed_checks": wilson(l3_ok, len(dl_done)),
                      "checks": dict(Counter(str(d.get("check") or d.get("outcome")) for d in downloads)),
                      "bytes": _pct([float(d["bytes"]) for d in dl_done if d.get("bytes")]),
                      "pages": _pct([float(d["pages"]) for d in dl_done if d.get("pages")])},
        "per_source": per_source,
        "verdict": "PASS" if n and len(fetched) / n >= 0.60 else "FAIL",
    }


def _total_size(hop: dict[str, Any]) -> int | None:
    rng = hop.get("content_range") or ""
    if "/" in rng and rng.rsplit("/", 1)[1].isdigit():
        return int(rng.rsplit("/", 1)[1])
    length = hop.get("content_length")
    if hop.get("status") == 200 and length and str(length).isdigit():
        return int(length)
    return None


def allowlist_evidence(run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    works = run["probe_works"]
    initial, schemes, kinds = Counter(), Counter(), Counter()
    for w in works:
        for c in w["candidates"]:
            initial[(urlsplit(c["url"]).hostname or "?").lower()] += 1
            schemes[urlsplit(c["url"]).scheme.lower() or "?"] += 1
            kinds[c["kind"]] += 1
    attempts = [a for w in works for a in w["attempts"] if "hops" in a]
    final_ok, via, hop_counts, types, sizes, refusals, outcomes = (Counter() for _ in range(7))
    for a in attempts:
        outcomes[a["outcome"]] += 1
        sent = [h for h in a["hops"] if h.get("sent")]
        if a["outcome"] == "fetched":
            final_ok[a.get("final_host")] += 1
            hop_counts[len(sent) - 1] += 1
            for h in sent[:-1]:
                via[h.get("host")] += 1
            size = _total_size(sent[-1]) if sent else None
            if size is not None:
                sizes[
                    "<1MB" if size < 1 << 20 else "1-5MB" if size < 5 << 20 else "5-20MB" if size < 20 << 20
                    else "20-50MB" if size <= 50 << 20 else ">50MB"
                ] += 1
        last = a["hops"][-1] if a["hops"] else {}
        if last.get("content_type"):
            types[(last["content_type"].split(";")[0].strip().lower(), a["outcome"])] += 1
        if a["outcome"] == "refused":
            refusals[last.get("reason")] += 1
        elif a["outcome"] not in ("fetched",):
            refusals[f"{a['outcome']}:{last.get('status')}"] += 1
    return {
        "candidate_initial_hosts": dict(initial.most_common(40)),
        "candidate_schemes_as_given": dict(schemes),
        "candidate_kinds": dict(kinds),
        "attempt_outcomes": dict(outcomes),
        "fetched_final_hosts": dict(final_ok.most_common()),
        "fetched_via_intermediate_hosts": dict(via.most_common()),
        "fetched_redirect_counts": {str(k): v for k, v in sorted(hop_counts.items())},
        "final_content_types": {f"{t} -> {o}": c for (t, o), c in types.most_common(20)},
        "fetched_pdf_sizes": dict(sizes),
        "failure_reasons": dict(refusals.most_common(30)),
        "upgraded_http_attempts": sum(1 for a in attempts if a.get("upgraded")),
    }


def licence_evidence(run: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    works = {w.key: w for w in build_works(p1_keyword_records())}
    openalex_licences, s2_licences = Counter(), Counter()
    for w in run["probe_works"]:
        if w["result"] != "fetched":
            continue
        attempt = next(a for a in w["attempts"] if a.get("outcome") == "fetched")
        url = w["candidates"][attempt["url_index"]]["url"]
        work = works.get(w["work"])
        licence = None
        found = False
        if work is not None:
            url_arxiv = arxiv_from_url(url)
            for rec in (r for r in work.records if r.source == "openalex"):
                for landing, pdf, lic in rec.locations:
                    if (pdf and pdf.strip() == url) or (url_arxiv and url_arxiv in (arxiv_from_url(pdf), arxiv_from_url(landing))):
                        licence, found = lic, True
                        break
                if found:
                    break
            if not found:
                licence = next((r.best_oa_licence for r in work.records
                                if r.source == "openalex" and r.best_oa_licence), None)
            has_openalex = any(r.source == "openalex" for r in work.records)
        else:
            has_openalex = False
        openalex_licences[licence or ("none reported" if has_openalex else "no OpenAlex record")] += 1
        s2 = {c["licences"].get("semantic_scholar") for c in w["candidates"] if "semantic_scholar" in c["licences"]}
        s2 = {x for x in s2 if x}
        s2_licences["reported" if s2 else "none reported"] += 1
    return {"openalex_licence_of_fetched": dict(openalex_licences.most_common()),
            "s2_licence_field_on_fetched": dict(s2_licences)}


def dedup_evidence() -> dict[str, Any]:
    by_topic = search_records("P1", "keyword")
    per_source_ids: dict[str, Counter[str]] = defaultdict(Counter)
    cross, link_types, spreads, title_only, within = Counter(), Counter(), Counter(), 0, 0
    title_only_cross = 0
    for topic, recs in by_topic.items():
        for r in recs:
            keys = record_keys(r)
            per_source_ids[r.source]["records"] += 1
            per_source_ids[r.source]["doi"] += any(k.startswith("doi:") for k in keys)
            per_source_ids[r.source]["arxiv"] += any(k.startswith("arxiv:") for k in keys)
        works = build_works({topic: recs})
        for w in works:
            sources = sorted(w.sources)
            if len(sources) >= 2:
                cross["+".join(sources)] += 1
                shared = Counter(k.split(":")[0] for r in w.records for k in record_keys(r))
                link_types["arxiv" if shared.get("arxiv", 0) >= 2 else "doi"] += 1
            years = [r.year for r in w.records if r.year]
            if len(w.records) >= 2 and len(years) >= 2:
                spread = max(years) - min(years)
                spreads["0" if spread == 0 else "1" if spread == 1 else ">=2"] += 1
        for a, b in combinations(works, 2):
            ta = {r.ntitle for r in a.records if len(r.ntitle.split()) >= 3}
            tb = {r.ntitle for r in b.records if len(r.ntitle.split()) >= 3}
            if ta & tb:
                title_only += 1
                title_only_cross += a.sources != b.sources
        for source in SOURCES:
            titles = Counter(r.ntitle for r in recs if r.source == source and len(r.ntitle.split()) >= 3)
            within += sum(c * (c - 1) // 2 for c in titles.values())
    ids = {s: {"records": c["records"], "doi": wilson(c["doi"], c["records"]),
               "arxiv_id": wilson(c["arxiv"], c["records"])} for s, c in per_source_ids.items()}
    return {"scope": "per topic, session-1 keyword results, all 50 ranks per source",
            "identifier_coverage": ids, "cross_source_linked_works": dict(cross),
            "cross_source_link_type": dict(link_types), "title_only_pairs": title_only,
            "title_only_pairs_across_different_source_sets": title_only_cross,
            "within_source_same_title_pairs": within, "year_spread_of_linked_works": dict(spreads)}


def stability(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    s1 = {(e["source"], e["topic"]): [r.source_id for r in parse(e["source"], e["body"]).records[:10]]
          for e in bodies if e.get("pass") == "P1" and e.get("phrasing") == "keyword"
          and e.get("kind") == "search" and e.get("status") == 200}
    s2 = {(e["source"], e["topic"]): [r.source_id for r in parse(e["source"], e["body"]).records[:10]]
          for e in bodies if e.get("pass") == "P5" and e.get("kind") == "search" and e.get("status") == 200}
    out = {}
    for source in SOURCES:
        overlaps = [len(set(s1[k]) & set(s2[k])) / max(1, min(len(s1[k]), len(s2[k]), 10))
                    for k in s1 if k[0] == source and k in s2]
        out[source] = {"topics": len(overlaps),
                       "mean_top10_overlap": round(sum(overlaps) / len(overlaps), 3) if overlaps else None,
                       "min": min(overlaps) if overlaps else None}
    return out


def phrasing(bodies: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for source in SOURCES:
        row = {}
        for ph in ("keyword", "natural_language"):
            counts = [len(parse(e["source"], e["body"]).records) for e in bodies
                      if e.get("pass") == "P1" and e.get("phrasing") == ph and e.get("source") == source
                      and e.get("kind") == "search" and e.get("status") == 200]
            totals = [parse(e["source"], e["body"]).total for e in bodies
                      if e.get("pass") == "P1" and e.get("phrasing") == ph and e.get("source") == source
                      and e.get("kind") == "search" and e.get("status") == 200]
            row[ph] = {"queries_ok": len(counts), "zero_result": wilson(sum(1 for c in counts if c == 0), len(counts)),
                       "mean_returned": round(sum(counts) / len(counts), 1) if counts else None,
                       "reported_totals": totals}
        out[source] = row
    return out


def singles(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for source in SOURCES:
        rows = [o for o in outcomes if o.get("pass") == "P2" and o.get("lookup") == "single_record"
                and o.get("source") == source]
        out[source] = {"requests": len(rows), "outcomes": dict(Counter(o["outcome"] for o in rows)),
                       "latency_s": _pct([o["latency_s"] for o in rows if o["outcome"] == "ok"])}
    return out


def addendum_a1(protocol: dict[str, Any], bodies: list[dict[str, Any]],
                outcomes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A1 (D120): anchor recall under AND and OR, the same day, and the pre-registered decision."""
    a1 = [e for e in bodies if e.get("pass") == "A1"]
    if not a1:
        return None
    anchor_list = [(t["id"], a) for t in protocol["topics"] for a in t["anchors"]]
    verified = _verified_anchor_ids(protocol, bodies)
    out: dict[str, Any] = {}
    found: dict[tuple[str, str], set[str]] = {}
    for join in ("AND", "OR"):
        row: dict[str, Any] = {}
        for phrasing in ("keyword", "natural_language"):
            ok = {e["topic"]: parse("arxiv", e["body"]) for e in a1
                  if e.get("construction") == join and e.get("phrasing") == phrasing and e.get("status") == 200}
            hits10 = hits50 = 0
            rows = []
            found[(join, phrasing)] = set()
            for topic, anchor in anchor_list:
                parsed = ok.get(topic)
                rank, rule, near = (_anchor_rank(parsed.records, anchor, verified[anchor["arxiv_id"]])
                                    if parsed is not None else (None, None, []))
                hits10 += rank is not None and rank <= 10
                if rank is not None and rank <= 50:
                    hits50 += 1
                    found[(join, phrasing)].add(anchor["arxiv_id"])
                rows.append({"topic": topic, "anchor": anchor["title"], "rank": rank, "rule": rule,
                             "near_miss_ranks": near, "query_failed": parsed is None})
            counts = [len(p.records) for p in ok.values()]
            row[phrasing] = {
                "queries_ok": len(ok),
                "top10": wilson(hits10, len(anchor_list)),
                "top50": wilson(hits50, len(anchor_list)),
                "zero_result": wilson(sum(1 for c in counts if c == 0), len(counts)),
                "mean_returned": round(sum(counts) / len(counts), 1) if counts else None,
                "reported_totals": [ok[t].total for t in sorted(ok)],
                "anchors": rows,
            }
        rows_out = [o for o in outcomes if o.get("pass") == "A1" and o.get("construction") == join]
        row["outcomes"] = dict(Counter(o["outcome"] for o in rows_out))
        row["latency_s"] = _pct([o["latency_s"] for o in rows_out if o["outcome"] == "ok"])
        out[join] = row
    or50 = out["OR"]["keyword"]["top50"]["k"]
    and50 = out["AND"]["keyword"]["top50"]["k"]
    n = len(anchor_list)
    adopt_or = 2 * or50 >= n and or50 > and50
    out["paired_keyword_top50"] = {
        "only_or": sorted(found[("OR", "keyword")] - found[("AND", "keyword")]),
        "only_and": sorted(found[("AND", "keyword")] - found[("OR", "keyword")]),
        "both": len(found[("OR", "keyword")] & found[("AND", "keyword")]),
    }
    out["decision"] = {
        "rule": protocol["addenda"]["A1"]["decision_rule"],
        "or_top50": or50, "and_top50": and50, "n": n,
        "adopt": "OR" if adopt_or else "AND",
        "or_beats_and_below_bar": (not adopt_or) and or50 > and50,
    }
    return out


#: D122's source deadline: the bar A2 judges each page size's latency against (D125).
A2_DEADLINE_S = 2.0


def a2_choice(per_size: dict[int, dict[str, Any]], n_anchors: int) -> int | None:
    """D125's rule: the largest page whose p95 fits the deadline and whose keyword anchors reach 50%.

    `p95_s` is None when the 95th-ranked request failed, which no deadline admits.
    """
    for size in sorted(per_size, reverse=True):
        row = per_size[size]
        p95 = row["p95_s"]
        fits = p95 is not None and p95 <= A2_DEADLINE_S  # MUTATION-POINT: a2-deadline
        if fits and 2 * row["keyword_anchors"] >= n_anchors:  # MUTATION-POINT: a2-anchors
            return size
    return None


def addendum_a2(protocol: dict[str, Any], bodies: list[dict[str, Any]],
                outcomes: list[dict[str, Any]]) -> dict[str, Any] | None:
    """A2 (D125): OpenAlex latency and anchors at three page sizes, and the pre-registered choice."""
    a2_bodies = [e for e in bodies if e.get("pass") == "A2"]
    a2_out = {o["rid"]: o for o in outcomes if o.get("pass") == "A2"}
    if not a2_out:
        return None
    anchor_list = [(t["id"], a) for t in protocol["topics"] for a in t["anchors"]]
    verified = _verified_anchor_ids(protocol, bodies)
    query_ids = [f"{t['id']}-{ph}" for t in protocol["topics"] for ph in ("keyword", "natural_language")]
    parsed = {(e["query_id"], e["per_page"]): parse("openalex", e["body"])
              for e in a2_bodies if e.get("status") == 200}
    per_size: dict[int, dict[str, Any]] = {}
    for size in A2_SIZES:
        rows = [a2_out.get(f"A2/{q}/{size}") for q in query_ids]
        ok = [o for o in rows if o is not None and o["outcome"] == "ok"]
        # Every planned request counts; one that failed or was never sent did not finish.
        all_s = [o["latency_s"] if o is not None and o["outcome"] == "ok" else float("inf")  # MUTATION-POINT: a2-failures
                 for o in rows]
        p95 = nearest_rank(all_s, 95)
        hits = []
        for topic, anchor in anchor_list:
            page = parsed.get((f"{topic}-keyword", size))
            rank, rule, _ = (_anchor_rank(page.records, anchor, verified[anchor["arxiv_id"]])
                             if page is not None else (None, None, []))
            hits.append({"topic": topic, "anchor": anchor["title"], "rank": rank, "rule": rule,
                         "query_failed": page is None})
        per_size[size] = {
            "requests": len([o for o in rows if o is not None]),
            "outcomes": dict(Counter(o["outcome"] for o in rows if o is not None)),
            "p95_s": None if p95 is None or p95 == float("inf") else p95,
            "latency_ok_s": _pct([o["latency_s"] for o in ok]),
            "within_deadline": wilson(sum(1 for x in all_s if x <= A2_DEADLINE_S), len(query_ids)),
            "headers_s": _pct([o["headers_s"] for o in ok if o.get("headers_s") is not None]),
            "bytes": _pct([float(o["bytes"]) for o in ok if o.get("bytes") is not None]),
            "keyword_anchors": sum(1 for h in hits if h["rank"] is not None and h["rank"] <= size),
            "anchors": hits,
        }
    prefix: dict[str, Any] = {}
    for size in A2_SIZES[1:]:
        same = differ = 0
        examples = []
        for q in query_ids:
            big, small = parsed.get((q, 50)), parsed.get((q, size))
            if big is None or small is None:
                continue
            if [r.source_id for r in small.records] == [r.source_id for r in big.records[:size]]:
                same += 1
            else:
                differ += 1
                examples.append(q)
        prefix[str(size)] = {"same_as_first_of_50": same, "differ": differ, "differing_queries": examples}
    cost = sum(float((o.get("headers") or {}).get("x-ratelimit-cost-usd") or 0) for o in a2_out.values())
    choice = a2_choice(per_size, len(anchor_list))
    return {
        "deadline_s": A2_DEADLINE_S,
        "per_size": {str(k): v for k, v in per_size.items()},
        "prefix_of_50": prefix,
        "cost_usd": round(cost, 4),
        "decision": {"rule": protocol["addenda"]["A2"]["decision_rule"], "n_anchors": len(anchor_list),
                     "chosen_per_page": choice, "owner_decides": choice is None},
    }


def planned_coverage(protocol: dict[str, Any], api: list[dict[str, Any]]) -> dict[str, Any]:
    decided_rids = decided(api)
    missing = []
    for pass_id, phrasings in (("P1", ("keyword", "natural_language")), ("P5", ("reversed_keyword",))):
        for t in protocol["topics"]:
            for ph in phrasings:
                for s in SOURCES:
                    rid = f"{pass_id}/{t['id']}-{ph}/{s}"
                    if rid not in decided_rids:
                        missing.append(rid)
    for s in SOURCES:
        if f"P2/anchor/{s}" not in decided_rids:
            missing.append(f"P2/anchor/{s}")
    if any(r.get("pass") == "A1" for r in api):  # A1 is planned only once its run has started
        for t in protocol["topics"]:
            for ph in ("keyword", "natural_language"):
                for join in ("AND", "OR"):
                    if f"A1/{t['id']}-{ph}/{join}" not in decided_rids:
                        missing.append(f"A1/{t['id']}-{ph}/{join}")
    if any(r.get("pass") == "A2" for r in api):  # and A2 once its run has started
        for t in protocol["topics"]:
            for ph in ("keyword", "natural_language"):
                for size in A2_SIZES:
                    if f"A2/{t['id']}-{ph}/{size}" not in decided_rids:
                        missing.append(f"A2/{t['id']}-{ph}/{size}")
    return {"missing": missing, "complete": not missing}


def analyse() -> int:
    protocol, digest = load_protocol()
    run = load_run()
    outcomes = _outcomes(run["api"])
    bodies = run["bodies"]
    lat1 = latency(outcomes, "P1")
    comp = completeness(bodies)
    lim = limits(outcomes)
    anc = anchors(protocol, bodies)
    per_source = {}
    for s in SOURCES:
        parts = {"1a": lat1[s]["verdict"] in ("fits", "tight"), "1b": comp[s]["meets"],
                 "1c": lim[s]["meets"], "1d": anc["recall"][s]["meets"]}
        per_source[s] = {**parts, "meets_all": all(parts.values())}
    meeting = [s for s, v in per_source.items() if v["meets_all"]]
    if len(meeting) >= 2 and ({"openalex", "semantic_scholar"} & set(meeting)):
        asm1 = "PASS"
    elif len(meeting) == 1:
        asm1 = "PARTIAL"
    else:
        asm1 = "FAIL"
    records = run["api"] + run["pdf"]
    violations = check_log(merged(records), protocol, decided(records), api_addresses(run["api"]))
    relevance_present = sum(1 for e in bodies if e.get("source") == "openalex" and e.get("pass") == "P1"
                            and e.get("status") == 200 and '"relevance_score"' in e["body"])
    result = {
        "protocol_sha256": digest,
        "protocol_original_sha256": protocol["amended"]["original_sha256"],
        "protocol_hashes_in_run_logs": sorted({r["protocol_sha256"] for r in run["api"]
                                               if r.get("event") == "run" and r.get("protocol_sha256")}),
        "generated_at": iso(wall()),
        "s2_mode": "keyless",
        "deviations": deviations(run["api"]),
        "stopped_hosts": [r.get("stopped_hosts") for r in run["api"] + run["pdf"]
                          if r.get("event") == "run_end" and r.get("stopped_hosts")],
        "seed_streams": "20260930/P1, /P2/<source>, /P3, /P4, /P5",
        "interpretations": {
            "hops": "at most 5 redirects are followed, so a chain is at most 6 requests; every one counts",
            "retry_after_ceiling_s": 600,
            "latency": "request issued to last body byte, 2xx only; failures count as never completing",
            "host_stopped_works": "a work whose remaining URLs went unsent because the host-stop rule "
                                  "fired counts as a failure, as the ASM-2 denominator is written; "
                                  "asm2.failed_with_a_stopped_host shows how many, so the other reading "
                                  "can be computed without rerunning",
            "s2_chain_urls": "every hop of a chain that started from an S2-only URL is recorded as host "
                             "plus hash, not only the URL itself",
            "host_stop_on_resume": "a resumed run replays the stop rule and Retry-After from the same "
                                   "session's log; a request logged as sent with no end is taken to "
                                   "have ended when the resume starts",
        },
        "planned_coverage": planned_coverage(protocol, run["api"]),
        "log_check": {"sent_requests": len(merged(records)), "violations": violations,
                      "sent_without_outcome": unfinished(records)},
        "asm1": {"verdict": asm1, "per_source": per_source, "latency_session1": lat1,
                 "latency_session2": latency(outcomes, "P5"), "completeness": comp, "limits": lim,
                 "anchors": anc, "single_record": singles(outcomes),
                 "openalex_bodies_with_relevance_score": relevance_present},
        "asm2": asm2(protocol, run),
        "addendum_a1": addendum_a1(protocol, bodies, outcomes),
        "addendum_a2": addendum_a2(protocol, bodies, outcomes),
        "evidence": {"allowlist": allowlist_evidence(run), "licence": licence_evidence(run),
                     "dedup": dedup_evidence(), "stability": stability(bodies), "phrasing": phrasing(bodies)},
    }
    text = json.dumps(result, indent=1, ensure_ascii=False, default=str)
    OUT_ANALYSIS.write_text(text + "\n", encoding="utf-8")
    print(f"wrote {OUT_ANALYSIS.relative_to(ROOT)}; ASM-1 {asm1}, ASM-2 {result['asm2']['verdict']}, "
          f"{len(violations)} log violations")
    return 0


# ============================================================ tracked outputs


def _s2_ids(bodies: list[dict[str, Any]]) -> set[str]:
    ids = set()
    for e in bodies:
        if e.get("source") == "semantic_scholar":
            ids.update(re.findall(r'"paperId":\s*"([0-9a-f]{40})"', e["body"]))
    return ids


def _sanitise_url(url: str | None, s2_only: bool, s2_ids: set[str]) -> Any:
    if not url:
        return url
    if s2_only:
        return {"host": (urlsplit(url).hostname or "?").lower(), "url": s2_hash(url)}
    for pid in s2_ids:
        if pid in url:
            url = url.replace(pid, s2_hash(pid))
    return url


def finalize() -> int:
    protocol, digest = load_protocol()
    env = load_env()
    secrets = Secrets([env.get("OPENALEX_API_KEY", ""), env.get("S2_API_KEY", "")])
    run = load_run()
    s2_ids = _s2_ids(run["bodies"])

    # 17: the API request log.
    requests_doc = _requests_doc(run, digest, s2_ids)

    # 18: probe and download logs, S2-only identifiers and URLs hashed.
    works = {w.key: w for w in build_works(p1_keyword_records())}
    probe_doc = probe_document(run, digest, s2_ids, works)

    for path, doc in ((OUT_REQUESTS, requests_doc), (OUT_PROBE, probe_doc)):
        text = json.dumps(doc, indent=1, ensure_ascii=False, default=str)
        secrets.assert_absent(text, where=str(path))
        path.write_text(text + "\n", encoding="utf-8")
        print("wrote", path.relative_to(ROOT))

    # The frozen pools: OpenAlex and arXiv bodies only (both CC0). S2 stays local.
    # Frozen means frozen: once written they are verified against the log, never
    # rewritten, because the protocol hash stamped inside would change their bytes.
    entries = sorted((e for e in run["bodies"] if e.get("source") in ("openalex", "arxiv")
                      and e.get("pass") in ASM1_PASSES), key=lambda e: e["rid"])
    if POOLS_PATH.exists():
        _verify_frozen_pools(entries)
        print(f"{POOLS_PATH.relative_to(ROOT)} exists and matches the log; not rewritten")
        return 0
    payload = json.dumps({"protocol_sha256": digest, "entries": entries}, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    secrets.assert_absent(payload.decode("utf-8"), where=str(POOLS_PATH))
    POOLS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with POOLS_PATH.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(payload)
    blob = POOLS_PATH.read_bytes()
    manifest = {
        "dataset": "scholiast-m4-discovery-pools",
        "decision": "D116",
        "protocol_sha256": digest,
        "file": POOLS_PATH.name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "entries": len(entries),
        "by_source_and_pass": dict(Counter(f"{e['source']}/{e['pass']}/{e['kind']}" for e in entries)),
        "recorded_from": min(e["received_at"] for e in entries) if entries else None,
        "recorded_to": max(e["received_at"] for e in entries) if entries else None,
        "sources_included": ["openalex", "arxiv"],
        "sources_excluded": {"semantic_scholar": "licence restricts redistribution (D116); kept under var/spike/"},
        "licence": "OpenAlex data CC0; arXiv metadata CC0",
        "requests": {s: protocol["sources"][s]["search"] for s in ("openalex", "arxiv")},
        "query_construction": protocol["query_construction"],
        "deviations_after_approval": protocol["deviations_after_approval"],
        "note": "The OpenAlex searches DEV-2 repeats were first sent with the unamended query and refused "
                "with HTTP 400; both the refusal and the repeat are in the pools (D117).",
        "caveat": protocol["pools_caveat"],
    }
    POOLS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    # Reload and verify, as the loader in later phases will.
    check_blob = POOLS_PATH.read_bytes()
    assert hashlib.sha256(check_blob).hexdigest() == manifest["sha256"], "pools hash mismatch on reload"
    reloaded = json.loads(gzip.decompress(check_blob))
    assert len(reloaded["entries"]) == len(entries)
    bad = [e["rid"] for e in reloaded["entries"] if e["status"] == 200 and parse(e["source"], e["body"]).error]
    print(f"wrote {POOLS_PATH.relative_to(ROOT)} ({len(blob)} bytes, {len(entries)} entries); "
          f"reload verified; unparseable 200s: {bad or 'none'}")
    return 0


def _s2_only_flags(work: dict[str, Any]) -> list[bool]:
    """Per candidate URL: did only S2 supply it? (pdf_probe.recorded_per_attempt)"""
    return [set(c["supplied_by"]) == {"semantic_scholar"} for c in work["candidates"]]


def _followed_s2_only(work: dict[str, Any] | None, url_index: int | None) -> bool:
    """Whether an attempt or download followed an S2-only URL. Unknown counts as yes."""
    if work is None or url_index is None or url_index >= len(work["candidates"]):
        return True
    return _s2_only_flags(work)[url_index]


def probe_document(run: dict[str, list[dict[str, Any]]], digest: str, s2_ids: set[str],
                   works: dict[str, Any]) -> dict[str, Any]:
    """The tracked 18_oa_probe.json (D116: S2 data stays local).

    A URL only S2 supplied, and every hop of a chain that started from one, is
    recorded as host plus hash, in the probe attempts and the downloads alike; a
    work key only S2 supplied is hashed; S2 paper ids anywhere are hashed.
    """
    probe_by_work = {w["work"]: w for w in run["probe_works"]}

    def key_public(key: str) -> str:
        work = works.get(key)
        if work and any(r.source != "semantic_scholar" and key in record_keys(r) for r in work.records):
            return key
        if work is None and not key.startswith("semantic_scholar:"):
            return key
        return s2_hash(key)

    def hops_public(hops: list[dict[str, Any]], only: bool, work_key: str) -> list[dict[str, Any]]:
        return [{**h, "url": _sanitise_url(h.get("url"), only, s2_ids),
                 "location": _sanitise_url(h.get("location"), only, s2_ids),
                 "rid": h["rid"].replace(work_key, key_public(work_key))} for h in hops]

    def clean_work(w: dict[str, Any]) -> dict[str, Any]:
        attempts = []
        for a in w["attempts"]:
            only = _followed_s2_only(w, a.get("url_index"))  # MUTATION-POINT: probe-s2-only
            attempts.append({**a, "hops": hops_public(a.get("hops", []), only, w["work"])})
        return {"work": key_public(w["work"]), "topics": w["topics"], "sources": w["sources"],
                "result": w["result"], "host_stopped": w.get("host_stopped"),
                "candidates": [{"url": _sanitise_url(c["url"], only, s2_ids), "kind": c["kind"],
                                "supplied_by": c["supplied_by"], "url_check": c["url_check"],
                                "licences": {k: v for k, v in c["licences"].items() if k != "semantic_scholar"}}
                               for c, only in zip(w["candidates"], _s2_only_flags(w), strict=True)],
                "attempts": attempts}

    def clean_download(d: dict[str, Any]) -> dict[str, Any]:
        only = _followed_s2_only(probe_by_work.get(d["work"]), d.get("url_index"))  # MUTATION-POINT: download-s2-only
        return {**{k: v for k, v in d.items() if k != "hops"}, "work": key_public(d["work"]),
                "hops": hops_public(d.get("hops", []), only, d["work"])}

    return {"protocol_sha256": digest,
            "works": [clean_work(w) for w in run["probe_works"]],
            "downloads": [clean_download(d) for d in run["downloads"]],
            "requests": [{**m, "rid": _rid_public(m["rid"], key_public)} for m in merged(run["pdf"])]}


def s2_only_urls(run: dict[str, list[dict[str, Any]]]) -> set[str]:
    """Every URL string only S2 supplied, or that a chain from one met; leakcheck's needles.

    A URL that some other source also supplied, or that a public chain met, is public.
    """
    s2: set[str] = set()
    public: set[str] = set()
    probe_by_work = {w["work"]: w for w in run["probe_works"]}

    def chain_urls(hops: list[dict[str, Any]]) -> set[str]:
        return {u for h in hops for u in (h.get("url"), h.get("location")) if u}

    for w in run["probe_works"]:
        for c, only in zip(w["candidates"], _s2_only_flags(w), strict=True):
            (s2 if only else public).add(c["url"])
        for a in w["attempts"]:
            (s2 if _followed_s2_only(w, a.get("url_index")) else public).update(chain_urls(a.get("hops", [])))
    for d in run["downloads"]:
        followed = _followed_s2_only(probe_by_work.get(d["work"]), d.get("url_index"))
        (s2 if followed else public).update(chain_urls(d.get("hops", [])))
    return s2 - public


def leaked_urls(text: str, needles: set[str], *, quoted: bool) -> list[str]:
    """Needles present in `text`; in JSON, matched as whole string values."""
    return sorted(u for u in needles if (json.dumps(u, ensure_ascii=False) if quoted else u) in text)


def _requests_doc(run: dict[str, list[dict[str, Any]]], digest: str, s2_ids: set[str]) -> dict[str, Any]:
    """17_discovery_requests.json: every API request, joined to its outcome, S2 ids hashed."""
    api_rows = []
    outcomes = {r["rid"]: r for r in run["api"] if r.get("event") == "outcome"}
    for r in run["api"]:
        if r.get("event") == "sending":
            row = {**r, **{k: v for k, v in outcomes.get(r["rid"], {}).items() if k != "event"}}
            row["event"] = "request"
        elif r.get("event") in ("outcome",):
            continue
        else:
            row = dict(r)
        if "url" in row:
            row["url"] = _sanitise_url(row["url"], False, s2_ids)
        for k in ("t_start", "t_end"):
            if isinstance(row.get(k), float):
                row[f"{k}_iso"] = iso(row[k])
        api_rows.append(row)
    return {"protocol_sha256": digest, "s2_mode": "keyless",
            "note": "S2 paper ids are replaced by s2h: hashes (D116: S2 data stays local). Each run "
                    "line records the protocol hash that run used.",
            "records": api_rows}


def _verify_frozen_pools(entries: list[dict[str, Any]], path: Path = POOLS_PATH,
                         manifest_path: Path = POOLS_MANIFEST_PATH) -> None:
    """Committed pools still hash to their manifest and hold exactly these entries."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blob = path.read_bytes()
    if hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
        raise SystemExit(f"{path.name} no longer matches its manifest's SHA-256")
    frozen = sorted(e["rid"] for e in json.loads(gzip.decompress(blob))["entries"])
    if frozen != sorted(e["rid"] for e in entries):
        raise SystemExit(f"{path.name} and the run log disagree about which responses it holds")


def finalize_a1() -> int:
    """A1's tracked outputs (D120): its own pools, and the request log with A1 in it.

    The spike's pools and probe log are not touched: the pools are verified against the
    log instead, so A1 cannot alter what D116's run recorded.
    """
    protocol, digest = load_protocol()
    env = load_env()
    secrets = Secrets([env.get("OPENALEX_API_KEY", ""), env.get("S2_API_KEY", "")])
    run = load_run()
    s2_ids = _s2_ids(run["bodies"])
    _verify_frozen_pools([e for e in run["bodies"] if e.get("source") in ("openalex", "arxiv")
                          and e.get("pass") in ASM1_PASSES])
    print(f"{POOLS_PATH.relative_to(ROOT)} verified against the log; not rewritten")

    text = json.dumps(_requests_doc(run, digest, s2_ids), indent=1, ensure_ascii=False, default=str)
    secrets.assert_absent(text, where=str(OUT_REQUESTS))
    OUT_REQUESTS.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT_REQUESTS.relative_to(ROOT))

    entries = sorted((e for e in run["bodies"] if e.get("pass") == "A1"), key=lambda e: e["rid"])
    if not entries or any(e.get("source") != "arxiv" for e in entries):
        raise SystemExit("A1 holds no responses, or holds a response not from arXiv")
    payload = json.dumps({"protocol_sha256": digest, "entries": entries}, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    secrets.assert_absent(payload.decode("utf-8"), where=str(POOLS_A1_PATH))
    with POOLS_A1_PATH.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(payload)
    blob = POOLS_A1_PATH.read_bytes()
    manifest = {
        "dataset": "scholiast-m4-discovery-pools-arxiv-a1",
        "decision": "D120",
        "protocol_sha256": digest,
        "file": POOLS_A1_PATH.name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "entries": len(entries),
        "by_construction_and_phrasing": dict(Counter(f"{e['construction']}/{e['phrasing']}" for e in entries)),
        "recorded_from": min(e["received_at"] for e in entries),
        "recorded_to": max(e["received_at"] for e in entries),
        "licence": "arXiv metadata CC0",
        "request": protocol["sources"]["arxiv"]["search"],
        "query_construction": protocol["query_construction"]["arxiv"],
        "addendum": protocol["addenda"]["A1"],
        "caveat": protocol["pools_caveat"],
    }
    POOLS_A1_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    reloaded = json.loads(gzip.decompress(POOLS_A1_PATH.read_bytes()))
    assert hashlib.sha256(POOLS_A1_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(reloaded["entries"]) == len(entries)
    bad = [e["rid"] for e in reloaded["entries"] if e["status"] == 200 and parse("arxiv", e["body"]).error]
    print(f"wrote {POOLS_A1_PATH.relative_to(ROOT)} ({len(blob)} bytes, {len(entries)} entries); "
          f"reload verified; unparseable 200s: {bad or 'none'}")
    return 0


def finalize_a2() -> int:
    """A2's tracked outputs (D125): its own pools, and the request log with A2 in it.

    The spike's pools and A1's are verified against the log and not rewritten: a rerun
    of finalize-a1 now would restamp A1's manifest with this protocol's hash.
    """
    protocol, digest = load_protocol()
    env = load_env()
    secrets = Secrets([env.get("OPENALEX_API_KEY", ""), env.get("S2_API_KEY", "")])
    run = load_run()
    s2_ids = _s2_ids(run["bodies"])
    _verify_frozen_pools([e for e in run["bodies"] if e.get("source") in ("openalex", "arxiv")
                          and e.get("pass") in ASM1_PASSES])
    _verify_frozen_pools([e for e in run["bodies"] if e.get("pass") == "A1"],
                         POOLS_A1_PATH, POOLS_A1_MANIFEST_PATH)
    print(f"{POOLS_PATH.relative_to(ROOT)} and {POOLS_A1_PATH.relative_to(ROOT)} verified against the "
          "log; not rewritten")

    text = json.dumps(_requests_doc(run, digest, s2_ids), indent=1, ensure_ascii=False, default=str)
    secrets.assert_absent(text, where=str(OUT_REQUESTS))
    OUT_REQUESTS.write_text(text + "\n", encoding="utf-8")
    print("wrote", OUT_REQUESTS.relative_to(ROOT))

    entries = sorted((e for e in run["bodies"] if e.get("pass") == "A2"), key=lambda e: e["rid"])
    if not entries or any(e.get("source") != "openalex" for e in entries):
        raise SystemExit("A2 holds no responses, or holds a response not from OpenAlex")
    payload = json.dumps({"protocol_sha256": digest, "entries": entries}, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    secrets.assert_absent(payload.decode("utf-8"), where=str(POOLS_A2_PATH))
    with POOLS_A2_PATH.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        gz.write(payload)
    blob = POOLS_A2_PATH.read_bytes()
    manifest = {
        "dataset": "scholiast-m4-discovery-pools-openalex-a2",
        "decision": "D125",
        "protocol_sha256": digest,
        "file": POOLS_A2_PATH.name,
        "sha256": hashlib.sha256(blob).hexdigest(),
        "bytes": len(blob),
        "entries": len(entries),
        "by_per_page_and_phrasing": dict(Counter(f"{e['per_page']}/{e['phrasing']}" for e in entries)),
        "recorded_from": min(e["received_at"] for e in entries),
        "recorded_to": max(e["received_at"] for e in entries),
        "licence": "OpenAlex data CC0",
        "request": {"built_by": "scholiast.infrastructure.discovery.openalex.OpenAlexSource.request",
                    "adapter_version": ADAPTER_VERSION, "select": SELECT, "per_page": list(A2_SIZES)},
        "addendum": protocol["addenda"]["A2"],
        "caveat": protocol["pools_caveat"],
    }
    POOLS_A2_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    reloaded = json.loads(gzip.decompress(POOLS_A2_PATH.read_bytes()))
    assert hashlib.sha256(POOLS_A2_PATH.read_bytes()).hexdigest() == manifest["sha256"]
    assert len(reloaded["entries"]) == len(entries)
    bad = [e["rid"] for e in reloaded["entries"] if e["status"] == 200 and parse("openalex", e["body"]).error]
    print(f"wrote {POOLS_A2_PATH.relative_to(ROOT)} ({len(blob)} bytes, {len(entries)} entries); "
          f"reload verified; unparseable 200s: {bad or 'none'}")
    return 0


def _rid_public(rid: str, key_public: Any) -> str:
    parts = rid.split("/")
    if len(parts) >= 3 and parts[0] in ("P3", "P4"):
        # the work key itself may contain '/', e.g. an old-style arXiv id or a DOI
        head, tail = parts[0], parts[1:]
        suffix = [p for p in tail if re.fullmatch(r"u\d+|h\d+", p)]
        key = "/".join(p for p in tail if not re.fullmatch(r"u\d+|h\d+", p))
        return "/".join([head, key_public(key), *suffix])
    return rid


def leakcheck() -> int:
    env = load_env()
    secret_values = [v for v in (env.get("OPENALEX_API_KEY", ""), env.get("S2_API_KEY", "")) if len(v) >= 8]
    run = load_run()
    s2_ids = _s2_ids(run["bodies"])
    s2_urls = s2_only_urls(run)
    tracked_outputs = [OUT_REQUESTS, OUT_PROBE, OUT_ANALYSIS, RESULTS_DIR / "19_selftest.json",
                       POOLS_MANIFEST_PATH, FINDINGS, ROOT / "spike" / "m4-protocol.json",
                       ROOT / "spike" / ".env.example", ROOT / "spike" / "README.md",
                       POOLS_A1_MANIFEST_PATH, POOLS_A2_MANIFEST_PATH]
    texts = {str(p.relative_to(ROOT)): p.read_text(encoding="utf-8") for p in tracked_outputs if p.exists()}
    for pools in (POOLS_PATH, POOLS_A1_PATH, POOLS_A2_PATH):
        if pools.exists():
            texts[str(pools.relative_to(ROOT))] = gzip.decompress(pools.read_bytes()).decode("utf-8")
    for p in sorted((ROOT / "spike" / "scripts").glob("m4_*.py")) + sorted((ROOT / "spike" / "scripts").glob("1[789]_*.py")):
        texts[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8")
    problems = []
    for name, text in texts.items():
        if any(v in text for v in secret_values):
            problems.append(f"{name}: contains a key value")
        leaked = [pid for pid in s2_ids if pid in text]
        if leaked:
            problems.append(f"{name}: contains {len(leaked)} S2 paper id(s)")
        # The pools hold OpenAlex and arXiv bodies only (checked by source below);
        # scripts hold no run data. Everything else is searched for S2-only URLs.
        if not name.endswith((".gz", ".py")):
            urls = leaked_urls(text, s2_urls, quoted=name.endswith(".json"))
            if urls:
                problems.append(f"{name}: contains {len(urls)} URL(s) only S2 supplied")
        if name.endswith(".json") or name.endswith(".gz"):
            if re.search(r"Bearer\s+[A-Za-z0-9._\-]{8,}", text):
                problems.append(f"{name}: contains a bearer token")
            if re.search(r"api_key=", text, re.IGNORECASE):
                problems.append(f"{name}: contains an api_key parameter")
    for pools in (POOLS_PATH, POOLS_A1_PATH, POOLS_A2_PATH):
        if '"source": "semantic_scholar"' in texts.get(str(pools.relative_to(ROOT)), ""):
            problems.append(f"{pools.name} contains Semantic Scholar entries")

    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=ROOT,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    paths = [line[3:] for line in status.stdout.splitlines()]
    for path in paths:
        low = path.lower()
        name = low.rsplit("/", 1)[-1]
        # .env.example is the tracked template; it is scanned for key values above instead.
        env_file = name == ".env" or (name.startswith(".env.") and name != ".env.example")
        if low.endswith(".pdf") or low.startswith("var/") or env_file:
            problems.append(f"git would see {path}")
    ignored = subprocess.run(["git", "check-ignore", "-v", "var/spike/pdf/x.pdf", "var/spike/run/bodies.jsonl",
                              "spike/.env"], cwd=ROOT, capture_output=True, text=True, check=False)
    if len(ignored.stdout.splitlines()) != 3:
        problems.append(f"git check-ignore did not cover all local paths: {ignored.stdout!r}")
    print(f"leakcheck: {len(texts)} files scanned, {len(s2_ids)} S2 ids and {len(s2_urls)} S2-only URLs checked, "
          f"{len(secret_values)} secret value(s) checked, {len(paths)} changed/untracked paths")
    print("git check-ignore:\n" + ignored.stdout.rstrip())
    for p in problems:
        print("LEAK:", p)
    print("leakcheck:", "CLEAN" if not problems else f"{len(problems)} problem(s)")
    return 0 if not problems else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["selftest", "check", "analyse", "finalize", "finalize-a1",
                                            "finalize-a2", "leakcheck"])
    command = parser.parse_args().command
    return {"selftest": selftest, "check": check, "analyse": analyse, "finalize": finalize,
            "finalize-a1": finalize_a1, "finalize-a2": finalize_a2, "leakcheck": leakcheck}[command]()


if __name__ == "__main__":
    sys.exit(main())
