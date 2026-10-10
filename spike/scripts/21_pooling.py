"""Addendum A5 (D161, D162): fresh against pooled connections to each source, live.

    .venv\\Scripts\\python.exe spike\\scripts\\21_pooling.py plan
    .venv\\Scripts\\python.exe spike\\scripts\\21_pooling.py addendum-a5
    .venv\\Scripts\\python.exe spike\\scripts\\21_pooling.py dry-run [--scale 0.1]

`plan` prints the schedule and sends nothing. `addendum-a5` runs A5 against the live
sources under the approved protocol hash and writes spike/results/21_pooling.json.
`dry-run` runs the same schedule against a local keep-alive stub serving A3's recorded
t03 responses, with a placeholder key and no approval needed, because it asks no third
party; `--scale` shrinks the gaps for it, and only for it. Every request's record is
written to var/spike/run/ as it completes, so a crash or a standby loses only the
request in flight.

Each request is built by the production adapter; the arms differ only in the client
that sends it: a new one per request (fresh, as shipped) or one per source for the
whole condition (pooled). Both are cut at 2.0 s from send to last body byte, as the
fan-out cuts them. httpx's trace extension times connect, TLS, headers and body, and
a request with no connect event reused a pooled connection.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
from m4_common import (
    RESULTS_DIR,
    ROOT,
    RUN_DIR,
    Secrets,
    load_env,
    load_protocol,
    nearest_rank,
    wilson,
)

from scholiast.infrastructure.discovery.arxiv import ArxivSource, parse_feed
from scholiast.infrastructure.discovery.openalex import OpenAlexSource, parse_works
from scholiast.infrastructure.tls import tls_context

PHRASINGS = ("keyword", "natural_language", "reversed_keyword")
SOURCES = ("openalex", "arxiv")
DEADLINE_S = 2.0
KEEPALIVE_S = 120.0
#: Seconds between a lane's pooled requests (A5 "conditions"). arXiv cannot run at 7 s.
GAPS = {"short": {"openalex": 7.0, "arxiv": 15.0}, "long": {"openalex": 90.0, "arxiv": 90.0}}
#: Pacing as P1: OpenAlex from the previous request's start, arXiv from the previous response's end.
PACING = {"openalex": ("start", 1.0), "arxiv": ("end", 3.0)}
BUDGET_KEY = {"openalex": "openalex_searches", "arxiv": "arxiv_queries"}
RESULTS_PATH = RESULTS_DIR / "21_pooling.json"
A3_POOLS = ROOT / "data" / "eval" / "discovery-pools-a3.json.gz"
BOOTSTRAP = 10_000


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


def a5_order(protocol: dict[str, Any]) -> list[Query]:
    """A5's order: the 30 queries shuffled once with stream '<seed>/A5'."""
    rng = random.Random(f"{protocol['seed']}/A5")
    queries = [Query(f"{t['id']}-{ph}", t[ph]) for t in protocol["topics"] for ph in PHRASINGS]
    rng.shuffle(queries)
    return queries


def slots(index: int, gap: float) -> dict[str, float]:
    """Offsets from the lane's start: pooled in the middle slot, fresh first or last by parity."""
    third = gap / 3.0
    base = index * gap
    return {"pooled": base + third, "fresh": base + (0.0 if index % 2 == 0 else 2 * third)}


# -- one request ------------------------------------------------------------------


class Run:
    def __init__(self, protocol: dict[str, Any], digest: str, key: str, bases: dict[str, str] | None,
                 scale: float, log_path: Path) -> None:
        self.protocol, self.digest, self.key, self.bases, self.scale = protocol, digest, key, bases, scale
        self.log_path = log_path
        self.budget = dict(protocol["addenda"]["A5"]["budget"]["max"])
        self.sent = {source: 0 for source in SOURCES}
        self.guard = Secrets([load_env().get("OPENALEX_API_KEY", ""), key])
        self.last: dict[str, dict[str, float]] = {source: {} for source in SOURCES}
        self.records: list[dict[str, Any]] = []

    def keep(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self.guard.assert_absent(line, where=str(self.log_path))
        with open(self.log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.records.append(record)

    def request(self, source: str, text: str) -> httpx.Request:
        """What production would send: the adapter's own request, at the shipped page size."""
        if source == "openalex":
            adapter = OpenAlexSource(self.key, base_url=self.bases["openalex"]) if self.bases else OpenAlexSource(self.key)
            return adapter.request(text, limit=25)
        adapter_a = ArxivSource(base_url=self.bases["arxiv"] + "/api/query") if self.bases else ArxivSource()
        return adapter_a.request(text, limit=50)

    async def paced(self, source: str) -> None:
        mark, gap = PACING[source]
        last = self.last[source].get(mark)
        if last is not None:
            while (wait := last + gap * (1.0 if self.bases is None else self.scale) - time.time()) > 0:
                await asyncio.sleep(wait)

    async def send(self, client: httpx.AsyncClient, source: str, query: Query, arm: str, condition: str,
                   first: bool, gap: float | None) -> None:
        record: dict[str, Any] = {"kind": "request", "condition": condition, "source": source, "arm": arm,
                                  "query_id": query.query_id,
                                  "sent_first": first, "actual_gap_s": None if gap is None else round(gap, 3)}
        try:
            request = self.request(source, query.text)
        except ValueError as exc:  # a query this source cannot express: not sent, not counted
            self.keep({**record, "outcome": "refused", "detail": str(exc)[:120]})
            return
        if self.sent[source] >= int(self.budget[BUDGET_KEY[source]]):
            self.keep({**record, "outcome": "budget"})
            return
        events: dict[str, float] = {}

        async def trace(name: str, info: dict[str, Any]) -> None:
            events.setdefault(name, time.perf_counter())

        request.extensions["trace"] = trace
        await self.paced(source)
        self.sent[source] += 1
        record["t_start"] = time.time()
        self.last[source]["start"] = record["t_start"]
        began = time.perf_counter()
        response: httpx.Response | None = None
        try:
            async with asyncio.timeout(DEADLINE_S):
                response = await client.send(request)
            outcome = "ok" if response.status_code == 200 else "status"
            if outcome == "ok":
                try:
                    (parse_works(response.json()) if source == "openalex" else parse_feed(response.content))
                except Exception as exc:  # noqa: BLE001 - any parser refusal is "not ok"
                    outcome, record["detail"] = "unparsed", type(exc).__name__
        except TimeoutError:
            outcome = "cut"
        except httpx.HTTPError as exc:
            outcome, record["detail"] = "error", type(exc).__name__
        ended = time.perf_counter()
        record["t_end"] = time.time()
        self.last[source]["end"] = record["t_end"]

        def at(name: str) -> float | None:
            return events.get(name)

        def span(a: str, b: str) -> float | None:
            x, y = at(a), at(b)
            return round(y - x, 4) if x is not None and y is not None else None

        record.update(
            outcome=outcome,
            status=response.status_code if response is not None else None,
            bytes=len(response.content) if response is not None and outcome != "cut" else None,
            total_s=round(ended - began, 4),
            reused=at("connection.connect_tcp.started") is None and at("http11.send_request_headers.started") is not None,
            connect_s=span("connection.connect_tcp.started", "connection.connect_tcp.complete"),
            tls_s=span("connection.start_tls.started", "connection.start_tls.complete"),
            headers_s=span("http11.send_request_headers.started", "http11.receive_response_headers.complete"),
            body_s=span("http11.receive_response_body.started", "http11.receive_response_body.complete"),
            cost_usd=(response.headers.get("x-ratelimit-cost-usd") if response is not None and source == "openalex"
                      else None),
        )
        self.keep(record)


def fresh_client() -> httpx.AsyncClient:
    """As the production adapters build theirs (D119, D122)."""
    return httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=False, verify=tls_context())


def pooled_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=False, verify=tls_context(),
                             limits=httpx.Limits(keepalive_expiry=KEEPALIVE_S))


async def lane(run: Run, source: str, condition: str, queries: list[Query], pooled: httpx.AsyncClient) -> None:
    gap = GAPS[condition][source] * run.scale
    start = time.time()
    previous_pooled: float | None = None
    for index, query in enumerate(queries):
        at = slots(index, gap)
        for arm in sorted(at, key=lambda name: at[name]):
            # A5: a lane more than one slot late (a stall, a standby) re-anchors, so the
            # requests after it keep the condition's gap instead of firing back to back.
            late = time.time() - (start + at[arm])
            if late > gap / 3.0:
                start += late
                run.keep({"kind": "shift", "condition": condition, "source": source, "query_id": query.query_id,
                          "late_s": round(late, 3), "t": time.time()})
            while (wait := start + at[arm] - time.time()) > 0:
                await asyncio.sleep(wait)
            if arm == "pooled":
                now = time.time()
                actual = None if previous_pooled is None else now - previous_pooled
                previous_pooled = now
                await run.send(pooled, source, query, "pooled", condition, at["pooled"] < at["fresh"], actual)
            else:
                async with fresh_client() as client:
                    await run.send(client, source, query, "fresh", condition, at["fresh"] < at["pooled"], None)
        print(f"{condition:5s} {source:8s} {index + 1:2d}/30 {query.query_id}", flush=True)


async def execute(run: Run) -> None:
    queries = a5_order(run.protocol)
    for condition in ("short", "long"):
        pooled = {source: pooled_client() for source in SOURCES}
        try:
            await asyncio.gather(*(lane(run, source, condition, queries, pooled[source]) for source in SOURCES))
        finally:
            for client in pooled.values():
                await client.aclose()


# -- standby, analysis -----------------------------------------------------------


def standby_entries(since: float, until: float) -> list[float]:
    """Kernel-Power 506 (entering modern standby) between two times, as epoch seconds."""
    script = (
        "Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; Id=506; "
        f"StartTime=[DateTimeOffset]::FromUnixTimeSeconds({int(since) - 5}).LocalDateTime; "
        f"EndTime=[DateTimeOffset]::FromUnixTimeSeconds({int(until) + 5}).LocalDateTime}} -ErrorAction SilentlyContinue | "
        "ForEach-Object { ([DateTimeOffset]$_.TimeCreated).ToUnixTimeMilliseconds() }"
    )
    out = subprocess.run(["powershell.exe", "-NoProfile", "-Command", script], capture_output=True, text=True)
    return [int(x) / 1000 for x in out.stdout.split() if x.strip().isdigit()]


def paired(records: list[dict[str, Any]], source: str, condition: str, seed: str) -> dict[str, Any]:
    by: dict[str, dict[str, bool]] = {}
    for r in records:
        if r["source"] == source and r["condition"] == condition and r["outcome"] not in ("refused", "budget") \
                and not r.get("void"):
            by.setdefault(r["query_id"], {})[r["arm"]] = r["outcome"] == "ok"
    pairs = [(v["pooled"], v["fresh"]) for v in by.values() if "pooled" in v and "fresh" in v]
    diffs = [int(p) - int(f) for p, f in pairs]
    if not diffs:
        return {"n": 0}
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP))
    return {"n": len(diffs), "pooled_ok": sum(p for p, _ in pairs), "fresh_ok": sum(f for _, f in pairs),
            "difference": round(sum(diffs) / len(diffs), 4),
            "interval": [round(boots[int(0.025 * BOOTSTRAP)], 4), round(boots[int(0.975 * BOOTSTRAP) - 1], 4)]}


def describe(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "p50": nearest_rank(values, 50), "p95": nearest_rank(values, 95)}


def analyse(run: Run) -> dict[str, Any]:
    records = [r for r in run.records if r.get("kind") == "request" and r["outcome"] not in ("refused", "budget")]
    sent = [r for r in records if "t_start" in r]
    entries = standby_entries(min(r["t_start"] for r in sent), max(r["t_end"] for r in sent)) if sent else []
    for r in sent:
        r["void"] = any(r["t_start"] <= t <= r["t_end"] for t in entries)
    # A5: a pooled request off its condition's gap takes its pair with it.
    off = set()
    for r in sent:
        if r["arm"] == "pooled" and r["actual_gap_s"] is not None:
            g = GAPS[r["condition"]][r["source"]] * run.scale
            if not 0.5 * g <= r["actual_gap_s"] <= 1.5 * g:
                off.add((r["condition"], r["source"], r["query_id"]))
    for r in sent:
        r["off_condition"] = (r["condition"], r["source"], r["query_id"]) in off
        r["void"] = r["void"] or r["off_condition"]
    seed = run.protocol["seed"]
    out: dict[str, Any] = {"addendum": "A5", "protocol_sha256": run.digest, "standby_entries": entries}
    for condition in ("short", "long"):
        for source in SOURCES:
            mine = [r for r in sent if r["condition"] == condition and r["source"] == source and not r["void"]]
            block: dict[str, Any] = {"paired": paired(sent, source, condition, f"{seed}/A5/bootstrap/{condition}/{source}")}
            for arm in ("fresh", "pooled"):
                a = [r for r in mine if r["arm"] == arm]
                block[arm] = {"completion": wilson(sum(r["outcome"] == "ok" for r in a), len(a)),
                              "reused": sum(bool(r["reused"]) for r in a),
                              "outcomes": {o: sum(r["outcome"] == o for r in a) for o in sorted({r["outcome"] for r in a})}}
            for kind, flag in (("new_connection", False), ("reused_connection", True)):
                done = [r for r in mine if bool(r["reused"]) is flag and r["outcome"] == "ok"]
                block[kind] = {k: describe([r[k] for r in done if r.get(k) is not None])
                               for k in ("connect_s", "tls_s", "headers_s", "body_s", "total_s")}
            block["order"] = {f"sent_{w}": wilson(sum(r["outcome"] == "ok" for r in mine if r["sent_first"] is (w == "first")),
                                                 sum(1 for r in mine if r["sent_first"] is (w == "first")))
                              for w in ("first", "second")}
            out[f"{condition}/{source}"] = block
    stale = [r for r in sent if r["arm"] == "pooled" and r["reused"] and r["outcome"] == "error" and not r["void"]]
    short_oa, short_ax = out["short/openalex"]["paired"], out["short/arxiv"]["paired"]
    conditions = {
        "openalex_short_gain_at_least_0_15_and_interval_above_zero":
            short_oa.get("n", 0) > 0 and short_oa["difference"] >= 0.15 and short_oa["interval"][0] > 0,
        "arxiv_short_not_worse_by_more_than_2":
            short_ax.get("n", 0) > 0 and short_ax["pooled_ok"] >= short_ax["fresh_ok"] - 2,
        "at_most_1_error_on_a_reused_connection": len(stale) <= 1,
    }
    out["verdict"] = {"adopt_pooling": all(conditions.values()), "conditions": conditions,
                      "errors_on_reused_connections": len(stale),
                      "void": sum(1 for r in sent if r["void"] and not r["off_condition"]),
                      "off_condition": sum(1 for r in sent if r["off_condition"]),
                      "schedule_shifts": sum(1 for r in run.records if r.get("kind") == "shift"),
                      "requests_sent": run.sent,
                      "cost_usd": round(sum(float(r["cost_usd"]) for r in sent if r.get("cost_usd")), 4)}
    out["records"] = run.records
    return out


# -- dry-run stub ---------------------------------------------------------------


class Stub:
    """Keep-alive HTTP/1.1 server answering A3's t03 responses after 300 ms, for the dry run."""

    def __init__(self) -> None:
        entries = json.loads(gzip.decompress(A3_POOLS.read_bytes()))["entries"]
        bodies = {e["source"]: e["body"].encode("utf-8") for e in entries if e["query_id"] == "t03-keyword"}

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802
                time.sleep(0.3)
                name = "openalex" if self.path.startswith("/works") else "arxiv" if self.path.startswith("/api/query") else None
                body = bodies[name] if name else b""
                self.send_response(200 if name else 404)
                self.send_header("Content-Type", "application/json" if name == "openalex" else "application/atom+xml")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


# -- entry points ---------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("addendum-a5")
    dry = sub.add_parser("dry-run")
    dry.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    if args.command == "plan":
        protocol, _ = load_protocol()
        for index, query in enumerate(a5_order(protocol)):
            at = slots(index, 1.0)
            print(f"{index:2d} {query.query_id:28s} {'pooled first' if at['pooled'] < at['fresh'] else 'fresh first'}")
        return 0
    if args.command == "dry-run":
        protocol = json.loads((ROOT / "spike" / "m4-protocol.json").read_text(encoding="utf-8"))
        stub = Stub()
        run = Run(protocol, "dry-run", "placeholder-not-a-key", {s: stub.base for s in SOURCES}, args.scale,
                  RUN_DIR / f"a5-dry-{stamp}.jsonl")
        out_path = RUN_DIR / "a5-dry-run.json"
    else:
        protocol, digest = load_protocol()
        key = load_env().get("OPENALEX_API_KEY")
        if not key:
            raise SystemExit("A5 runs OpenAlex keyed, as shipped; no key in spike/.env")
        stub = None
        run = Run(protocol, digest, key, None, 1.0, RUN_DIR / f"a5-{stamp}.jsonl")
        out_path = RESULTS_PATH
    if "A5" not in protocol["addenda"]:
        raise SystemExit("the protocol has no addendum A5")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    try:
        asyncio.run(execute(run))
    finally:
        if stub:
            stub.close()
    results = analyse(run)
    text = json.dumps(results, indent=1, ensure_ascii=False, allow_nan=False)
    run.guard.assert_absent(text, where=str(out_path))
    out_path.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(results["verdict"], indent=1))
    for key_ in ("short/openalex", "short/arxiv", "long/openalex", "long/arxiv"):
        b = results[key_]
        print(key_, "paired", b["paired"], "reused fresh/pooled", b["fresh"]["reused"], b["pooled"]["reused"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
