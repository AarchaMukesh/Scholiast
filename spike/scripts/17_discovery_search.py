"""M4 discovery spike, passes P1, P2 and P5 (D116, spike/m4-protocol.json).

    .venv\\Scripts\\python.exe spike\\scripts\\17_discovery_search.py plan
    .venv\\Scripts\\python.exe spike\\scripts\\17_discovery_search.py session1
    .venv\\Scripts\\python.exe spike\\scripts\\17_discovery_search.py session2 [--wait]
    .venv\\Scripts\\python.exe spike\\scripts\\17_discovery_search.py addendum-a1

`plan` prints every planned API request and sends nothing. `session1` runs P1 (the
20 search queries, three sources at once per query), then DEV-2's repeats of any
OpenAlex search refused for a wildcard before the amendment (D117), and then P2 (one
anchor check per source, then single-record timing). `session2` runs P5, and refuses
to start until 6 hours after P1's last request unless told to wait for it.
`addendum-a1` runs A1 (D120): the 20 P1 queries against arXiv only, each joined with
AND and with OR.

Everything is resumable without repeating a request: the run log under
var/spike/run/ records a `sending` line before every request, and a rerun skips
any request id already there. S2 runs keyless, at the owner's instruction, even if
an S2 key is present in spike/.env.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from dataclasses import dataclass
from typing import Any

import httpx
from m4_common import (
    API_HOST,
    RUN_DIR,
    RunLog,
    Secrets,
    address_allowed,
    api_budget_keys,
    arxiv_query,
    iso,
    load_env,
    load_protocol,
    openalex_query,
    read_jsonl,
    resolve,
    s2_query,
    wall,
)
from m4_http import Ctx, api_request, build_ctx
from m4_parse import parse

SOURCES = ("openalex", "semantic_scholar", "arxiv")
SESSION_GAP_S = 6 * 3600.0
#: How OpenAlex words the refusal DEV-2 answers (D117).
WILDCARD_REFUSAL = "Wildcards (* or ?)"


@dataclass(frozen=True)
class Step:
    query_id: str
    topic: str
    phrasing: str
    text: str
    order: tuple[str, ...]


def plan(protocol: dict[str, Any], pass_id: str) -> list[Step]:
    """The pre-registered query order for P1 or P5, shuffled with a named seed stream."""
    seed = protocol["seed"]
    rng = random.Random(f"{seed}/{pass_id}")
    phrasings = ("keyword", "natural_language") if pass_id == "P1" else ("reversed_keyword",)
    steps = [
        (topic["id"], phrasing, topic[phrasing])
        for topic in protocol["topics"]
        for phrasing in phrasings
    ]
    rng.shuffle(steps)
    out = []
    for topic_id, phrasing, text in steps:
        order = list(SOURCES)
        rng.shuffle(order)
        out.append(Step(f"{topic_id}-{phrasing}", topic_id, phrasing, text, tuple(order)))
    return out


@dataclass(frozen=True)
class A1Step:
    query_id: str
    topic: str
    phrasing: str
    text: str
    joins: tuple[str, ...]


def a1_plan(protocol: dict[str, Any]) -> list[A1Step]:
    """Addendum A1's order (D120): queries shuffled, and AND/OR shuffled within each."""
    rng = random.Random(f"{protocol['seed']}/A1")
    steps = [(t["id"], ph, t[ph]) for t in protocol["topics"] for ph in ("keyword", "natural_language")]
    rng.shuffle(steps)
    out = []
    for topic_id, phrasing, text in steps:
        joins = ["AND", "OR"]
        rng.shuffle(joins)
        out.append(A1Step(f"{topic_id}-{phrasing}", topic_id, phrasing, text, tuple(joins)))
    return out


def a1_request(protocol: dict[str, Any], step: A1Step, join: str) -> dict[str, Any]:
    spec = protocol["sources"]["arxiv"]["search"]
    params = dict(spec["params"])
    params["search_query"] = arxiv_query(step.text, protocol["query_construction"]["arxiv_stopwords"], join)
    return {"method": spec["method"], "url": spec["url"], "params": params}


def search_request(protocol: dict[str, Any], source: str, text: str, select: str) -> dict[str, Any]:
    spec = protocol["sources"][source]["search"]
    params = dict(spec["params"])
    if source == "openalex":
        params["search"] = openalex_query(text)
        params["select"] = select
    elif source == "semantic_scholar":
        params["query"] = s2_query(text)
    else:
        params["search_query"] = arxiv_query(text, protocol["query_construction"]["arxiv_stopwords"])
    return {"method": spec["method"], "url": spec["url"], "params": params}


def wildcard_refusals(bodies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """P1 OpenAlex searches refused for a wildcard: what DEV-2 repeats (D117)."""
    return [
        b for b in bodies
        if b.get("pass") == "P1" and b.get("source") == "openalex" and b.get("kind") == "search"
        and b.get("status") == 400 and WILDCARD_REFUSAL in b.get("body", "")
        and not b["rid"].endswith("/DEV-2")
    ]


class State:
    """What must survive a resume: DEV-1 and the reduced OpenAlex select list."""

    def __init__(self, protocol: dict[str, Any], records: list[dict[str, Any]],
                 bodies: list[dict[str, Any]]) -> None:
        full = protocol["sources"]["openalex"]["search"]["params"]["select"]
        self.full_select = full
        self.reduced_select = ",".join(f for f in full.split(",") if f != "relevance_score")
        self.dev1 = any(r.get("event") == "deviation" and r.get("id") == "DEV-1" for r in records)
        # DEV-2: a search refused for a wildcard never had its select list judged, so
        # DEV-1 still belongs to the first search that was not.
        refused = {b["rid"] for b in wildcard_refusals(bodies)}
        self.openalex_searched = any(
            r.get("event") == "sending" and r.get("source") == "openalex" and r.get("kind") == "search"
            and r["rid"] not in refused
            for r in records
        )

    @property
    def select(self) -> str:
        return self.reduced_select if self.dev1 else self.full_select


def make_clients(protocol: dict[str, Any], openalex_key: str) -> dict[str, httpx.AsyncClient]:
    ua = protocol["http_safety"]["headers"]["User-Agent"]
    t = protocol["timeouts_s"]
    timeout = httpx.Timeout(float(t["api_total"]), connect=float(t["api_connect"]))

    def client(extra: dict[str, str], limits: httpx.Limits | None = None) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": ua, **extra},
            limits=limits or httpx.Limits(max_connections=2),
        )

    return {
        # D116: the key travels in a header, never in a URL a log could record.
        "openalex": client({"Authorization": f"Bearer {openalex_key}"}),
        "semantic_scholar": client({}),
        "arxiv": client({}, httpx.Limits(max_connections=1)),
    }


async def preflight(ctx: Ctx) -> None:
    """Resolve the three API hosts once; refuse to start if any is not public."""
    for host in API_HOST.values():
        addresses = await resolve(host)
        allowed = bool(addresses) and all(address_allowed(a) for a in addresses)
        ctx.log.append({"event": "preflight", "host": host, "addresses": addresses,
                        "allowed": allowed, "at": iso(wall())})
        if not allowed:
            raise SystemExit(f"{host} resolved to a non-public address; nothing sent")


def store_body(bodies: RunLog, record: dict[str, Any], body: bytes | None) -> None:
    if body is None:
        return
    bodies.append(
        {
            "rid": record["rid"],
            "source": record["source"],
            "pass": record["pass"],
            "kind": record["kind"],
            "query_id": record.get("query_id"),
            "topic": record.get("topic"),
            "phrasing": record.get("phrasing"),
            "construction": record.get("construction"),
            "status": record["status"],
            "received_at": iso(record["t_end"]),
            "url": record.get("url"),
            "body": body.decode("utf-8", errors="replace"),
        }
    )


async def run_search(ctx: Ctx, clients: dict[str, httpx.AsyncClient], bodies: RunLog,
                     state: State, step: Step, source: str, pass_id: str, session: int) -> None:
    rid = f"{pass_id}/{step.query_id}/{source}"
    meta = {"pass": pass_id, "session": session, "kind": "search", "source": source,
            "query_id": step.query_id, "topic": step.topic, "phrasing": step.phrasing}
    first_openalex = source == "openalex" and not state.openalex_searched
    req = search_request(ctx.protocol, source, step.text, state.select)
    if source == "openalex":
        state.openalex_searched = True
    result = await api_request(
        ctx, clients[source], rid=rid, host=API_HOST[source],
        budget_keys=api_budget_keys(source, "search"), method=req["method"], url=req["url"],
        params=req["params"], meta=meta,
        total_timeout=float(ctx.protocol["timeouts_s"]["api_total"]),
    )
    if result is None:
        return
    record, body = result
    store_body(bodies, record, body)
    text = (body or b"").decode("utf-8", errors="replace").lower()
    if first_openalex and record["status"] == 400 and ("select" in text or "relevance_score" in text):
        # DEV-1, the protocol's one pre-registered repeat.
        state.dev1 = True
        ctx.log.append({"event": "deviation", "id": "DEV-1", "rid": rid, "at": iso(wall()),
                        "detail": "first OpenAlex search refused the select list; "
                                  "repeating once without relevance_score"})
        req = search_request(ctx.protocol, source, step.text, state.select)
        retry = await api_request(
            ctx, clients[source], rid=f"{rid}/DEV-1", host=API_HOST[source],
            budget_keys=api_budget_keys(source, "search"), method=req["method"], url=req["url"],
            params=req["params"], meta={**meta, "deviation": "DEV-1"},
            total_timeout=float(ctx.protocol["timeouts_s"]["api_total"]),
        )
        if retry is not None:
            store_body(bodies, retry[0], retry[1])


async def run_steps(ctx: Ctx, clients: dict[str, httpx.AsyncClient], bodies: RunLog,
                    state: State, steps: list[Step], pass_id: str, session: int) -> None:
    for number, step in enumerate(steps, 1):
        pending = [s for s in step.order if f"{pass_id}/{step.query_id}/{s}" not in ctx.done]
        if not pending:
            continue
        # A query's requests start together, once every source's gap has elapsed.
        start_at = max(ctx.pacer.earliest(API_HOST[s]) for s in pending)
        while (delay := start_at - wall()) > 0:
            await asyncio.sleep(delay)
        await asyncio.gather(
            *(run_search(ctx, clients, bodies, state, step, s, pass_id, session) for s in pending)
        )
        print(f"{pass_id} {number:2d}/{len(steps)} {step.query_id}", flush=True)


async def run_dev2_repeats(ctx: Ctx, clients: dict[str, httpx.AsyncClient], bodies: RunLog,
                           state: State, steps: list[Step]) -> None:
    """DEV-2 (D117): each wildcard-refused P1 OpenAlex search, once, with the amended query."""
    refused = wildcard_refusals(bodies.records())
    if not refused:
        return
    if not any(r.get("event") == "deviation" and r.get("id") == "DEV-2" for r in ctx.log.records()):
        ctx.log.append({"event": "deviation", "id": "DEV-2", "at": iso(wall()),
                        "refused": [b["rid"] for b in refused],
                        "detail": "OpenAlex refused these searches for a wildcard ('?'); each is repeated "
                                  "once with '?' and '*' replaced by spaces (protocol amended, D117)"})
    by_query = {s.query_id: s for s in steps}
    for entry in refused:
        step = by_query[entry["query_id"]]
        req = search_request(ctx.protocol, "openalex", step.text, state.select)
        result = await api_request(
            ctx, clients["openalex"], rid=f"{entry['rid']}/DEV-2", host=API_HOST["openalex"],
            budget_keys=api_budget_keys("openalex", "search"), method=req["method"], url=req["url"],
            params=req["params"],
            meta={"pass": "P1", "session": 1, "kind": "search", "source": "openalex",
                  "query_id": step.query_id, "topic": step.topic, "phrasing": step.phrasing,
                  "deviation": "DEV-2"},
            total_timeout=float(ctx.protocol["timeouts_s"]["api_total"]),
        )
        if result is not None:
            store_body(bodies, result[0], result[1])
            print(f"P1 DEV-2 {step.query_id}: {result[0].get('outcome')}", flush=True)


#: Creating this file pauses a live pass between requests, never during one. Killing
#: the process instead can cut a request off mid-flight and leave it logged as sent
#: with no outcome, which the no-repeats rule then makes permanent (it happened once,
#: to A1/t08-natural_language/OR).
STOP_FILE = RUN_DIR / "STOP"


async def run_a1(ctx: Ctx, client: httpx.AsyncClient, bodies: RunLog) -> None:
    """Addendum A1 (D120): one arXiv request per query and join, paced as P1's were."""
    total = float(ctx.protocol["timeouts_s"]["api_total"])
    for number, step in enumerate(a1_plan(ctx.protocol), 1):
        for join in step.joins:
            if f"A1/{step.query_id}/{join}" in ctx.done:
                continue
            # Wait out the pacing gap first, then look for the stop file, so a pause
            # written during the gap stops the very next request. Checked before the
            # wait instead, one request slipped through on 2026-10-02.
            while (delay := ctx.pacer.earliest(API_HOST["arxiv"]) - wall()) > 0:
                await asyncio.sleep(delay)
            if STOP_FILE.exists():
                print(f"{STOP_FILE} exists: paused before A1/{step.query_id}/{join}", flush=True)
                return
            req = a1_request(ctx.protocol, step, join)
            result = await api_request(
                ctx, client, rid=f"A1/{step.query_id}/{join}", host=API_HOST["arxiv"],
                budget_keys=[("api", "arxiv_api", "a1")], method=req["method"], url=req["url"],
                params=req["params"],
                meta={"pass": "A1", "session": 3, "kind": "search", "source": "arxiv",
                      "query_id": step.query_id, "topic": step.topic, "phrasing": step.phrasing,
                      "construction": join},
                total_timeout=total,
            )
            if result is not None:
                store_body(bodies, result[0], result[1])
        print(f"A1 {number:2d}/20 {step.query_id}", flush=True)


def keyword_ids(bodies: list[dict[str, Any]], source: str) -> list[str]:
    ids: set[str] = set()
    for entry in bodies:
        if (entry.get("pass") == "P1" and entry.get("phrasing") == "keyword"
                and entry.get("source") == source and entry.get("status") == 200):
            for rec in parse(source, entry["body"]).records:
                if rec.source_id:
                    ids.add(rec.source_id)
    return sorted(ids)


async def run_lookups(ctx: Ctx, clients: dict[str, httpx.AsyncClient], bodies: RunLog,
                      state: State) -> None:
    protocol = ctx.protocol
    anchor_ids = [a["arxiv_id"] for t in protocol["topics"] for a in t["anchors"]]
    stored = bodies.records()
    total = float(protocol["timeouts_s"]["api_total"])

    async def lookup(source: str, rid: str, method: str, url: str, params: dict[str, Any] | None,
                     json_body: Any, kind_detail: str) -> None:
        meta = {"pass": "P2", "session": 1, "kind": "lookup", "source": source,
                "lookup": kind_detail}
        result = await api_request(
            ctx, clients[source], rid=rid, host=API_HOST[source],
            budget_keys=api_budget_keys(source, "lookup"), method=method, url=url,
            params=params, json_body=json_body, meta=meta, total_timeout=total,
        )
        if result is not None:
            store_body(bodies, result[0], result[1])

    def sample(source: str, k: int) -> list[str]:
        ids = keyword_ids(stored, source)
        rng = random.Random(f"{protocol['seed']}/P2/{source}")
        return rng.sample(ids, min(k, len(ids)))

    async def openalex_part() -> None:
        spec = protocol["sources"]["openalex"]
        dois = "|".join(f"https://doi.org/10.48550/arxiv.{i}" for i in anchor_ids)
        await lookup("openalex", "P2/anchor/openalex", "GET", spec["anchor_check"]["url"],
                     {"filter": f"doi:{dois}", "per_page": 50, "select": state.reduced_select},
                     None, "anchor_check")
        for n, work in enumerate(sample("openalex", spec["single_record"]["count"])):
            await lookup("openalex", f"P2/single/openalex/{n:02d}", "GET",
                         f"https://api.openalex.org/works/{work}", None, None, "single_record")

    async def s2_part() -> None:
        spec = protocol["sources"]["semantic_scholar"]
        fields = spec["search"]["params"]["fields"]
        await lookup("semantic_scholar", "P2/anchor/semantic_scholar", "POST",
                     spec["anchor_check"]["url"], {"fields": fields},
                     {"ids": [f"arXiv:{i}" for i in anchor_ids]}, "anchor_check")
        for n, paper in enumerate(sample("semantic_scholar", spec["single_record"]["count"])):
            await lookup("semantic_scholar", f"P2/single/semantic_scholar/{n:02d}", "GET",
                         f"https://api.semanticscholar.org/graph/v1/paper/{paper}",
                         {"fields": fields}, None, "single_record")

    async def arxiv_part() -> None:
        spec = protocol["sources"]["arxiv"]
        await lookup("arxiv", "P2/anchor/arxiv", "GET", spec["anchor_check"]["url"],
                     {"id_list": ",".join(anchor_ids), "max_results": 20}, None, "anchor_check")
        for n, paper in enumerate(sample("arxiv", spec["single_record"]["count"])):
            await lookup("arxiv", f"P2/single/arxiv/{n:02d}", "GET", spec["single_record"]["url"],
                         {"id_list": paper}, None, "single_record")

    await asyncio.gather(openalex_part(), s2_part(), arxiv_part())
    print("P2 done", flush=True)


async def main_async(args: argparse.Namespace) -> int:
    protocol, digest = load_protocol()
    if args.command == "plan":
        state = State(protocol, [], [])
        for pass_id in ("P1", "P5"):
            for step in plan(protocol, pass_id):
                for source in step.order:
                    req = search_request(protocol, source, step.text, state.select)
                    print(pass_id, step.query_id, source, httpx.URL(req["url"], params=req["params"]))
        for a1 in a1_plan(protocol):
            for join in a1.joins:
                req = a1_request(protocol, a1, join)
                print("A1", a1.query_id, join, httpx.URL(req["url"], params=req["params"]))
        return 0

    env = load_env()
    key = env.get("OPENALEX_API_KEY", "")
    if not key and args.command != "addendum-a1":  # A1 sends nothing to OpenAlex
        print("OPENALEX_API_KEY is missing from spike/.env; nothing was sent.", file=sys.stderr)
        return 2
    secrets = Secrets([key, env.get("S2_API_KEY", "")])
    log = RunLog("api", secrets)
    bodies = RunLog("bodies", secrets)
    session = {"session1": 1, "session2": 2, "addendum-a1": 3}[args.command]
    ctx = build_ctx(protocol, log, session=session)
    state = State(protocol, log.records(), read_jsonl(bodies.path))
    log.append({"event": "run", "command": args.command, "protocol_sha256": digest,
                "s2_mode": "keyless", "at": iso(wall())})

    if args.command == "session2":
        planned_p1 = {f"P1/{s.query_id}/{src}" for s in plan(protocol, "P1") for src in s.order}
        missing = planned_p1 - ctx.done
        p1_ends = [r["t_end"] for r in log.records()
                   if r.get("event") == "outcome" and r.get("pass") == "P1"]
        if missing or not p1_ends:
            print(f"P1 is not complete ({len(missing)} planned requests missing); "
                  "run session1 first.", file=sys.stderr)
            return 2
        earliest = max(p1_ends) + SESSION_GAP_S
        if wall() < earliest:
            if not args.wait:
                print(f"P5 may not start before {iso(earliest)}.", file=sys.stderr)
                return 2
            print(f"waiting until {iso(earliest)} for P5", flush=True)
            while (delay := earliest - wall()) > 0:
                await asyncio.sleep(min(delay, 300.0))

    await preflight(ctx)
    clients = make_clients(protocol, key)
    try:
        if args.command == "session1":
            steps = plan(protocol, "P1")
            await run_steps(ctx, clients, bodies, state, steps, "P1", 1)
            await run_dev2_repeats(ctx, clients, bodies, state, steps)
            await run_lookups(ctx, clients, bodies, state)
        elif args.command == "addendum-a1":
            await run_a1(ctx, clients["arxiv"], bodies)
        else:
            await run_steps(ctx, clients, bodies, state, plan(protocol, "P5"), "P5", 2)
    finally:
        for client in clients.values():
            await client.aclose()
    for event in ctx.events:
        print("event:", event)
    log.append({"event": "run_end", "command": args.command, "at": iso(wall()),
                "stopped_hosts": ctx.health.stopped})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "session1", "session2", "addendum-a1"])
    parser.add_argument("--wait", action="store_true", help="session2: wait for the 6 h gap")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
