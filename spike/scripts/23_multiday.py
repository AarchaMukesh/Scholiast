"""Addendum A6-multiday (D164, D165): the sources' completion inside a deadline, on three different days.

    .venv\\Scripts\\python.exe spike\\scripts\\23_multiday.py plan --day N
    .venv\\Scripts\\python.exe spike\\scripts\\23_multiday.py addendum-a6 --day N
    .venv\\Scripts\\python.exe spike\\scripts\\23_multiday.py dry-run [--scale 0.1]

`addendum-a6 --day N` runs day N (1, 2 or 3) against the live sources under the
approved protocol hash, refusing a day out of order, on a calendar day already used,
or within 18 hours of the previous day's start. It rewrites
spike/results/23_multiday.json after each day, and adds the verdict after day 3.
`dry-run` runs all three days back to back against a local stub serving A3's recorded
responses, with a placeholder key and no calendar checks, so the harness is checked
before any live call; it writes only under var/spike/run/.

Every request is built by the production adapter and sent on a new client, as
shipped, timed to its last byte (20 s backstop, as P1) rather than cut, so completion
inside 2.0 s and inside 2.5 s both come from the one measurement. Every record is
written to disk as it completes.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from m4_common import RESULTS_DIR, ROOT, RUN_DIR, Secrets, load_env, load_protocol, nearest_rank, wilson

from scholiast.infrastructure.discovery.arxiv import ArxivSource, parse_feed
from scholiast.infrastructure.discovery.openalex import OpenAlexSource, parse_works
from scholiast.infrastructure.tls import tls_context

PHRASINGS = ("keyword", "natural_language", "reversed_keyword")
CONFIGS = [("as shipped", 25, 2.0), ("longer deadline", 25, 2.5), ("smaller page", 15, 2.0), ("both", 15, 2.5)]
REPORT_DEADLINES = (1.5, 2.0, 2.5, 3.0)
OPENALEX_EVERY_S, ARXIV_EVERY_S = 6.0, 12.0
PACING = {"openalex": ("start", 1.0), "arxiv": ("end", 3.0)}
MIN_HOURS_BETWEEN_DAYS = 18.0
RESULTS_PATH = RESULTS_DIR / "23_multiday.json"
DRY_PATH = RUN_DIR / "a6-multiday-dry-run.json"


def _a5() -> Any:
    """A5's runner, for its keep-alive stub and its standby check (loaded once)."""
    if "a5_runner" not in sys.modules:
        spec = importlib.util.spec_from_file_location("a5_runner", Path(__file__).with_name("21_pooling.py"))
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules["a5_runner"] = module  # dataclasses look their module up while it loads
        spec.loader.exec_module(module)
    return sys.modules["a5_runner"]


def _stub_class() -> Any:
    return _a5().Stub


def day_order(protocol: dict[str, Any], day: int) -> list[tuple[str, str]]:
    rng = random.Random(f"{protocol['seed']}/A6-multiday/day{day}")
    queries = [(f"{t['id']}-{ph}", t[ph]) for t in protocol["topics"] for ph in PHRASINGS]
    rng.shuffle(queries)
    return queries


def openalex_schedule(queries: list[tuple[str, str]]) -> list[tuple[int, str, str, int]]:
    """(pass, query_id, text, page): pass 1 sends even positions at 25, odd at 15; pass 2 the other way."""
    out = []
    for pass_no in (1, 2):
        for i, (qid, text) in enumerate(queries):
            first_size = 25 if i % 2 == 0 else 15
            out.append((pass_no, qid, text, first_size if pass_no == 1 else 40 - first_size))
    return out


class Day:
    def __init__(self, day: int, key: str, bases: dict[str, str] | None, scale: float, log: Path,
                 budget_left: dict[str, int], guard: Secrets) -> None:
        self.day, self.key, self.bases, self.scale, self.log = day, key, bases, scale, log
        self.left, self.guard = budget_left, guard
        self.last: dict[str, dict[str, float]] = {"openalex": {}, "arxiv": {}}
        self.records: list[dict[str, Any]] = []

    def keep(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False)
        self.guard.assert_absent(line, where=str(self.log))
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        self.records.append(record)

    def request(self, source: str, text: str, page: int) -> httpx.Request:
        if source == "openalex":
            adapter = OpenAlexSource(self.key, base_url=self.bases["openalex"]) if self.bases else OpenAlexSource(self.key)
            return adapter.request(text, limit=page)
        arxiv = ArxivSource(base_url=self.bases["arxiv"] + "/api/query") if self.bases else ArxivSource()
        return arxiv.request(text, limit=page)

    async def send(self, source: str, qid: str, text: str, page: int, pass_no: int | None) -> None:
        record: dict[str, Any] = {"day": self.day, "source": source, "page": page, "query_id": qid, "pass": pass_no}
        try:
            request = self.request(source, text, page)
        except ValueError as exc:
            self.keep({**record, "outcome": "refused", "detail": str(exc)[:120]})
            return
        if self.left[source] <= 0:
            self.keep({**record, "outcome": "budget"})
            return
        events: dict[str, float] = {}

        async def trace(name: str, info: dict[str, Any]) -> None:
            events.setdefault(name, time.perf_counter())

        request.extensions["trace"] = trace
        mark, gap = PACING[source]
        last = self.last[source].get(mark)
        if last is not None:
            while (wait := last + gap * (self.scale if self.bases else 1.0) - time.time()) > 0:
                await asyncio.sleep(wait)
        self.left[source] -= 1
        record["t_start"] = time.time()
        self.last[source]["start"] = record["t_start"]
        began = time.perf_counter()
        response = None
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0), follow_redirects=False,
                                         verify=tls_context()) as client:
                response = await client.send(request)
            outcome = "ok" if response.status_code == 200 else "status"
            if outcome == "ok":
                try:
                    (parse_works(response.json()) if source == "openalex" else parse_feed(response.content))
                except Exception as exc:  # noqa: BLE001 - any parser refusal is "not ok"
                    outcome, record["detail"] = "unparsed", type(exc).__name__
        except httpx.TimeoutException:
            outcome = "timeout"
        except httpx.HTTPError as exc:
            outcome, record["detail"] = "error", type(exc).__name__
        ended = time.perf_counter()
        record["t_end"] = time.time()
        self.last[source]["end"] = record["t_end"]
        headers = events.get("http11.receive_response_headers.complete")
        record.update(
            outcome=outcome, status=response.status_code if response is not None else None,
            bytes=len(response.content) if response is not None else None,
            headers_s=round(headers - began, 4) if headers is not None else None,
            last_byte_s=round(ended - began, 4),
            cost_usd=response.headers.get("x-ratelimit-cost-usd") if response is not None and source == "openalex" else None,
        )
        self.keep(record)


async def run_day(day: Day, queries: list[tuple[str, str]]) -> None:
    start = time.time()

    async def openalex_lane() -> None:
        for k, (pass_no, qid, text, page) in enumerate(openalex_schedule(queries)):
            while (wait := start + k * OPENALEX_EVERY_S * day.scale - time.time()) > 0:
                await asyncio.sleep(wait)
            await day.send("openalex", qid, text, page, pass_no)

    async def arxiv_lane() -> None:
        for k, (qid, text) in enumerate(queries):
            while (wait := start + k * ARXIV_EVERY_S * day.scale - time.time()) > 0:
                await asyncio.sleep(wait)
            await day.send("arxiv", qid, text, 50, None)
            print(f"day {day.day} arxiv {k + 1:2d}/30", flush=True)

    await asyncio.gather(openalex_lane(), arxiv_lane())


def standby_entries(since: float, until: float) -> list[float]:
    return list(_a5().standby_entries(since, until))


def mark_void(records: list[dict[str, Any]]) -> list[float]:
    sent = [r for r in records if "t_start" in r]
    if not sent:
        return []
    entries = standby_entries(min(r["t_start"] for r in sent), max(r["t_end"] for r in sent))
    for r in sent:
        r["void"] = any(r["t_start"] <= t <= r["t_end"] for t in entries)
    return entries


def valid(records: list[dict[str, Any]], day: int, source: str, page: int) -> list[dict[str, Any]]:
    return [r for r in records if r["day"] == day and r["source"] == source and r["page"] == page
            and "t_start" in r and not r.get("void")]


def finished(r: dict[str, Any], deadline: float) -> bool:
    return r["outcome"] == "ok" and r["last_byte_s"] <= deadline


def summarise_day(records: list[dict[str, Any]], day: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source, page in (("openalex", 25), ("openalex", 15), ("arxiv", 50)):
        rows = valid(records, day, source, page)
        ok = [r for r in rows if r["outcome"] == "ok"]
        block: dict[str, Any] = {
            "n": len(rows),
            "within": {str(d): wilson(sum(finished(r, d) for r in rows), len(rows)) for d in REPORT_DEADLINES},
            "headers_s": {"p50": nearest_rank([r["headers_s"] for r in ok if r["headers_s"] is not None], 50),
                          "p95": nearest_rank([r["headers_s"] for r in ok if r["headers_s"] is not None], 95)},
            "last_byte_s": {"p50": nearest_rank([r["last_byte_s"] for r in ok], 50),
                            "p95": nearest_rank([r["last_byte_s"] for r in ok], 95)},
            "outcomes": {o: sum(r["outcome"] == o for r in rows) for o in sorted({r["outcome"] for r in rows})},
        }
        if source == "openalex":
            block["by_pass"] = {str(p): wilson(sum(finished(r, 2.0) for r in rows if r["pass"] == p),
                                               sum(1 for r in rows if r["pass"] == p)) for p in (1, 2)}
        out[f"{source}{page}"] = block
    return out


def verdict(records: list[dict[str, Any]], days: list[int]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    chosen = None
    for name, page, deadline in CONFIGS:
        per_day = {}
        for day in days:
            per_source = {}
            for source, size in (("openalex", page), ("arxiv", 50)):
                rows = valid(records, day, source, size)
                k = sum(finished(r, deadline) for r in rows)
                per_source[source] = {"k": k, "n": len(rows), "floor": math.ceil(0.8 * len(rows)),
                                      "passes": len(rows) > 0 and k >= math.ceil(0.8 * len(rows))}
            per_day[str(day)] = {"sources": per_source, "passes": all(s["passes"] for s in per_source.values())}
        passes = len(days) == 3 and all(d["passes"] for d in per_day.values())
        table[name] = {"openalex_page": page, "deadline_s": deadline, "days": per_day, "passes": passes}
        if passes and chosen is None:
            chosen = name
    return {"configurations": table, "chosen": chosen,
            "meaning": ("none passes: NFR-P1's cold half cannot be met with these sources inside 3.0 s; the owner decides"
                        if chosen is None else f"'{chosen}'; A7 re-judges criterion 1 end to end")}


def day_refusal(done: dict[str, Any], day: int, now: float) -> str | None:
    """Why day N may not run now, or None: A6-multiday's order, calendar-day and 18-hour rules."""
    if str(day) in done:
        return f"day {day} has already run (no repeats)"
    if day != len(done) + 1:
        return f"day {day} is out of order: {len(done)} day(s) have run"
    if done:
        if datetime.fromtimestamp(now).date().isoformat() in {d["date"] for d in done.values()}:
            return "A6-multiday needs a different calendar day for each run"
        if now - done[str(day - 1)]["t_start"] < MIN_HOURS_BETWEEN_DAYS * 3600:
            return f"day {day} is too soon: at least {MIN_HOURS_BETWEEN_DAYS:.0f} h after day {day - 1}"
    return None


def write(path: Path, data: dict[str, Any], guard: Secrets) -> None:
    text = json.dumps(data, indent=1, ensure_ascii=False, allow_nan=False)
    guard.assert_absent(text, where=str(path))
    path.write_text(text + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "addendum-a6"):
        p = sub.add_parser(name)
        p.add_argument("--day", type=int, choices=(1, 2, 3), required=True)
    dry = sub.add_parser("dry-run")
    dry.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())

    if args.command == "plan":
        protocol, _ = load_protocol()
        for pass_no, qid, _text, page in openalex_schedule(day_order(protocol, args.day)):
            print(f"openalex pass {pass_no} {qid:28s} page {page}")
        return 0

    if args.command == "dry-run":
        protocol = json.loads((ROOT / "spike" / "m4-protocol.json").read_text(encoding="utf-8"))
        guard = Secrets([load_env().get("OPENALEX_API_KEY", "")])
        stub = _stub_class()()
        budget = dict(protocol["addenda"]["A6-multiday"]["budget"]["max"])
        left = {"openalex": int(budget["openalex_searches"]), "arxiv": int(budget["arxiv_queries"])}
        records: list[dict[str, Any]] = []
        try:
            for n in (1, 2, 3):
                day = Day(n, "placeholder-not-a-key", {"openalex": stub.base, "arxiv": stub.base}, args.scale,
                          RUN_DIR / f"a6-multiday-dry-day{n}-{stamp}.jsonl", left, guard)
                asyncio.run(run_day(day, day_order(protocol, n)))
                records += day.records
        finally:
            stub.close()
        mark_void(records)
        out = {"days": {str(n): summarise_day(records, n) for n in (1, 2, 3)},
               "verdict": verdict(records, [1, 2, 3]), "records": records}
        write(DRY_PATH, out, guard)
        print(json.dumps(out["verdict"]["configurations"]["as shipped"]["days"], indent=1)[:800])
        print("chosen:", out["verdict"]["chosen"], "| budget left:", left)
        return 0

    protocol, digest = load_protocol()
    if "A6-multiday" not in protocol["addenda"]:
        raise SystemExit("the approved protocol has no addendum A6-multiday")
    key = load_env().get("OPENALEX_API_KEY")
    if not key:
        raise SystemExit("A6-multiday runs OpenAlex keyed, as shipped; no key in spike/.env")
    guard = Secrets([key])
    state: dict[str, Any] = json.loads(RESULTS_PATH.read_text(encoding="utf-8")) if RESULTS_PATH.exists() else \
        {"addendum": "A6-multiday", "days_run": {}, "records": []}
    done = state["days_run"]
    now = time.time()
    refusal = day_refusal(done, args.day, now)
    if refusal:
        raise SystemExit(refusal)
    budget = protocol["addenda"]["A6-multiday"]["budget"]["max"]
    sent = {s: sum(1 for r in state["records"] if r["source"] == s and "t_start" in r) for s in ("openalex", "arxiv")}
    left = {"openalex": int(budget["openalex_searches"]) - sent["openalex"],
            "arxiv": int(budget["arxiv_queries"]) - sent["arxiv"]}
    day = Day(args.day, key, None, 1.0, RUN_DIR / f"a6-multiday-day{args.day}-{stamp}.jsonl", left, guard)
    asyncio.run(run_day(day, day_order(protocol, args.day)))
    entries = mark_void(day.records)
    done[str(args.day)] = {"date": datetime.fromtimestamp(now).date().isoformat(), "t_start": now,
                           "local_start": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
                           "protocol_sha256": digest, "standby_entries": entries,
                           "cost_usd": round(sum(float(r["cost_usd"]) for r in day.records if r.get("cost_usd")), 4)}
    state["records"] += day.records
    days = sorted(int(d) for d in done)
    state["days"] = {str(n): summarise_day(state["records"], n) for n in days}
    if len(days) == 3:
        state["verdict"] = verdict(state["records"], days)
    write(RESULTS_PATH, state, guard)
    summary = state["days"][str(args.day)]
    for k, v in summary.items():
        print(k, {d: (w["k"], w["n"]) for d, w in v["within"].items()})
    if "verdict" in state:
        print("chosen:", state["verdict"]["chosen"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
