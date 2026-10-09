"""M4 phase 10b's live run, addendum A4 (D158, D159): the shipped app, end to end, live sources.

    .venv\\Scripts\\python.exe spike\\scripts\\20_discovery_live.py plan
    .venv\\Scripts\\python.exe spike\\scripts\\20_discovery_live.py addendum-a4
    .venv\\Scripts\\python.exe spike\\scripts\\20_discovery_live.py ui-hold
    .venv\\Scripts\\python.exe spike\\scripts\\20_discovery_live.py dry-run

`plan` prints the schedule and sends nothing. `addendum-a4` runs A4: 30 queries over 5
fresh starts of the real app, each searched cold and then repeated, then the 4 kill
searches, and writes spike/results/20_discovery_live.json with D158's verdict.
`ui-hold` starts the app with OpenAlex refused and serves on 127.0.0.1:8770 until a
STOP file appears next to this script, for A4's UI check by hand. `dry-run` runs the
whole schedule against a local stub serving A3's recorded t03 responses, with a
placeholder key, so the harness is checked end to end before any live call; it needs
no approval because it asks no third party, and writes only under var/spike/run/.

The app runs in a child process of this script (`serve`, internal), so each start is
a fresh process and the client's work is not the app's. In the child, httpx is
wrapped to record every outbound request (host, path, status, times to headers and
to last byte; never the query string, which carries OpenAlex's key) and to refuse
any request beyond A4's budget; the use case and the dense scorer are wrapped to
record when the fan-out ended and how long ranking took. Nothing about what the app
sends or how it ranks is changed.
"""

from __future__ import annotations

import argparse
import functools
import gzip
import json
import math
import random
import secrets
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from m4_common import (
    RESULTS_DIR,
    ROOT,
    RUN_DIR,
    Secrets,
    load_env,
    load_protocol,
    nearest_rank,
    norm_arxiv,
    norm_title,
    wilson,
)

HERE = Path(__file__).resolve()
PY = sys.executable
APP_PORT = 8770
STOP_FILE = HERE.parent / "STOP"
RESULTS_PATH = RESULTS_DIR / "20_discovery_live.json"
A3_POOLS = ROOT / "data" / "eval" / "discovery-pools-a3.json.gz"
BROKER, CACHE = "redis://localhost:6379/11", "redis://localhost:6380/11"
HOSTS = {"openalex": "api.openalex.org", "arxiv": "export.arxiv.org"}
BUDGET_KEY = {"openalex": "openalex_searches", "arxiv": "arxiv_queries"}
#: The production adapters' paths, so a dry run's stub host is classified like the real one.
PATHS = {"openalex": "/works", "arxiv": "/api/query"}
PHRASINGS = ("keyword", "natural_language", "reversed_keyword")
SPACING_S = 6.5
STARTS = 5
#: A4's kill searches, in order: (source, how it is killed, topic whose keyword query is searched).
#: "refused" takes Windows 2.04 s to report (it retries a closed port), so it reads as a
#: timeout at the 2.0 s deadline; "error" answers 500 at once. Not 503: the arXiv adapter
#: reads 503 as throttling (arXiv throttles with it), so it would report rate_limited.
KILLS = (("openalex", "refused", "t01"), ("openalex", "hang", "t02"), ("openalex", "error", "t03"),
         ("arxiv", "refused", "t04"), ("arxiv", "hang", "t05"), ("arxiv", "error", "t06"))


@dataclass(frozen=True)
class Step:
    query_id: str
    topic: str
    phrasing: str
    text: str


def a4_plan(protocol: dict[str, Any]) -> list[list[Step]]:
    """A4's order: the 30 queries shuffled with stream '<seed>/A4', dealt into 5 blocks of 6."""
    rng = random.Random(f"{protocol['seed']}/A4")
    steps = [Step(f"{t['id']}-{ph}", t["id"], ph, t[ph]) for t in protocol["topics"] for ph in PHRASINGS]
    rng.shuffle(steps)
    size = len(steps) // STARTS
    return [steps[i * size:(i + 1) * size] for i in range(STARTS)]


# -- the app process ------------------------------------------------------------


def source_of(host: str, path: str) -> str | None:
    for name in HOSTS:
        if host == HOSTS[name] or path.startswith(PATHS[name]):
            return name
    return None


def serve(args: argparse.Namespace) -> None:
    """One app process: the real app, its outbound requests recorded and budgeted."""
    import httpx
    import uvicorn

    log = open(args.log, "a", encoding="utf-8", buffering=1)  # noqa: SIM115
    lock = threading.Lock()
    remaining = {"openalex": args.openalex_left, "arxiv": args.arxiv_left}
    key = "placeholder-not-a-key" if args.dry_run else load_env().get("OPENALEX_API_KEY") or ""
    guard = Secrets([key])

    def write(record: dict[str, Any]) -> None:
        line = json.dumps(record)
        guard.assert_absent(line, where=args.log)
        with lock:
            log.write(line + "\n")

    original_async = httpx.AsyncClient.send

    async def send(self: httpx.AsyncClient, request: httpx.Request, *a: Any, **k: Any) -> httpx.Response:
        host, path = request.url.host, request.url.path
        source = source_of(host, path)
        record: dict[str, Any] = {"kind": "outbound", "host": host, "path": path, "source": source,
                                  "start": args.start, "t_start": time.time()}
        if source is not None:
            if remaining[source] <= 0:
                write({**record, "outcome": "refused_by_budget", "t_end": time.time()})
                raise httpx.ConnectError("A4 budget exhausted for this source", request=request)
            remaining[source] -= 1
        stream = k.pop("stream", False)
        try:
            response = await original_async(self, request, *a, stream=True, **k)
        except BaseException as exc:
            write({**record, "outcome": type(exc).__name__, "t_end": time.time()})
            raise
        record.update(t_headers=time.time(), status=response.status_code)
        if not stream:
            try:
                await response.aread()
            except BaseException as exc:
                write({**record, "outcome": type(exc).__name__, "t_end": time.time()})
                await response.aclose()
                raise
            record.update(bytes=len(response.content))
        write({**record, "outcome": "response", "t_end": time.time()})
        return response

    httpx.AsyncClient.send = send  # type: ignore[method-assign]

    original_sync = httpx.Client.send

    def send_sync(self: httpx.Client, request: httpx.Request, *a: Any, **k: Any) -> httpx.Response:
        # The model hub's client is synchronous; D157 expects it never to be asked.
        write({"kind": "outbound", "host": request.url.host, "path": request.url.path, "source": None,
               "start": args.start, "t_start": time.time(), "outcome": "sync"})
        return original_sync(self, request, *a, **k)

    httpx.Client.send = send_sync  # type: ignore[method-assign]

    from scholiast.api.app import create_app
    from scholiast.application.use_cases import discover_papers, rank_works
    from scholiast.config.settings import Settings
    from scholiast.infrastructure.discovery import sources

    def timed(cls: Any, name: str, kind: str) -> None:
        original = getattr(cls, name)

        async def wrapper(self: Any, *a: Any, **k: Any) -> Any:
            started = time.time()
            try:
                return await original(self, *a, **k)
            finally:
                write({"kind": kind, "start": args.start, "t_start": started, "t_end": time.time()})

        setattr(cls, name, wrapper)

    timed(discover_papers.DiscoverPapersUseCase, "execute", "execute")
    timed(rank_works.DenseRelevance, "score", "rank")
    timed(rank_works.LexicalRelevance, "score", "rank")

    # Only for a killed source or a dry run; otherwise the adapters' own production URLs.
    if args.openalex_base:
        sources.OpenAlexSource = functools.partial(  # type: ignore[misc]
            sources.OpenAlexSource, base_url=args.openalex_base)
    if args.arxiv_base:
        sources.ArxivSource = functools.partial(  # type: ignore[misc]
            sources.ArxivSource, base_url=args.arxiv_base + PATHS["arxiv"])

    settings = Settings(
        database_url=args.db,  # type: ignore[arg-type]
        object_store_path=Path(args.objects),
        qdrant_collection=args.collection,
        redis_url=BROKER,
        discovery_cache_url=CACHE,
        openalex_api_key=key or None,
        jwt_secret=secrets.token_urlsafe(48),
        worker_in_process=False,
        log_level="WARNING",
    )
    write({"kind": "boot", "start": args.start, "t_start": time.time()})
    uvicorn.run(create_app(settings), host="127.0.0.1", port=APP_PORT, log_level="warning")


# -- the client -------------------------------------------------------------------


class Hang:
    """A local socket that accepts connections and never answers: a source that hangs."""

    def __init__(self) -> None:
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]
        self.held: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            self.held.append(conn)

    def close(self) -> None:
        self.sock.close()
        for conn in self.held:
            conn.close()


class Unavailable:
    """A local server that answers every request with 500 at once: a source that is broken."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                self.send_response(500)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def log_message(self, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = int(self.server.server_address[1])
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


class Stub:
    """The dry run's sources: A3's recorded t03 responses, 300 ms after each request, any query."""

    def __init__(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        entries = json.loads(gzip.decompress(A3_POOLS.read_bytes()))["entries"]
        bodies = {e["source"]: e["body"].encode("utf-8") for e in entries if e["query_id"] == "t03-keyword"}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                time.sleep(0.3)
                name = source_of("", self.path.split("?")[0])
                if name is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                kind = "application/json" if name == "openalex" else "application/atom+xml"
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(bodies[name])))
                self.end_headers()
                self.wfile.write(bodies[name])

            def log_message(self, *args: object) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()


def unused_port() -> int:
    """A local port nobody listens on: bound once to learn a free number, then released."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class App:
    """One start: a fresh app process, up until the block ends."""

    def __init__(self, run: Run, start: str, kill: tuple[str, int] | None = None) -> None:
        self.run, self.start, self.kill = run, start, kill
        self.collection = f"a4_{uuid.uuid4().hex[:8]}"

    def __enter__(self) -> App:
        import httpx

        left = self.run.budget_left()
        cmd = [PY, str(HERE), "serve", "--db", self.run.db, "--collection", self.collection,
               "--objects", str(self.run.objects), "--log", str(self.run.log_path), "--start", self.start,
               "--openalex-left", str(left["openalex"]), "--arxiv-left", str(left["arxiv"])]
        bases = {name: self.run.stub_base for name in HOSTS} if self.run.stub_base else {}
        if self.kill:
            bases[self.kill[0]] = f"http://127.0.0.1:{self.kill[1]}"
        for name, base in bases.items():
            cmd += [f"--{name}-base", base]
        if self.run.stub_base:
            cmd += ["--dry-run"]
        began = time.time()
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        self.client = httpx.Client(base_url=f"http://127.0.0.1:{APP_PORT}", timeout=120)
        while True:
            if self.proc.poll() is not None:
                raise SystemExit(f"start {self.start}: the app exited: {(self.proc.stderr.read() or '')[-3000:]}")
            try:
                if self.client.get("/health/live").status_code == 200:
                    break
            except httpx.TransportError:
                pass
            time.sleep(0.05)
        self.run.starts.append({"start": self.start, "kill": self.kill[0] if self.kill else None,
                                "t_start": began, "t_live": time.time(),
                                "startup_s": round(time.time() - began, 3)})
        self.run.login(self.client)
        return self

    def __exit__(self, *exc: object) -> None:
        self.client.close()
        self.proc.terminate()
        self.proc.wait(timeout=60)
        try:
            from qdrant_client import QdrantClient

            QdrantClient(url="http://localhost:6333").delete_collection(self.collection)
        except Exception as error:  # noqa: BLE001 - cleanup is reported, never fatal
            print(f"qdrant cleanup: {type(error).__name__}", flush=True)


class Run:
    def __init__(self, protocol: dict[str, Any], digest: str, db: str, objects: Path,
                 stub_base: str | None = None) -> None:
        self.protocol, self.digest, self.db, self.objects = protocol, digest, db, objects
        self.stub_base = stub_base
        self.budget = protocol["addenda"]["A4"]["budget"]["max"]
        stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        self.log_path = RUN_DIR / f"a4-{'dry-' if stub_base else ''}{stamp}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.creds = {"email": f"a4-{uuid.uuid4().hex[:10]}@example.com", "password": secrets.token_urlsafe(18)}
        self.registered = False
        self.token = ""
        self.last_search = 0.0
        self.searches: list[dict[str, Any]] = []
        self.starts: list[dict[str, Any]] = []

    def server_records(self) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines() if line]

    def budget_left(self) -> dict[str, int]:
        sent = {name: 0 for name in HOSTS}
        for r in self.server_records():
            if r.get("kind") == "outbound" and r.get("source") and r.get("outcome") != "refused_by_budget":
                sent[r["source"]] += 1
        return {name: int(self.budget[BUDGET_KEY[name]]) - sent[name] for name in HOSTS}

    def login(self, client: Any) -> None:
        if not self.registered:
            client.post("/v1/auth/register", json=self.creds).raise_for_status()
            self.registered = True
        response = client.post("/v1/auth/login", json=self.creds)
        response.raise_for_status()
        self.token = response.json()["access_token"]

    def search(self, app: App, step: Step, kind: str, **extra: Any) -> dict[str, Any]:
        while (wait := self.last_search + SPACING_S - time.time()) > 0:
            time.sleep(wait)
        if STOP_FILE.exists():
            raise SystemExit(f"{STOP_FILE} exists: stopped before {kind} {step.query_id}")
        self.last_search = time.time()
        sent_wall, sent = time.time(), time.perf_counter()
        response = app.client.post("/v1/discovery/search", json={"query": step.text},
                                   headers={"Authorization": f"Bearer {self.token}"})
        elapsed = time.perf_counter() - sent
        body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        record = {
            "kind": kind, "query_id": step.query_id, "topic": step.topic, "phrasing": step.phrasing,
            "start": app.start, "t_sent": sent_wall, "t_done": time.time(), "latency_s": round(elapsed, 4),
            "status": response.status_code, "partial": body.get("partial"),
            "sources": {s["source"]: s["status"] for s in body.get("source_status", [])},
            "cached": {s["source"]: s.get("cached", False) for s in body.get("source_status", [])},
            "relevance": (body.get("ranking") or {}).get("relevance"),
            "results": [{"title": w["title"], "year": w.get("year"), "arxiv_id": w.get("arxiv_id"),
                         "members": [[m["source"], m["source_rank"]] for m in w["members"]]}
                        for w in body.get("results", [])],
            **extra,
        }
        self.searches.append(record)
        print(f"{kind:7s} {step.query_id:28s} {record['status']} {elapsed:6.3f} s {record['sources']}", flush=True)
        return record


def run_a4(dry_run: bool = False) -> int:
    import redis
    from testcontainers.community.postgres import PostgresContainer

    if dry_run:
        # Nothing reaches a third party, so the approval the hash stands for is not needed.
        protocol = json.loads((ROOT / "spike" / "m4-protocol.json").read_text(encoding="utf-8"))
        digest = "dry-run"
    else:
        protocol, digest = load_protocol()
        if not load_env().get("OPENALEX_API_KEY"):
            raise SystemExit("A4 runs OpenAlex keyed, as shipped; no key in spike/.env")
    if "A4" not in protocol["addenda"]:
        raise SystemExit("the protocol has no addendum A4")
    blocks = a4_plan(protocol)
    topics = {t["id"]: t for t in protocol["topics"]}
    for url in (BROKER, CACHE):
        client = redis.Redis.from_url(url)
        if client.dbsize():
            raise SystemExit(f"{url} is not empty; A4 uses database 11 as throwaway state, and will not empty it")
    STOP_FILE.unlink(missing_ok=True)
    objects = RUN_DIR / f"a4-objects-{uuid.uuid4().hex[:8]}"
    stub = Stub() if dry_run else None
    try:
        with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
            db = container.get_connection_url()
            subprocess.run([PY, "-m", "alembic", "-x", f"db_url={db}", "upgrade", "head"],
                           cwd=ROOT, check=True, capture_output=True)
            run = Run(protocol, digest, db, objects, stub.base if stub else None)
            for number, block in enumerate(blocks, 1):
                with App(run, f"s{number}") as app:
                    for position, step in enumerate(block):
                        run.search(app, step, "cold", first_after_start=position == 0)
                        run.search(app, step, "repeat")
            hang, unavailable = Hang(), Unavailable()
            try:
                for source, mode, topic in KILLS:
                    redis.Redis.from_url(CACHE).flushdb()
                    port = {"refused": unused_port(), "hang": hang.port, "error": unavailable.port}[mode]
                    step = Step(f"{topic}-keyword", topic, "keyword", topics[topic]["keyword"])
                    with App(run, f"kill-{source}-{mode}", kill=(source, port)) as app:
                        run.search(app, step, "kill", killed=source, how=mode)
            finally:
                hang.close()
                unavailable.close()
            results = analyse(run)
    finally:
        if stub:
            stub.close()
        for url in (BROKER, CACHE):
            redis.Redis.from_url(url).flushdb()
        import shutil

        shutil.rmtree(objects, ignore_errors=True)
    out = RUN_DIR / "a4-dry-run.json" if dry_run else RESULTS_PATH
    text = json.dumps(results, indent=1, ensure_ascii=False)
    Secrets([load_env().get("OPENALEX_API_KEY", "")]).assert_absent(text, where=str(out))
    out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(results["verdict"], indent=1))
    return 0


# -- analysis ---------------------------------------------------------------------


def within(records: list[dict[str, Any]], search: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [r for r in records if r.get("kind") == kind and search["t_sent"] <= r["t_start"] <= search["t_done"]]


def pct(values: list[float]) -> dict[str, Any]:
    finite = [v for v in values if math.isfinite(v)]
    return {"n": len(values), "p50": nearest_rank(values, 50), "p95": nearest_rank(values, 95),
            "max": max(values) if values else None, "failed": len(values) - len(finite)}


def stage(records: list[dict[str, Any]], search: dict[str, Any]) -> dict[str, Any]:
    execute = within(records, search, "execute")
    rank = within(records, search, "rank")
    out: dict[str, Any] = {}
    if execute:
        e = execute[0]
        out["use_case_s"] = round(e["t_end"] - e["t_start"], 4)
        if rank:
            out["fan_out_s"] = round(rank[0]["t_start"] - e["t_start"], 4)
            out["ranking_s"] = round(rank[0]["t_end"] - rank[0]["t_start"], 4)
        out["outside_use_case_s"] = round(search["latency_s"] - out["use_case_s"], 4)
    return out


def source_timings(records: list[dict[str, Any]], search: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for name in HOSTS:
        calls = [r for r in within(records, search, "outbound") if r.get("source") == name]
        if calls:
            c = calls[0]
            out[name] = {"outcome": c.get("outcome"), "status": c.get("status"),
                         "headers_s": round(c["t_headers"] - c["t_start"], 4) if "t_headers" in c else None,
                         "last_byte_s": round(c["t_end"] - c["t_start"], 4) if c.get("outcome") == "response" else None}
    return out


def anchor_hits(results: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any] | None:
    want_title, want_id = norm_title(anchor["title"]), norm_arxiv(anchor.get("arxiv_id"))
    for rank, work in enumerate(results, 1):
        by_id = want_id is not None and norm_arxiv(work.get("arxiv_id")) == want_id
        if by_id or norm_title(work["title"]) == want_title:
            return {"rank": rank, "rule": "arxiv_id" if by_id else "title",
                    "sources": sorted({m[0] for m in work["members"]})}
    return None


def a3_anchor_pages(protocol: dict[str, Any]) -> dict[str, dict[str, bool]]:
    """Whether each source's recorded A3 keyword page contained each anchor, by the same rule."""
    from scholiast.infrastructure.discovery.arxiv import parse_feed
    from scholiast.infrastructure.discovery.openalex import parse_works

    entries = json.loads(gzip.decompress(A3_POOLS.read_bytes()))["entries"]
    pages: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for e in entries:
        if not e["query_id"].endswith("-keyword"):
            continue
        page = parse_works(json.loads(e["body"])) if e["source"] == "openalex" else parse_feed(e["body"])
        pages[(e["query_id"].split("-")[0], e["source"])] = [
            {"title": c.title, "arxiv_id": c.arxiv_id, "members": [[c.source, c.rank]]} for c in page.candidates]
    out: dict[str, dict[str, bool]] = {}
    for topic in protocol["topics"]:
        for anchor in topic["anchors"]:
            out[anchor["title"]] = {source: anchor_hits(pages.get((topic["id"], source), []), anchor) is not None
                                    for source in HOSTS}
    return out


def analyse(run: Run) -> dict[str, Any]:
    records = run.server_records()
    cold = [s for s in run.searches if s["kind"] == "cold"]
    repeats = [s for s in run.searches if s["kind"] == "repeat"]
    by_query = {s["query_id"]: s for s in cold}

    def latency(s: dict[str, Any]) -> float:
        return s["latency_s"] if s["status"] == 200 else math.inf

    complete = {q for q, s in by_query.items() if s["status"] == 200 and all(v == "ok" for v in s["sources"].values())
                and set(s["sources"]) == set(HOSTS)}
    qualifying = [r for r in repeats if r["query_id"] in complete]
    outbound_on_repeat = {r["query_id"]: len(within(records, r, "outbound")) for r in repeats}
    completion = {name: wilson(sum(1 for s in cold if s["sources"].get(name) == "ok"), len(cold)) for name in HOSTS}
    cold_stats = pct([latency(s) for s in cold])
    cached_stats = pct([latency(s) for s in qualifying])

    floor = math.ceil(0.8 * len(cold))
    conditions = {
        "cold_p95_le_3s": cold_stats["p95"] is not None and cold_stats["p95"] <= 3.0,
        "cached_p95_le_0_4s": cached_stats["p95"] is not None and cached_stats["p95"] <= 0.4,
        "each_source_ok_on_at_least_24_of_30": all(c["k"] >= floor for c in completion.values()) and len(cold) == 30,
        "zero_outbound_on_every_counted_repeat": all(outbound_on_repeat[r["query_id"]] == 0 for r in qualifying),
    }
    first = [s for s in cold if s.get("first_after_start")]
    rest = [s for s in cold if not s.get("first_after_start")]
    keyword_cold = {s["topic"]: s for s in cold if s["phrasing"] == "keyword"}
    a3_pages = a3_anchor_pages(run.protocol)
    anchors = []
    for topic in run.protocol["topics"]:
        for anchor in topic["anchors"]:
            today = keyword_cold.get(topic["id"])
            anchors.append({"topic": topic["id"], "title": anchor["title"],
                            "today": anchor_hits(today["results"], anchor) if today else None,
                            "a3_pages": a3_pages[anchor["title"]]})
    boots = []
    for start in run.starts:
        during = [r for r in records if r.get("kind") == "outbound" and r.get("start") == start["start"]
                  and r["t_start"] <= start["t_live"]]
        boots.append({**start, "outbound_while_booting": sorted({r["host"] for r in during}),
                      "outbound_count": len(during)})
    kills = [{"killed": s["killed"], "how": s["how"], "status": s["status"], "partial": s["partial"],
              "sources": s["sources"], "latency_s": s["latency_s"], "results": len(s["results"]),
              "as_expected": s["status"] == 200 and s["partial"] is True
              and s["sources"].get(s["killed"]) != "ok"
              and all(v == "ok" for k, v in s["sources"].items() if k != s["killed"])}
             for s in run.searches if s["kind"] == "kill"]
    return {
        "addendum": "A4", "protocol_sha256": run.digest,
        "ran": {"from": min(s["t_sent"] for s in run.searches), "to": max(s["t_done"] for s in run.searches)},
        "verdict": {
            "criterion_1": "met" if all(conditions.values()) else "not met as written",
            "conditions": conditions,
            "cold": cold_stats, "cached": cached_stats,
            "completion": completion,
            "repeats_counted": len(qualifying), "repeats_of_partial_searches": len(repeats) - len(qualifying),
        },
        "first_after_start": {"first": pct([latency(s) for s in first]), "rest": pct([latency(s) for s in rest]),
                              "first_sources": [s["sources"] for s in first]},
        "repeats_of_partial_searches": [{"query_id": r["query_id"], "latency_s": r["latency_s"],
                                         "sources": r["sources"], "outbound": outbound_on_repeat[r["query_id"]]}
                                        for r in repeats if r["query_id"] not in complete],
        "starts": boots,
        "kills": kills,
        "anchors": anchors,
        "searches": [{**{k: v for k, v in s.items() if k != "results"}, "results": len(s["results"]),
                      "stages": stage(records, s), "source_timings": source_timings(records, s)}
                     for s in run.searches],
        "outbound": [r for r in records if r.get("kind") == "outbound"],
        "budget_left": run.budget_left(),
    }


# -- the UI check -----------------------------------------------------------------


def ui_hold() -> int:
    """One start with OpenAlex refused, held open for A4's UI check by hand."""
    import redis
    from testcontainers.community.postgres import PostgresContainer

    protocol, digest = load_protocol()
    STOP_FILE.unlink(missing_ok=True)
    objects = RUN_DIR / f"a4-ui-objects-{uuid.uuid4().hex[:8]}"
    try:
        with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
            db = container.get_connection_url()
            subprocess.run([PY, "-m", "alembic", "-x", f"db_url={db}", "upgrade", "head"],
                           cwd=ROOT, check=True, capture_output=True)
            run = Run(protocol, digest, db, objects)
            # The UI check's 3 searches are A4's budget too: it starts from what the run left.
            previous = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["budget_left"]
            run.budget = {BUDGET_KEY[name]: left for name, left in previous.items()}
            # At most 3 searches by hand; OpenAlex is killed by its dead URL, as A4 says,
            # and its local attempts count against what is left of its budget.
            run.budget["arxiv_queries"] = min(run.budget["arxiv_queries"], 3)
            with App(run, "ui-openalex-refused", kill=("openalex", unused_port())):
                print(f"serving on http://127.0.0.1:{APP_PORT}, OpenAlex refused; touch {STOP_FILE} to stop",
                      flush=True)
                while not STOP_FILE.exists():
                    time.sleep(1)
            print(json.dumps([r for r in run.server_records() if r.get("kind") == "outbound"], indent=1))
    finally:
        for url in (BROKER, CACHE):
            redis.Redis.from_url(url).flushdb()
        import shutil

        shutil.rmtree(objects, ignore_errors=True)
        STOP_FILE.unlink(missing_ok=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    sub.add_parser("addendum-a4")
    sub.add_parser("ui-hold")
    sub.add_parser("dry-run")
    s = sub.add_parser("serve")
    for name in ("--db", "--collection", "--objects", "--log", "--start"):
        s.add_argument(name, required=True)
    s.add_argument("--openalex-left", type=int, required=True)
    s.add_argument("--arxiv-left", type=int, required=True)
    s.add_argument("--openalex-base")
    s.add_argument("--arxiv-base")
    s.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "serve":
        serve(args)
        return 0
    if args.command == "plan":
        protocol, _ = load_protocol()
        for number, block in enumerate(a4_plan(protocol), 1):
            print(f"start s{number}: " + ", ".join(step.query_id for step in block))
        print("kills: " + ", ".join(f"{s} {m} ({t}-keyword)" for s, m, t in KILLS))
        return 0
    if args.command == "ui-hold":
        return ui_hold()
    return run_a4(dry_run=args.command == "dry-run")


if __name__ == "__main__":
    sys.exit(main())
