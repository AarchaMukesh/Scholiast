"""M4 discovery spike, passes P3 and P4 (D116, spike/m4-protocol.json).

    .venv\\Scripts\\python.exe spike\\scripts\\18_oa_probe.py works      # no network
    .venv\\Scripts\\python.exe spike\\scripts\\18_oa_probe.py probe
    .venv\\Scripts\\python.exe spike\\scripts\\18_oa_probe.py download

`works` builds the P3 work list from session 1's keyword results and prints it, with
each work's candidate URLs in the protocol's preference order; it sends nothing.
`probe` reads the first 2 KiB of candidate URLs, stopping at the first that starts
with %PDF-. `download` fetches up to 30 of those in full and runs the production
`validate_upload` and `preflight` over them.

Every request goes through `m4_http.fetch_chain`: https only, public addresses
only, redirects re-checked hop by hop, byte caps, per-host pacing, and the budget.
Results land in var/spike/run/ (local); `19_discovery_analysis.py finalize` writes
the tracked, S2-free versions.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
from typing import Any

import httpx
from m4_common import (
    PDF_DIR,
    RUN_DIR,
    RunLog,
    Secrets,
    check_url,
    download_request_keys,
    iso,
    load_env,
    load_protocol,
    probe_budget_keys,
    read_jsonl,
    wall,
)
from m4_http import Chain, build_ctx, fetch_chain
from m4_works import Work, build_works, ordered, p1_keyword_records

from scholiast.application.ports.parsing import DocumentUnparseable
from scholiast.domain.upload import UploadRejected, validate_upload
from scholiast.infrastructure.parsing.preflight import preflight

WORK_CONCURRENCY = 6
PROBE_WORKS = RUN_DIR / "probe_works.jsonl"
DOWNLOADS = RUN_DIR / "downloads.jsonl"


def chain_summary(chain: Chain) -> dict[str, Any]:
    return {
        "outcome": chain.outcome,
        "hosts": chain.hosts,
        "final_host": chain.final_host,
        "upgraded": chain.upgraded,
        "hops": chain.hops,
    }


def make_client(protocol: dict[str, Any], mode: str) -> httpx.AsyncClient:
    t = protocol["timeouts_s"]
    total = float(t["probe_total" if mode == "probe" else "download_total"])
    connect = float(t["probe_connect" if mode == "probe" else "download_connect"])
    return httpx.AsyncClient(
        timeout=httpx.Timeout(total, connect=connect),
        follow_redirects=False,
        headers={"User-Agent": protocol["http_safety"]["headers"]["User-Agent"]},
        limits=httpx.Limits(max_connections=16),
    )


def _logged_outcome(records: list[dict[str, Any]], prefix: str) -> str:
    """What an interrupted earlier attempt ended as, read back from the log."""
    hops = [r for r in records if r.get("rid", "").startswith(prefix + "/")
            and r.get("event") in ("outcome", "not_sent")]
    if not hops:
        return "interrupted"
    if any(r.get("outcome") == "ok" and r.get("is_pdf") for r in hops):
        return "fetched"
    last = max(hops, key=lambda r: r["rid"])
    return "interrupted" if last.get("outcome") == "redirect" else str(last.get("outcome"))


async def probe(protocol: dict[str, Any], secrets: Secrets) -> int:
    works = build_works(p1_keyword_records())
    if not works:
        print("no session-1 keyword results to probe", file=sys.stderr)
        return 2
    log = RunLog("pdf", secrets)
    results = RunLog("probe_works", secrets)
    finished = {r["work"] for r in results.records()}
    ctx = build_ctx(protocol, log)
    earlier = log.records()
    log.append({"event": "run", "command": "probe", "at": iso(wall()), "works": len(works)})
    client = make_client(protocol, "probe")
    semaphore = asyncio.Semaphore(WORK_CONCURRENCY)
    done_count = 0

    async def one(work: Work) -> None:
        nonlocal done_count
        if work.key in finished:
            return
        async with semaphore:
            attempts = []
            result = "no_oa_link" if not work.candidates else "failed"
            for index, cand in enumerate(work.candidates):
                prefix = f"P3/{work.key}/u{index}"
                if f"{prefix}/h0" in ctx.done:
                    outcome = _logged_outcome(earlier, prefix)
                    attempts.append({"url_index": index, "outcome": outcome, "resumed": True})
                else:
                    chain = await fetch_chain(
                        ctx, client, rid_prefix=prefix, url=cand.url, mode="probe",
                        budget_keys_for=probe_budget_keys,
                        meta={"pass": "P3", "work": work.key, "url_index": index,
                              "url_kind": cand.kind},
                    )
                    outcome = chain.outcome
                    attempts.append({"url_index": index, **chain_summary(chain)})
                if outcome == "fetched":
                    result = "fetched"
                    break
            if result != "fetched" and work.candidates and all(
                a.get("outcome") == "not_sent_budget" for a in attempts
            ):
                result = "unprobed_budget"
            elif result != "fetched" and any(
                a.get("outcome") == "not_sent_budget" for a in attempts
            ):
                result = "unresolved_budget"
            results.append(
                {
                    "work": work.key,
                    "topics": sorted(work.topics),
                    "sources": sorted(work.sources),
                    "candidates": [
                        {"url": c.url, "kind": c.kind, "supplied_by": sorted(c.supplied_by),
                         "licences": c.licences, "url_check": check_url(c.url).reason or "ok"}
                        for c in work.candidates
                    ],
                    "attempts": attempts,
                    "result": result,
                    # A failure here may be the host-stop rule, not the host: reported apart.
                    "host_stopped": any(a.get("outcome") == "not_sent_host_stopped" for a in attempts),
                    "at": iso(wall()),
                }
            )
            done_count += 1
            if done_count % 10 == 0:
                print(f"P3 {done_count}/{len(works)} works", flush=True)

    try:
        await asyncio.gather(*(one(w) for w in ordered(protocol, works, "P3")))
    finally:
        await client.aclose()
    log.append({"event": "run_end", "command": "probe", "at": iso(wall()),
                "stopped_hosts": ctx.health.stopped,
                "budget_used": {"/".join(k): v for k, v in ctx.budget.used.items()}})
    for event in ctx.events:
        print("event:", event)
    print(f"P3 finished: {done_count} works probed this run", flush=True)
    return 0


async def download(protocol: dict[str, Any], secrets: Secrets) -> int:
    fetched = []
    for work in read_jsonl(PROBE_WORKS):
        if work["result"] != "fetched":
            continue
        attempt = next(a for a in work["attempts"] if a.get("outcome") == "fetched")
        if attempt.get("resumed"):
            continue  # no chain detail survived to re-download from; counted by omission
        fetched.append((work, attempt))
    arxiv = sorted(((w, a) for w, a in fetched if a["final_host"] == "arxiv.org"),
                   key=lambda p: p[0]["work"])
    other = sorted(((w, a) for w, a in fetched if a["final_host"] != "arxiv.org"),
                   key=lambda p: p[0]["work"])
    rng = random.Random(f"{protocol['seed']}/P4")
    cap_arxiv = protocol["budget"]["arxiv_org_pdfs"]["max"]["downloads"]
    cap_other = protocol["budget"]["other_pdf_hosts"]["max"]["downloads"]
    chosen = rng.sample(arxiv, min(cap_arxiv, len(arxiv))) + rng.sample(
        other, min(cap_other, len(other))
    )

    log = RunLog("pdf", secrets)
    results = RunLog("downloads", secrets)
    finished = {r["work"] for r in results.records()}
    ctx = build_ctx(protocol, log)
    log.append({"event": "run", "command": "download", "at": iso(wall()), "chosen": len(chosen)})
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    client = make_client(protocol, "download")
    semaphore = asyncio.Semaphore(4)

    async def one(work: dict[str, Any], attempt: dict[str, Any]) -> None:
        if work["work"] in finished:
            return
        lane = "arxiv.org" if attempt["final_host"] == "arxiv.org" else "other"
        url = work["candidates"][attempt["url_index"]]["url"]
        async with semaphore:
            if not ctx.budget.can([("download_work", lane)]):
                results.append({"work": work["work"], "lane": lane,
                                "outcome": "not_sent_budget", "at": iso(wall())})
                return
            ctx.budget.take([("download_work", lane)])
            chain = await fetch_chain(
                ctx, client, rid_prefix=f"P4/{work['work']}", url=url, mode="download",
                budget_keys_for=lambda host: download_request_keys(),
                meta={"pass": "P4", "work": work["work"], "lane": lane},
            )
            record: dict[str, Any] = {"work": work["work"], "lane": lane,
                                      "url_index": attempt["url_index"], **chain_summary(chain)}
            body = chain.body
            if chain.outcome == "downloaded" and body is not None:
                digest = hashlib.sha256(body).hexdigest()
                (PDF_DIR / f"{digest}.pdf").write_bytes(body)
                record.update(bytes=len(body), sha256=digest)
                try:
                    validate_upload(body, filename=f"{digest}.pdf")
                    pages = preflight(body, filename=f"{digest}.pdf")
                    record.update(check="ok", pages=pages)
                except UploadRejected as exc:
                    record.update(check=exc.failure_class.value, detail=exc.detail[:200])
                except DocumentUnparseable as exc:
                    record.update(check=exc.failure_class.value, detail=str(exc.detail)[:200])
            elif chain.outcome == "too_large":
                record.update(check="too_large")
            results.append({**record, "at": iso(wall())})
            print(f"P4 {work['work']}: {record.get('check', chain.outcome)}", flush=True)

    try:
        await asyncio.gather(*(one(w, a) for w, a in chosen))
    finally:
        await client.aclose()
    log.append({"event": "run_end", "command": "download", "at": iso(wall()),
                "stopped_hosts": ctx.health.stopped})
    for event in ctx.events:
        print("event:", event)
    return 0


def show_works(protocol: dict[str, Any]) -> int:
    works = build_works(p1_keyword_records())
    print(json.dumps(
        [{"work": w.key, "topics": sorted(w.topics), "sources": sorted(w.sources),
          "candidates": [(c.kind, c.url, sorted(c.supplied_by)) for c in w.candidates]}
         for w in ordered(protocol, works, "P3")], indent=1)[:20000])
    print(f"{len(works)} works", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["works", "probe", "download"])
    args = parser.parse_args()
    protocol, _ = load_protocol()
    if args.command == "works":
        return show_works(protocol)
    env = load_env()
    secrets = Secrets([env.get("OPENALEX_API_KEY", ""), env.get("S2_API_KEY", "")])
    runner = probe if args.command == "probe" else download
    return asyncio.run(runner(protocol, secrets))


if __name__ == "__main__":
    sys.exit(main())
