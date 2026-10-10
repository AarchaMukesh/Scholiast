"""HTTP for the M4 spike: paced, budgeted API requests and guarded PDF fetch chains.

Two entry points, and both follow the same order for every request, because the
order is what makes the protocol's rules checkable afterwards:

  1. decide whether it may be sent at all (already sent? host stopped? budget?)
  2. wait for the host's pacing slot, and decide again once it comes
  3. write a `sending` line to the run log, take the budget
  4. send, read the body up to its cap, time it
  5. write an `outcome` line; apply Retry-After and the host-stop rule

A request that is decided against in step 1 or 2 gets a single `not_sent` line
with its reason, so every planned request has a logged outcome (acceptance 4).
"""

from __future__ import annotations

import asyncio
import functools
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx
from m4_common import (
    API_BODY_CAP,
    MAX_REDIRECTS,
    REDIRECT_STATUSES,
    RETRY_AFTER_CEILING_S,
    Budget,
    HostHealth,
    NotAdmitted,
    Pacer,
    RunLog,
    address_allowed,
    check_url,
    iso,
    kept_headers,
    parse_retry_after,
    resolve,
    started_rids,
    wall,
)


@dataclass
class Ctx:
    """Everything a request needs to obey the protocol."""

    protocol: dict[str, Any]
    pacer: Pacer
    budget: Budget
    health: HostHealth
    log: RunLog
    done: set[str]
    events: list[str] = field(default_factory=list)


def build_ctx(protocol: dict[str, Any], log: RunLog, session: int | None = None) -> Ctx:
    """A context that carries an earlier run's state forward, so a resume obeys it.

    Budget usage, each host's last start and end, the host-stop rule and any
    Retry-After still in force are all rebuilt from the log. Host health is
    replayed only from the same session (protocol: "nothing more is sent to it
    this session"); the PDF log has no sessions, so all of it counts.
    """
    records = log.records()
    budget = Budget(protocol)
    budget.replay(records)
    pacer = Pacer(protocol)
    ends = {r["rid"]: r.get("t_end") for r in records if r.get("event") == "outcome"}
    for r in records:
        if r.get("event") == "sending":
            pacer.seed(r["host"], r["t_start"], ends.get(r["rid"]))
    ctx = Ctx(protocol, pacer, budget, HostHealth(), log, started_rids(records))
    for r in records:
        if r.get("event") != "outcome" or (session is not None and r.get("session") != session):
            continue
        ctx.health.observe(r["host"], r.get("status"))
        _apply_retry_after(ctx, r["host"], (r.get("headers") or {}).get("retry-after"), r["t_end"])
    return ctx


def classify(status: int) -> str:
    if 200 <= status < 300:
        return "ok"
    if status in REDIRECT_STATUSES:
        return "redirect"
    if status == 429:
        return "throttled"
    if status >= 500:
        return "server_error"
    return "client_error"


def _refusal(ctx: Ctx, host: str, keys: list[tuple[str, ...]]) -> str | None:
    """Why a request may not be sent now, or None. Asked again when its turn comes."""
    if host in ctx.health.stopped:
        return "not_sent_host_stopped"
    if not ctx.budget.can(keys):
        return "not_sent_budget"
    return None


def _apply_retry_after(ctx: Ctx, host: str, value: str | None, received_at: float) -> None:
    """Retry-After, counted from when the response arrived (protocol: pacing)."""
    retry = parse_retry_after(value, received_at)
    if retry is None:
        return
    if retry > RETRY_AFTER_CEILING_S:
        ctx.health.stop(host, f"Retry-After {retry:.0f}s above the {RETRY_AFTER_CEILING_S:.0f}s ceiling")
        ctx.events.append(f"{host}: stopped, Retry-After {retry:.0f}s")
    else:
        ctx.pacer.defer_until(host, received_at + retry)


def _after_response(ctx: Ctx, host: str, status: int | None, headers: dict[str, str]) -> None:
    """Retry-After and the host-stop rule (protocol: pacing)."""
    ctx.health.observe(host, status)
    _apply_retry_after(ctx, host, headers.get("retry-after"), wall())


def _not_sent(ctx: Ctx, rid: str, outcome: str, **fields: Any) -> dict[str, Any]:
    # `outcome`, not `reason`: refusals log a `reason` field of their own.
    record = {**fields, "event": "not_sent", "rid": rid, "outcome": outcome, "at": iso(wall())}
    ctx.log.append(record)
    ctx.done.add(rid)
    return record


async def api_request(
    ctx: Ctx,
    client: httpx.AsyncClient,
    *,
    rid: str,
    host: str,
    budget_keys: list[tuple[str, ...]],
    method: str,
    url: str,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
    meta: dict[str, Any],
    total_timeout: float,
) -> tuple[dict[str, Any], bytes | None] | None:
    """One API request. Returns (outcome record, body), or None if already done."""
    if rid in ctx.done:
        return None
    admit = functools.partial(_refusal, ctx, host, budget_keys)
    if reason := admit():
        return _not_sent(ctx, rid, reason, host=host, **meta), None

    request = client.build_request(method, url, params=params, json=json_body)
    request_url = str(request.url)
    try:
        async with ctx.pacer.slot(host, admit) as t_start:
            ctx.budget.take(budget_keys)
            ctx.log.append(
                {
                    **meta,
                    "event": "sending",
                    "rid": rid,
                    "host": host,
                    "method": method,
                    "url": request_url,
                    "t_start": t_start,
                    "budget_keys": [list(k) for k in budget_keys],
                }
            )
            ctx.done.add(rid)
            status: int | None = None
            headers: dict[str, str] = {}
            body: bytes | None = None
            outcome = "ok"
            error = None
            history = 0
            headers_s = None
            t0 = time.perf_counter()
            try:
                async with asyncio.timeout(total_timeout):
                    response = await client.send(request, stream=True)
                    try:
                        headers_s = time.perf_counter() - t0
                        status = response.status_code
                        headers = kept_headers(response.headers)
                        history = len(response.history)
                        chunks: list[bytes] = []
                        size = 0
                        async for chunk in response.aiter_bytes():
                            size += len(chunk)
                            if size > API_BODY_CAP:
                                outcome = "too_large"
                                break
                            chunks.append(chunk)
                        body = b"".join(chunks)
                    finally:
                        await response.aclose()
                if outcome != "too_large":
                    outcome = classify(status)
            except TimeoutError:
                outcome, error = "timeout", f"no complete response within {total_timeout}s"
            except httpx.TimeoutException as exc:
                outcome, error = "timeout", type(exc).__name__
            except httpx.ConnectError as exc:
                outcome, error = "connect_error", f"{type(exc).__name__}: {exc}"[:300]
            except httpx.HTTPError as exc:
                outcome, error = "transport_error", f"{type(exc).__name__}: {exc}"[:300]
            except Exception as exc:  # sent, so it needs an outcome (acceptance 4), not a crash
                outcome, error = "client_exception", f"{type(exc).__name__}: {exc}"[:300]
            latency = time.perf_counter() - t0
    except NotAdmitted as exc:
        # Queued behind another request to this host, and by its turn the host was
        # stopped or a cap was reached (budget.enforcement, pacing.host_stop).
        return _not_sent(ctx, rid, exc.reason, host=host, **meta), None
    client.cookies.clear()
    record = {
        **meta,
        "event": "outcome",
        "rid": rid,
        "host": host,
        "url": request_url,
        "t_start": t_start,
        "t_end": wall(),
        "latency_s": round(latency, 4),
        "headers_s": None if headers_s is None else round(headers_s, 4),
        "status": status,
        "outcome": outcome,
        "error": error,
        "bytes": None if body is None else len(body),
        "headers": headers,
        "client_redirects": history,
    }
    ctx.log.append(record)
    _after_response(ctx, host, status, headers)
    return record, body


@dataclass
class Chain:
    """A guarded fetch: every hop checked, paced, budgeted and logged."""

    outcome: str
    hops: list[dict[str, Any]]
    body: bytes | None = None
    upgraded: bool = False

    @property
    def hosts(self) -> list[str]:
        return [h["host"] for h in self.hops if h.get("host") and h.get("sent")]

    @property
    def final_host(self) -> str | None:
        sent = [h for h in self.hops if h.get("sent")]
        return sent[-1]["host"] if sent else None


async def fetch_chain(
    ctx: Ctx,
    client: httpx.AsyncClient,
    *,
    rid_prefix: str,
    url: str,
    mode: str,
    budget_keys_for: Any,
    meta: dict[str, Any],
) -> Chain:
    """Probe (first 2 KiB) or download (up to 50 MiB) one URL under http_safety."""
    safety = ctx.protocol["http_safety"]
    timeouts = ctx.protocol["timeouts_s"]
    probe = mode == "probe"
    cap = int(safety["byte_caps"]["probe" if probe else "download"])
    total = float(timeouts["probe_total" if probe else "download_total"])
    headers = {"Accept-Encoding": "identity"}
    if probe:
        headers["Range"] = "bytes=0-2047"

    hops: list[dict[str, Any]] = []
    upgraded = False
    current = url
    for index in range(MAX_REDIRECTS + 1):
        rid = f"{rid_prefix}/h{index}"
        if rid in ctx.done:
            return Chain("interrupted_earlier", hops, None, upgraded)
        hop: dict[str, Any] = {"rid": rid, "hop": index, "url": current, "sent": False}
        hops.append(hop)
        checked = check_url(current)
        if not checked.ok or checked.url is None or checked.host is None:
            hop.update(outcome="refused", reason=checked.reason)
            _not_sent(ctx, rid, "refused", reason=checked.reason, url=current, **meta)
            return Chain("refused", hops, None, upgraded)
        upgraded = upgraded or checked.upgraded
        target, host = checked.url, checked.host
        hop.update(url=target, host=host, upgraded=checked.upgraded)
        try:
            # Built before anything is logged: httpx refuses some URLs that pass
            # check_url (a NUL, a bad IDNA label) with errors that are not HTTPError.
            request = client.build_request("GET", target, headers=headers)
        except (httpx.InvalidURL, ValueError) as exc:  # IDNA errors are ValueErrors
            reason = f"malformed:{type(exc).__name__}"
            hop.update(outcome="refused", reason=reason)
            _not_sent(ctx, rid, "refused", host=host, url=target, reason=reason, **meta)
            return Chain("refused", hops, None, upgraded)
        try:
            addresses = await resolve(host)
        except (OSError, UnicodeError) as exc:  # UnicodeError: a label IDNA cannot encode
            hop.update(outcome="dns_error", reason=str(exc)[:200])
            _not_sent(ctx, rid, "dns_error", host=host, url=target, reason=str(exc)[:200], **meta)
            return Chain("dns_error", hops, None, upgraded)
        hop["addresses"] = addresses
        if not addresses or not all(address_allowed(a) for a in addresses):  # MUTATION-POINT: guard-address
            hop.update(outcome="refused", reason="address")
            _not_sent(ctx, rid, "refused", host=host, url=target, reason="address",
                      addresses=addresses, **meta)
            return Chain("refused", hops, None, upgraded)
        keys = budget_keys_for(host)
        admit = functools.partial(_refusal, ctx, host, keys)
        if reason := admit():
            hop.update(outcome=reason)
            _not_sent(ctx, rid, reason, host=host, url=target, **meta)
            return Chain(reason, hops, None, upgraded)

        status: int | None = None
        got: dict[str, str] = {}
        data = b""
        outcome = "ok"
        error = None
        history = 0
        try:
            async with ctx.pacer.slot(host, admit) as t_start:
                ctx.budget.take(keys)
                ctx.log.append(
                    {
                        **meta,
                        "event": "sending",
                        "rid": rid,
                        "host": host,
                        "method": "GET",
                        "url": target,
                        "t_start": t_start,
                        "addresses": addresses,
                        "budget_keys": [list(k) for k in keys],
                        "mode": mode,
                    }
                )
                ctx.done.add(rid)
                hop["sent"] = True
                t0 = time.perf_counter()
                try:
                    async with asyncio.timeout(total):
                        response = await client.send(request, stream=True)
                        try:
                            status = response.status_code
                            got = kept_headers(response.headers)
                            history = len(response.history)
                            if 200 <= status < 300:
                                chunks: list[bytes] = []
                                size = 0
                                async for chunk in response.aiter_bytes():
                                    chunks.append(chunk)
                                    size += len(chunk)
                                    if probe and size >= cap:
                                        break
                                    if not probe and size > cap:
                                        outcome = "too_large"
                                        break
                                data = b"".join(chunks)
                                if probe:
                                    data = data[:cap]
                        finally:
                            await response.aclose()
                except TimeoutError:
                    outcome, error = "timeout", f"no complete response within {total}s"
                except httpx.TimeoutException as exc:
                    outcome, error = "timeout", type(exc).__name__
                except httpx.ConnectError as exc:
                    outcome, error = "connect_error", f"{type(exc).__name__}: {exc}"[:300]
                except httpx.HTTPError as exc:
                    outcome, error = "transport_error", f"{type(exc).__name__}: {exc}"[:300]
                except Exception as exc:  # sent, so it needs an outcome (acceptance 4), not a crash
                    outcome, error = "client_exception", f"{type(exc).__name__}: {exc}"[:300]
                elapsed = time.perf_counter() - t0
        except NotAdmitted as exc:
            # Queued behind another request to this host, and by its turn the host was
            # stopped or a cap was reached (budget.enforcement, pacing.host_stop).
            hop.update(outcome=exc.reason)
            _not_sent(ctx, rid, exc.reason, host=host, url=target, **meta)
            return Chain(exc.reason, hops, None, upgraded)
        client.cookies.clear()

        if status is not None and outcome == "ok":
            outcome = classify(status)
        is_pdf = data.startswith(b"%PDF-")
        location = got.get("location")
        hop.update(
            status=status,
            outcome=outcome,
            error=error,
            content_type=got.get("content-type"),
            content_length=got.get("content-length"),
            content_range=got.get("content-range"),
            content_encoding=got.get("content-encoding"),
            bytes_read=len(data),
            is_pdf=is_pdf,
            location=location,
            seconds=round(elapsed, 4),
        )
        ctx.log.append(
            {
                **meta,
                "event": "outcome",
                "rid": rid,
                "host": host,
                "url": target,
                "t_start": t_start,
                "t_end": wall(),
                "latency_s": round(elapsed, 4),
                "status": status,
                "outcome": outcome,
                "error": error,
                "bytes": len(data),
                "is_pdf": is_pdf,
                "headers": got,
                "client_redirects": history,
                "mode": mode,
            }
        )
        _after_response(ctx, host, status, got)

        if outcome == "redirect" and location:
            current = urljoin(target, location)
            continue
        if outcome == "ok":
            if not probe:
                return Chain("downloaded" if is_pdf else "not_pdf", hops, data, upgraded)
            return Chain("fetched" if is_pdf else "not_pdf", hops, None, upgraded)
        return Chain(outcome, hops, None, upgraded)
    return Chain("too_many_redirects", hops, None, upgraded)
