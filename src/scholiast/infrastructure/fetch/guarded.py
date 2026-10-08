"""The PDF fetcher that may follow a third party's link (D115, D129, D130).

Every hop, the first request and each redirect, goes through the same four steps:

  1. **the URL** is checked by `check_fetch_url`: https (http is upgraded), port 443,
     no userinfo, no IP literal, the host exactly on the allowlist;
  2. **the host is resolved once**, and refused unless EVERY address is public, since
     the order of DNS answers is the attacker's to choose;
  3. **the connection goes to the address that was checked.** The request is built
     against that IP, with the real hostname as the Host header and as the TLS
     server name, so the certificate is still verified against the real host and a
     second DNS answer has nothing left to redirect. Resolving, checking and then
     letting the client resolve again is the DNS-rebinding hole the spike recorded
     as `known_gap`;
  4. **the response is read under limits**: redirects by hand (at most 5), no
     encoding (asked for identity; anything else is refused unread, because a
     decompressor can turn one read into tens of megabytes before a cap counts it),
     `%PDF-` first, and MAX_PDF_BYTES counted as the bytes arrive.

Nothing else from the environment is trusted: no proxy settings (a proxy would make
step 3 meaningless), no cookies, an honest User-Agent.

**Paced per host, and the fetcher waits** (D129). It runs in the ingestion worker,
where a wait costs no user any latency. arxiv.org gets its robots.txt's 15 s, other
hosts 5 s, measured from the previous request's end. Per process, like D123's pacer.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx
import structlog

from scholiast.application.ports.fetch import FetchAttempt, FetchedPdf, FetchFailed
from scholiast.domain.errors import FailureClass
from scholiast.domain.fetch_policy import check_fetch_url, host_allowlist, is_public_address
from scholiast.domain.upload import MAX_PDF_BYTES, PDF_MAGIC
from scholiast.infrastructure.discovery import USER_AGENT
from scholiast.infrastructure.retry_after import retry_after_seconds
from scholiast.infrastructure.tls import tls_context

MAX_REDIRECTS = 5
_REDIRECTS = frozenset({301, 302, 303, 307, 308})

#: Outcomes a retry may fix. Everything else but `too_large` means the link will not
#: give a PDF however often it is asked.
#: `deferred` is a link not tried because its host said when to come back, and
#: that time has not come (D142): the retry is the remedy, at the host's time.
_TRANSIENT_PREFIXES = ("status:429", "status:5", "timeout", "error", "deferred")

Resolver = Callable[[str], Awaitable[list[str]]]
log = structlog.get_logger(__name__)


async def system_resolver(host: str) -> list[str]:
    """Every address the system resolver gives for `host`, in its order."""
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(str(info[4][0]) for info in infos))


class HostPacer:
    """One request at a time per host, each starting a gap after the previous one ended."""

    def __init__(
        self,
        *,
        default_gap: float,
        gaps: Mapping[str, float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._default = default_gap
        self._gaps = dict(gaps or {})
        self._clock = clock
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}
        self._free_at: dict[str, float] = {}
        #: When each host said it would take requests again (D142).
        self._deferred_until: dict[str, float] = {}

    def defer(self, host: str, seconds: float) -> None:
        """Note that `host` asked for nothing more for `seconds` (D142)."""
        until = self._clock() + max(0.0, seconds)
        self._deferred_until[host] = max(self._deferred_until.get(host, float("-inf")), until)

    def deferred(self, host: str) -> float | None:
        """How much longer `host`'s own stated wait outlasts its gap, or None.

        None when the host asked for nothing, or for less than its gap already
        imposes: the slot's ordinary wait covers that, and a stated time can
        only ever lengthen a wait, never shorten one. Otherwise the caller must
        not wait here (D142): a serial worker waiting out a host's stated time
        would hold every upload behind it.
        """
        until = self._deferred_until.get(host)
        now = self._clock()
        if until is None or until <= max(now, self._free_at.get(host, float("-inf"))):
            return None
        return until - now

    @contextlib.asynccontextmanager
    async def slot(self, host: str) -> AsyncIterator[None]:
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            wait = self._free_at.get(host, float("-inf")) - self._clock()
            if wait > 0:
                await self._sleep(wait)
            try:
                yield
            finally:
                self._free_at[host] = self._clock() + self._gaps.get(host, self._default)


@dataclass(frozen=True, slots=True)
class _Redirect:
    location: str


@dataclass(frozen=True, slots=True)
class _Throttled:
    """A 429 or 503: the host is refusing for now, and may have said for how long."""

    status: int
    retry_after: float | None


def _classify(attempts: Sequence[FetchAttempt]) -> FailureClass:
    """One class for all the links, by what can help next (the port's rule)."""
    if any(a.outcome.startswith(_TRANSIENT_PREFIXES) for a in attempts):
        return FailureClass.TRANSIENT
    if any(a.outcome == "too_large" for a in attempts):
        return FailureClass.TOO_LARGE
    return FailureClass.SOURCE_UNAVAILABLE


class GuardedPdfFetcher:
    """The `PdfFetcher` port, with D115's guard on every hop."""

    def __init__(
        self,
        *,
        allowed_hosts: Iterable[str],
        pacer: HostPacer,
        resolver: Resolver = system_resolver,
        transport: httpx.AsyncBaseTransport | None = None,
        max_bytes: int = MAX_PDF_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        connect_timeout: float = 5.0,
        read_timeout: float = 30.0,
        total_timeout: float = 60.0,
    ) -> None:
        self._allowed = host_allowlist(allowed_hosts)
        self._pacer = pacer
        self._resolver = resolver
        self._transport = transport
        self._max_bytes = max_bytes
        self._max_redirects = max_redirects
        self._timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self._total = total_timeout

    async def fetch(self, links: Sequence[str]) -> FetchedPdf:
        attempts: list[FetchAttempt] = []
        for link in links:
            attempt, data, final_url = await self._follow(link)
            attempts.append(attempt)
            log.info("fetch.attempt", outcome=attempt.outcome, redirects=attempt.redirects)
            if data is not None and final_url is not None:
                return FetchedPdf(
                    data=data, link=link, final_url=final_url, attempts=tuple(attempts)
                )
        raise FetchFailed(_classify(attempts), tuple(attempts))

    async def _follow(self, link: str) -> tuple[FetchAttempt, bytes | None, str | None]:
        """One link to its end: a PDF, or the reason it gave none."""
        url = link
        for redirects in range(self._max_redirects + 1):
            verdict = check_fetch_url(url, self._allowed)
            if not verdict.allowed or verdict.url is None or verdict.host is None:
                return FetchAttempt(link, f"refused:{verdict.refusal}", redirects), None, None
            remaining = self._pacer.deferred(verdict.host)
            if remaining is not None:
                # The host said when to come back and that is not yet (D142):
                # nothing is sent, and the job is retried at the host's time
                # rather than holding the worker until then.
                return FetchAttempt(link, "deferred", redirects, retry_after=remaining), None, None
            try:
                addresses = await self._resolver(verdict.host)
            except OSError:
                return FetchAttempt(link, "error:dns", redirects), None, None
            if not addresses or not all(is_public_address(a) for a in addresses):
                return FetchAttempt(link, "refused:address", redirects), None, None
            async with self._pacer.slot(verdict.host):
                result = await self._request(verdict.url, verdict.host, addresses[0])
            if isinstance(result, _Throttled):
                if result.retry_after is not None:
                    # Every later link to this host, from any job in this
                    # process, is deferred until then (D142).
                    self._pacer.defer(verdict.host, result.retry_after)
                return (
                    FetchAttempt(
                        link, f"status:{result.status}", redirects, retry_after=result.retry_after
                    ),
                    None,
                    None,
                )
            if isinstance(result, _Redirect):
                # Resolved against the checked hostname URL, never the IP one, so a
                # relative Location stays on the host that sent it.
                url = urljoin(verdict.url, result.location)
                continue
            if isinstance(result, bytes):
                return FetchAttempt(link, "fetched", redirects), result, verdict.url
            return FetchAttempt(link, result, redirects), None, None
        return FetchAttempt(link, "too_many_redirects", self._max_redirects), None, None

    async def _request(
        self, url: str, host: str, address: str
    ) -> _Redirect | _Throttled | bytes | str:
        """One hop, connected to `address`: a redirect, the PDF, or a failure outcome."""
        target = httpx.URL(url).copy_with(host=address)
        headers = {
            "Host": host,
            "User-Agent": USER_AGENT,
            "Accept": "application/pdf",
            "Accept-Encoding": "identity",
        }
        try:
            async with (
                httpx.AsyncClient(
                    verify=tls_context(),
                    follow_redirects=False,
                    trust_env=False,
                    timeout=self._timeout,
                    transport=self._transport,
                ) as client,
                asyncio.timeout(self._total),
            ):
                request = client.build_request(
                    "GET", target, headers=headers, extensions={"sni_hostname": host}
                )
                response = await client.send(request, stream=True)
                try:
                    return await self._read(response)
                finally:
                    await response.aclose()
        except (TimeoutError, httpx.TimeoutException):
            return "timeout"
        except httpx.HTTPError as exc:
            return f"error:{type(exc).__name__}"

    async def _read(self, response: httpx.Response) -> _Redirect | _Throttled | bytes | str:
        status = response.status_code
        if status in (429, 503):
            return _Throttled(status, retry_after_seconds(response))
        if status in _REDIRECTS:
            location = response.headers.get("location")
            return _Redirect(location) if location else f"status:{status}"
        if status != 200:
            return f"status:{status}"
        if response.headers.get("content-encoding", "identity").strip().lower() not in (
            "",
            "identity",
        ):
            return "encoded"
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self._max_bytes:
            return "too_large"
        body = bytearray()
        async for chunk in response.aiter_raw():
            body += chunk
            if len(body) > self._max_bytes:
                return "too_large"
            if len(body) >= len(PDF_MAGIC) and not body.startswith(PDF_MAGIC):
                return "not_pdf"
        if not body.startswith(PDF_MAGIC):
            return "not_pdf"
        return bytes(body)
