"""Shared machinery for the M4 discovery spike (D116, spike/m4-protocol.json).

Throwaway by design, like the rest of spike/: it exists to produce the evidence in
SPIKE-FINDINGS-M4.md, not to become the adapters phase 2 writes or the guard phase 5
writes. Five things it must get right anyway, because they happen against other
people's services and on the owner's disk:

  pacing       every host's gap from the protocol, enforced before each request
  budget       every cap, checked before each request and never exceeded
  containment  https only, public addresses only, no automatic redirects, byte caps
  secrets      keys only in headers to their own host, never in a written file
  S2 locality  nothing Semantic Scholar returns reaches a tracked file

The protocol is loaded through `load_protocol`, which refuses to run if its
canonical hash differs from the one the owner approved. Editing the protocol after
approval therefore stops the spike rather than silently measuring something else.
"""

from __future__ import annotations

import asyncio
import contextlib
import email.utils
import hashlib
import ipaddress
import json
import math
import re
import socket
import time
import unicodedata
from collections import Counter, defaultdict
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "spike" / "m4-protocol.json"
ENV_PATH = ROOT / "spike" / ".env"
RESULTS_DIR = ROOT / "spike" / "results"
LOCAL_DIR = ROOT / "var" / "spike"
RUN_DIR = LOCAL_DIR / "run"
PDF_DIR = LOCAL_DIR / "pdf"
POOLS_PATH = ROOT / "data" / "eval" / "discovery-pools.json.gz"
POOLS_MANIFEST_PATH = ROOT / "data" / "eval" / "discovery-pools.manifest.json"

#: The canonical hash the owner approved: D116 on 2026-09-30, amended by DEV-2 on
#: 2026-10-01 (D117). A protocol that hashes to anything else is not the approved
#: protocol, and nothing runs. The first 15 requests ran under the original.
APPROVED_PROTOCOL_SHA256 = "6f46de9af888c2998f9c3faf61d8c902692e8e688db012aef307a098df820884"
ORIGINAL_PROTOCOL_SHA256 = "6a2809025626b29b96119cd4d06639fd3341cd325c74c121e82ba5e9e8677a34"

#: Implementation reading of "at most 5 hops": at most 5 redirects are followed, so
#: a chain is at most 6 requests. Every one of them counts against the budget.
MAX_REDIRECTS = 5

#: A Retry-After longer than this stops the host for the session rather than
#: stalling every query behind it. Not in the protocol; recorded if it ever fires.
RETRY_AFTER_CEILING_S = 600.0

#: Added to every pacing gap, so the strict checker never sees a gap that the
#: event loop's timer resolution shaved a few milliseconds off.
PACING_MARGIN_S = 0.15

#: API responses are small; this only stops a pathological one filling memory.
API_BODY_CAP = 20 * 1024 * 1024

#: Where the budget names a source differently from `sources`.
BUDGET_NAME = {"openalex": "openalex", "semantic_scholar": "semantic_scholar", "arxiv": "arxiv_api"}
API_HOST = {
    "openalex": "api.openalex.org",
    "semantic_scholar": "api.semanticscholar.org",
    "arxiv": "export.arxiv.org",
}
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})


# -- protocol ----------------------------------------------------------------


def canonical_sha256(obj: Any) -> str:
    """The protocol's hashing rule: canonical JSON, not the file's bytes (D116)."""
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def load_protocol() -> tuple[dict[str, Any], str]:
    protocol: dict[str, Any] = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    digest = canonical_sha256(protocol)
    if digest != APPROVED_PROTOCOL_SHA256:
        raise SystemExit(
            f"protocol hash {digest} is not the approved {APPROVED_PROTOCOL_SHA256}; "
            "the protocol changed after approval, so nothing runs"
        )
    return protocol, digest


# -- environment and secrets -------------------------------------------------


def load_env() -> dict[str, str]:
    """Read spike/.env without printing anything from it."""
    values: dict[str, str] = {}
    if not ENV_PATH.exists():
        return values
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


class Secrets:
    """Every secret value in play, so any text can be checked before it is written."""

    def __init__(self, values: Iterable[str]) -> None:
        self._values = [v for v in values if v and len(v) >= 8]

    def assert_absent(self, text: str, *, where: str) -> None:
        for value in self._values:
            if value in text:
                raise SystemExit(f"a secret value was about to be written to {where}; aborted")


# -- normalisation (protocol: query_construction, definitions) -----------------


def norm_query(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def openalex_query(text: str) -> str:
    """DEV-2 (D117): OpenAlex reads '?' and '*' as wildcards and refuses them in stemmed search."""
    return re.sub(r"\s+", " ", re.sub(r"[?*]", " ", norm_query(text))).strip()


def s2_query(text: str) -> str:
    return re.sub(r"\s+", " ", norm_query(text).replace("-", " ")).strip()


def arxiv_query(text: str, stopwords: Iterable[str]) -> str:
    stop = set(stopwords)
    seen: set[str] = set()
    terms: list[str] = []
    for token in re.split(r"[^0-9a-z]+", _letters_and_digits(norm_query(text))):
        if len(token) < 2 or token in stop or token in seen:
            continue
        seen.add(token)
        terms.append(token)
    return " AND ".join(f"all:{t}" for t in terms)


def _letters_and_digits(text: str) -> str:
    """Keep only letters and digits (any script), everything else a space."""
    return "".join(ch if ch.isalnum() else " " for ch in text)


def norm_title(text: str | None) -> str:
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", _letters_and_digits(folded)).strip()


_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")


def norm_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip().lower()
    for prefix in _DOI_PREFIXES:
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi or None


_ARXIV_NEW = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?$")
_ARXIV_OLD = re.compile(r"^([a-z\-]+(?:\.[a-z]{2})?/\d{7})(v\d+)?$", re.IGNORECASE)
_ARXIV_URL_PREFIXES = (
    "https://arxiv.org/abs/",
    "http://arxiv.org/abs/",
    "https://arxiv.org/pdf/",
    "http://arxiv.org/pdf/",
    "https://export.arxiv.org/abs/",
    "http://export.arxiv.org/abs/",
)


def norm_arxiv(value: str | None) -> str | None:
    """An arXiv id without prefix, `.pdf` or version, or None if it is not one."""
    if not value:
        return None
    text = value.strip()
    lowered = text.lower()
    if lowered.startswith("10.48550/arxiv."):
        text = text[len("10.48550/arxiv.") :]
    elif lowered.startswith("arxiv:"):
        text = text[len("arxiv:") :]
    else:
        for prefix in _ARXIV_URL_PREFIXES:
            if lowered.startswith(prefix):
                text = text[len(prefix) :]
                break
    if text.lower().endswith(".pdf"):
        text = text[:-4]
    text = text.strip("/")
    match = _ARXIV_NEW.match(text) or _ARXIV_OLD.match(text)
    return match.group(1) if match else None


def arxiv_from_url(url: str | None) -> str | None:
    if not url:
        return None
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if host not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
        return None
    path = parts.path
    for marker in ("/abs/", "/pdf/"):
        if path.startswith(marker):
            return norm_arxiv(path[len(marker) :])
    return None


# -- addresses and URLs (protocol: http_safety) --------------------------------


def address_allowed(text: str) -> bool:
    """True only for a globally routable unicast address.

    IPv4-mapped IPv6 is unwrapped first: `::ffff:127.0.0.1` is loopback wearing a
    disguise. Multicast is tested explicitly because `is_global` alone says True
    for public multicast ranges.
    """
    ip = ipaddress.ip_address(text.split("%", 1)[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_multicast
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    return ip.is_global


async def resolve(host: str) -> list[str]:
    """Every address the host resolves to (A and AAAA), or an IP literal as given."""
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return [host]
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    return sorted({str(info[4][0]) for info in infos})


@dataclass(frozen=True)
class UrlCheck:
    ok: bool
    url: str | None
    host: str | None
    upgraded: bool
    reason: str | None


def check_url(url: str) -> UrlCheck:
    """Scheme, port and shape, before any resolution (http_safety)."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return UrlCheck(False, None, None, False, "malformed")
    scheme = parts.scheme.lower()
    upgraded = False
    if scheme == "http":
        upgraded = True
        scheme = "https"
        if port == 80:
            port = None
    if scheme != "https":
        return UrlCheck(False, None, None, False, f"scheme:{parts.scheme or 'none'}")
    if port not in (None, 443):
        return UrlCheck(False, None, None, upgraded, f"port:{port}")
    if parts.username or parts.password:
        return UrlCheck(False, None, None, upgraded, "userinfo")
    host = (parts.hostname or "").lower()
    if not host:
        return UrlCheck(False, None, None, upgraded, "no-host")
    netloc = host if ":" not in host else f"[{host}]"
    rebuilt = urlunsplit(("https", netloc, parts.path or "/", parts.query, ""))
    return UrlCheck(True, rebuilt, host, upgraded, None)


# -- time --------------------------------------------------------------------


def wall() -> float:
    return time.time()


def iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds")


# -- run log -----------------------------------------------------------------


class RunLog:
    """Append-only JSONL under var/spike/run/: the resume source and the raw record.

    Local only. A `sending` line is written before a request leaves and an
    `outcome` line after it, so a crash between the two leaves a request that is
    known to have been sent and will never be sent again (rules.no_repeats).
    """

    def __init__(self, name: str, secrets: Secrets, directory: Path = RUN_DIR) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{name}.jsonl"
        self._secrets = secrets

    def append(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str)
        self._secrets.assert_absent(line, where=str(self.path))
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def records(self) -> list[dict[str, Any]]:
        return read_jsonl(self.path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def started_rids(records: Iterable[dict[str, Any]]) -> set[str]:
    """Every request id that was ever sent or decided, whatever came after."""
    return {r["rid"] for r in records if "rid" in r}


# -- pacing ------------------------------------------------------------------


@dataclass(frozen=True)
class PacingRule:
    min_gap_s: float
    from_end: bool


def pacing_rule(protocol: dict[str, Any], host: str) -> PacingRule:
    rules = protocol["pacing"]
    rule = rules.get(host) or rules["any_other_host"]
    return PacingRule(
        float(rule["min_gap_s"]), rule.get("measured_from") == "previous response end"
    )


class NotAdmitted(Exception):
    """A request refused at the head of its host's queue, with the reason it records."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Pacer:
    """Per-host gaps, one request in flight per host, and Retry-After.

    Wall-clock time throughout, so the checker reads the same clock the pacer
    enforced against.
    """

    def __init__(self, protocol: dict[str, Any]) -> None:
        self._protocol = protocol
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._last_start: dict[str, float] = {}
        self._last_end: dict[str, float] = {}
        self._not_before: dict[str, float] = {}

    def seed(self, host: str, start: float, end: float | None) -> None:
        """Carry the last request of an earlier run forward, for resumes.

        A request logged as sent with no end crashed mid-flight; its end is known
        only to be no later than now, so now is the safe value to measure from.
        """
        if end is None:
            end = wall()
        if start > self._last_start.get(host, 0.0):
            self._last_start[host] = start
        if end > self._last_end.get(host, 0.0):
            self._last_end[host] = end

    def earliest(self, host: str) -> float:
        rule = pacing_rule(self._protocol, host)
        reference = self._last_end.get(host) if rule.from_end else self._last_start.get(host)
        earliest = 0.0 if reference is None else reference + rule.min_gap_s + PACING_MARGIN_S
        return max(earliest, self._not_before.get(host, 0.0))

    def defer(self, host: str, seconds: float) -> None:
        self.defer_until(host, wall() + seconds)

    def defer_until(self, host: str, when: float) -> None:
        self._not_before[host] = max(self._not_before.get(host, 0.0), when)

    @contextlib.asynccontextmanager
    async def slot(
        self, host: str, admit: Callable[[], str | None] | None = None
    ) -> AsyncIterator[float]:
        """The host's next turn. `admit` is asked again once the turn comes.

        A request decides it may be sent, then queues behind others to the same
        host; by its turn the host may have been stopped or a cap reached. So the
        decision is re-taken under the host's lock: before waiting (no point
        waiting out a Retry-After for a stopped host) and after (a shared cap can
        fill while it sleeps). A refusal raises NotAdmitted and uses no slot.
        """
        async with self._locks[host]:
            _admit(admit)  # MUTATION-POINT: admit-before-wait
            while (delay := self.earliest(host) - wall()) > 0:
                await asyncio.sleep(delay)
            _admit(admit)  # MUTATION-POINT: admit-after-wait
            start = wall()
            self._last_start[host] = start
            try:
                yield start
            finally:
                self._last_end[host] = wall()


def _admit(admit: Callable[[], str | None] | None) -> None:
    if admit is not None and (reason := admit()) is not None:
        raise NotAdmitted(reason)


# -- budget ------------------------------------------------------------------


def api_budget_keys(source: str, kind: str) -> list[tuple[str, ...]]:
    return [("api", BUDGET_NAME[source], kind)]


def probe_budget_keys(host: str) -> list[tuple[str, ...]]:
    lane = "arxiv.org" if host == "arxiv.org" else host
    return [("probe_host", lane), ("probe_total",)]


def download_request_keys() -> list[tuple[str, ...]]:
    return [("download_request",)]


class Budget:
    """Every cap in the protocol, enforced before a request is sent."""

    def __init__(self, protocol: dict[str, Any]) -> None:
        b = protocol["budget"]
        self._api = {
            ("api", "openalex", "search"): b["openalex"]["max"]["searches"],
            ("api", "openalex", "lookup"): b["openalex"]["max"]["lookups"],
            ("api", "semantic_scholar", "search"): b["semantic_scholar"]["max"]["searches"],
            ("api", "semantic_scholar", "lookup"): b["semantic_scholar"]["max"]["lookups"],
            ("api", "arxiv_api", "search"): b["arxiv_api"]["max"]["queries"],
            ("api", "arxiv_api", "lookup"): b["arxiv_api"]["max"]["lookups"],
        }
        self._arxiv_probe = b["arxiv_org_pdfs"]["max"]["probe_requests"]
        self._other_probe = b["other_pdf_hosts"]["max"]["probe_requests_per_host"]
        self._probe_total = b["pdf_probe_requests_total_max"]
        self._download_requests = b["download_requests_total_max"]
        self._download_works = {
            "arxiv.org": b["arxiv_org_pdfs"]["max"]["downloads"],
            "other": b["other_pdf_hosts"]["max"]["downloads"],
        }
        self.used: Counter[tuple[str, ...]] = Counter()

    def cap(self, key: tuple[str, ...]) -> int:
        if key[0] == "api":
            return int(self._api[key])
        if key[0] == "probe_host":
            return int(self._arxiv_probe if key[1] == "arxiv.org" else self._other_probe)
        if key == ("probe_total",):
            return int(self._probe_total)
        if key == ("download_request",):
            return int(self._download_requests)
        if key[0] == "download_work":
            return int(self._download_works[key[1]])
        raise KeyError(key)

    def can(self, keys: Iterable[tuple[str, ...]]) -> bool:
        return all(self.used[k] < self.cap(k) for k in keys)

    def take(self, keys: Iterable[tuple[str, ...]]) -> None:
        for key in keys:
            if self.used[key] >= self.cap(key):
                raise RuntimeError(f"budget exceeded for {key}")
            self.used[key] += 1

    def replay(self, records: Iterable[dict[str, Any]]) -> None:
        """Rebuild usage from an earlier run's log, so a resume cannot exceed a cap."""
        for record in records:
            if record.get("event") == "sending":
                for key in record.get("budget_keys", []):
                    self.used[tuple(key)] += 1


# -- host health (protocol: pacing.host_stop, pacing.retry_after) ---------------


class HostHealth:
    def __init__(self) -> None:
        self._consecutive: Counter[str] = Counter()
        self.stopped: dict[str, str] = {}

    def observe(self, host: str, status: int | None) -> None:
        if status is None:
            return
        if status == 429 or status >= 500:
            self._consecutive[host] += 1
            if self._consecutive[host] >= 5 and host not in self.stopped:
                self.stopped[host] = "5 consecutive 429/5xx"
        else:
            self._consecutive[host] = 0

    def stop(self, host: str, reason: str) -> None:
        self.stopped.setdefault(host, reason)


def parse_retry_after(value: str | None, received_at: float | None = None) -> float | None:
    """Seconds to wait, counted from when the response was received."""
    if not value:
        return None
    value = value.strip()
    if value.isdigit():
        return float(value)
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    return max(0.0, when.timestamp() - (wall() if received_at is None else received_at))


KEPT_HEADERS = (
    "content-type",
    "content-length",
    "content-range",
    "content-encoding",
    "retry-after",
    "location",
)


def kept_headers(headers: Any) -> dict[str, str]:
    """The response headers the protocol asks for, and nothing that could carry a secret."""
    out: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in KEPT_HEADERS or lowered.startswith(("x-ratelimit", "ratelimit")):
            out[lowered] = value
    return out


# -- statistics (protocol: statistics) -----------------------------------------


def nearest_rank(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100.0 * len(ordered)) - 1)
    return ordered[index]


def wilson(successes: int, n: int, z: float = 1.959963984540054) -> dict[str, Any]:
    if n == 0:
        return {"k": successes, "n": n, "rate": None, "lo": None, "hi": None}
    phat = successes / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return {
        "k": successes,
        "n": n,
        "rate": round(phat, 4),
        "lo": round(max(0.0, centre - half), 4),
        "hi": round(min(1.0, centre + half), 4),
    }


def s2_hash(value: str) -> str:
    """How an S2 identifier or S2-only URL appears in a tracked file."""
    return "s2h:" + hashlib.sha256(("s2:" + value).encode("utf-8")).hexdigest()[:24]
