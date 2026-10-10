"""What the PDF fetcher may ask for and where it may connect (D115, D129, D130).

M4 is the first thing that dereferences a URL it did not choose: OA links come from
third-party records, so any of them can point at `169.254.169.254`, at a service on
the private network, or at a host that redirects there. These are the rules, as pure
functions, so every one is testable without a socket and the adapter cannot quietly
weaken one.

**URLs.** https only; an http link is upgraded to https on the same host and path,
because the alternative is either refusing a working paper or sending a request in
plaintext. Port 443 only. No userinfo. No IP-literal host, even a public one: the
allowlist names hosts, and numeric spellings (`2130706433`, `0x7f.1`) are how
address checks get dodged. The host must be exactly on the allowlist (D130), after
lowercasing and dropping a trailing dot.

**Addresses.** Only globally routable unicast. IPv4 carried inside IPv6 is unwrapped
and judged as IPv4: an IPv4-mapped address, NAT64's well-known prefix and 6to4 can
all name a private IPv4 address while looking like a public IPv6 one. Teredo and
local-use NAT64 are refused whole: tunnels nothing here needs to reach.

**Some checks repeat Python's own.** On Python 3.12.9, `is_global` already refuses
loopback, link-local, IPv4-mapped private and local-use NAT64 addresses, so those
explicit checks change nothing today. They stay because `ipaddress`'s tables have
been wrong before (CVE-2024-4032 corrected several in 3.12.4), and this is the one
function where a table error is an SSRF hole.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum, unique
from urllib.parse import urlsplit, urlunsplit

from scholiast.domain.discovery import LinkKind, PaperCandidate

#: RFC 6052's well-known NAT64 prefix, which carries IPv4 in its last 32 bits: a
#: gateway that translates `64:ff9b::a9fe:a9fe` reaches 169.254.169.254.
_NAT64 = ipaddress.ip_network("64:ff9b::/96")
#: RFC 8215's local-use NAT64 prefix. Not globally routable, and where it carries
#: IPv4 depends on how the operator sized it, so it is refused whole.
_NAT64_LOCAL = ipaddress.ip_network("64:ff9b:1::/48")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> list[ipaddress.IPv4Address]:
    """Every IPv4 address an IPv6 address can carry to the network beyond it."""
    found: list[ipaddress.IPv4Address] = []
    if ip.ipv4_mapped is not None:
        found.append(ip.ipv4_mapped)
    if ip.sixtofour is not None:
        found.append(ip.sixtofour)
    if ip in _NAT64:
        found.append(ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF))
    return found


def _routable(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    # Multicast is tested explicitly because `is_global` alone says True for some
    # public multicast ranges.
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


def is_public_address(text: str) -> bool:
    """True only for a globally routable unicast address, and every IPv4 it carries."""
    try:
        ip = ipaddress.ip_address(text.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _NAT64_LOCAL or ip.teredo is not None:
            return False
        embedded = _embedded_ipv4(ip)
        if embedded:
            return all(_routable(v4) for v4 in embedded)
    return _routable(ip)


@unique
class Refusal(StrEnum):
    """Why a URL will not be requested."""

    SCHEME = "scheme"
    PORT = "port"
    USERINFO = "userinfo"
    NO_HOST = "no_host"
    IP_LITERAL = "ip_literal"
    HOST_NOT_ALLOWED = "host_not_allowed"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class UrlVerdict:
    """A URL judged: the https URL to request and its host, or the reason it is refused."""

    url: str | None
    host: str | None
    refusal: Refusal | None = None
    upgraded: bool = False

    @property
    def allowed(self) -> bool:
        return self.refusal is None


def _refused(reason: Refusal) -> UrlVerdict:
    return UrlVerdict(url=None, host=None, refusal=reason)


def host_allowlist(hosts: Iterable[str]) -> frozenset[str]:
    """An allowlist in the form `check_fetch_url` compares hosts in: lowercase, no final dot.

    One function, because the fetcher and the add endpoint must agree exactly on
    what is allowed (D137): an add that accepted a link the fetcher then refused
    would create the doomed job the refusal exists to prevent.
    """
    return frozenset(host.lower().rstrip(".") for host in hosts)


def check_fetch_url(url: str, allowed_hosts: frozenset[str]) -> UrlVerdict:
    """Whether the fetcher may request `url`, and the exact https URL it would request."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
        hostname = parts.hostname
    except ValueError:  # an out-of-range port or a malformed netloc
        return _refused(Refusal.MALFORMED)
    scheme = parts.scheme.lower()
    if scheme not in ("http", "https"):
        return _refused(Refusal.SCHEME)
    if parts.username is not None or parts.password is not None:
        return _refused(Refusal.USERINFO)
    if not hostname:
        return _refused(Refusal.NO_HOST)
    host = hostname.lower().rstrip(".")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return _refused(Refusal.IP_LITERAL)
    # An http URL's port 80 is its default and goes with the upgrade; anything else
    # explicit is a port this fetcher does not connect to.
    if port is not None and port != 443 and not (scheme == "http" and port == 80):
        return _refused(Refusal.PORT)
    if host not in allowed_hosts:
        return _refused(Refusal.HOST_NOT_ALLOWED)
    https = urlunsplit(("https", host, parts.path or "/", parts.query, ""))
    return UrlVerdict(url=https, host=host, upgraded=scheme == "http")


def fetchable_pdf_links(record: PaperCandidate, allowed_hosts: frozenset[str]) -> tuple[str, ...]:
    """The record's PDF links that D115's guard would request, best first (D137).

    Landing pages are left off: the spike never found a PDF behind one, and each
    try would spend a host's pacing slot. The links are kept as the source gave
    them; the fetcher re-checks every hop of every one. One function, because the
    add endpoint refuses a record without such a link and search chooses which
    record of a work to offer for adding by the same test (D147): if the two
    disagreed, search could offer a record that add then refused.
    """
    return tuple(
        link.url
        for link in record.oa_links
        if link.kind is LinkKind.PDF and check_fetch_url(link.url, allowed_hosts).allowed
    )
