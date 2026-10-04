"""D115's rules as pure functions (D129, D130): what the fetcher may ask for, and where.

Each refused case is a way an attacker-supplied link reaches something it should not:
the cloud metadata address, the private network, a non-TLS or non-standard port, a
credential smuggled in the URL, a host nobody approved, or a public-looking IPv6
address that carries a private IPv4 one to the network beyond it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scholiast.domain.fetch_policy import Refusal, check_fetch_url, is_public_address

ALLOWED = frozenset({"arxiv.org", "aclanthology.org", "www.aclweb.org"})
PROTOCOL = Path(__file__).resolve().parents[2] / "spike" / "m4-protocol.json"


def spike_table() -> dict[str, list[str]]:
    """The spike's own address table (protocol http_safety.self_test), inherited whole."""
    return json.loads(PROTOCOL.read_text(encoding="utf-8"))["http_safety"]["self_test"]


class TestAddresses:
    @pytest.mark.parametrize("address", spike_table()["must_refuse"])
    def test_the_spikes_refused_addresses_are_refused(self, address: str) -> None:
        assert not is_public_address(address)

    @pytest.mark.parametrize("address", spike_table()["must_accept"])
    def test_the_spikes_public_addresses_are_accepted(self, address: str) -> None:
        assert is_public_address(address)

    @pytest.mark.parametrize(
        "address",
        [
            "::ffff:10.0.0.1",
            "64:ff9b::a9fe:a9fe",  # NAT64 of 169.254.169.254
            "64:ff9b::7f00:1",  # NAT64 of 127.0.0.1
            "64:ff9b:1::8.8.8.8",  # local-use NAT64: refused whole
            "2002:a9fe:a9fe::1",  # 6to4 of 169.254.169.254
            "2002:c0a8:101::1",  # 6to4 of 192.168.1.1
            "2001:0:4136:e378:8000:63bf:3fff:fdd2",  # Teredo, client 192.0.2.45
        ],
        ids=[
            "mapped private",
            "NAT64 metadata",
            "NAT64 loopback",
            "NAT64 local-use",
            "6to4 metadata",
            "6to4 private",
            "Teredo documentation",
        ],
    )
    def test_ipv4_smuggled_inside_ipv6_is_judged_as_ipv4(self, address: str) -> None:
        assert not is_public_address(address)

    def test_teredo_is_refused_even_when_it_carries_public_addresses(self) -> None:
        # Server 65.54.227.120, client 8.8.8.8 (obfuscated): every embedded address is
        # public, and it is still a tunnel nothing here needs to reach.
        assert not is_public_address("2001:0:4136:e378:8000:63bf:f7f7:f7f7")

    @pytest.mark.parametrize("address", ["64:ff9b::808:808", "2002:808:808::1"])
    def test_public_ipv4_carried_in_ipv6_is_still_public(self, address: str) -> None:
        # The non-vacuity check: refusing every translated address would pass every
        # test above and break real NAT64 networks.
        assert is_public_address(address)

    @pytest.mark.parametrize("text", ["not an address", "", "arxiv.org", "1.2.3"])
    def test_anything_that_is_not_an_address_is_refused(self, text: str) -> None:
        assert not is_public_address(text)


class TestUrls:
    @pytest.mark.parametrize(
        ("url", "refusal"),
        [
            ("ftp://arxiv.org/pdf/1", Refusal.SCHEME),
            ("javascript:alert(1)", Refusal.SCHEME),
            ("file:///etc/passwd", Refusal.SCHEME),
            ("data:application/pdf;base64,JVBERi0=", Refusal.SCHEME),
            ("https://arxiv.org:8443/pdf/1", Refusal.PORT),
            ("http://arxiv.org:8080/pdf/1", Refusal.PORT),
            ("https://arxiv.org:0/pdf/1", Refusal.PORT),
            ("https://arxiv.org:99999/pdf/1", Refusal.MALFORMED),
            ("https://user:pw@arxiv.org/pdf/1", Refusal.USERINFO),
            ("https://user@arxiv.org/pdf/1", Refusal.USERINFO),
            ("https:///pdf/1", Refusal.NO_HOST),
            ("https://169.254.169.254/latest/meta-data/", Refusal.IP_LITERAL),
            ("https://[::1]/pdf", Refusal.IP_LITERAL),
            ("https://151.101.1.42/pdf/1", Refusal.IP_LITERAL),
            ("https://2130706433/pdf", Refusal.HOST_NOT_ALLOWED),
            ("https://evil.example/pdf", Refusal.HOST_NOT_ALLOWED),
            ("https://arxiv.org.evil.example/pdf", Refusal.HOST_NOT_ALLOWED),
            ("https://evilarxiv.org/pdf", Refusal.HOST_NOT_ALLOWED),
            ("https://export.arxiv.org/pdf/1", Refusal.HOST_NOT_ALLOWED),
        ],
    )
    def test_a_link_that_breaks_a_rule_is_refused_with_the_rule(
        self, url: str, refusal: Refusal
    ) -> None:
        verdict = check_fetch_url(url, ALLOWED)
        assert (verdict.allowed, verdict.refusal, verdict.url) == (False, refusal, None)

    @pytest.mark.parametrize(
        ("url", "requested", "upgraded"),
        [
            ("https://arxiv.org/pdf/2004.04906v3", "https://arxiv.org/pdf/2004.04906v3", False),
            ("HTTPS://ArXiv.ORG./pdf/1", "https://arxiv.org/pdf/1", False),
            ("https://arxiv.org:443/pdf/1?x=1#frag", "https://arxiv.org/pdf/1?x=1", False),
            ("http://arxiv.org/pdf/1", "https://arxiv.org/pdf/1", True),
            ("http://arxiv.org:80/pdf/1", "https://arxiv.org/pdf/1", True),
            (
                "https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf",
                "https://www.aclweb.org/anthology/2020.emnlp-main.550.pdf",
                False,
            ),
        ],
    )
    def test_an_allowed_link_is_requested_over_https_on_its_own_host(
        self, url: str, requested: str, upgraded: bool
    ) -> None:
        # http is upgraded, never sent: refusing would lose working papers, and the
        # alternative to upgrading is a request in plaintext.
        verdict = check_fetch_url(url, ALLOWED)
        assert (verdict.allowed, verdict.url, verdict.upgraded) == (True, requested, upgraded)
