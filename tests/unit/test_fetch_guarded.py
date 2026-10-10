"""The guarded PDF fetcher, offline (D115, D129).

A mock transport stands in for the network and a scripted resolver for DNS, so every
test can see exactly where a request was sent and with what. The claims that matter
are about requests NOT sent and bytes NOT read: the private address never dialled,
the second DNS answer never used, the redirect to the metadata service never
followed, the gzip bomb never decompressed, the oversized body never kept.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from scholiast.application.ports.fetch import FetchFailed, PdfFetcher
from scholiast.config.settings import Settings
from scholiast.domain.errors import FailureClass
from scholiast.infrastructure.fetch import build_pdf_fetcher
from scholiast.infrastructure.fetch.guarded import USER_AGENT, GuardedPdfFetcher, HostPacer

ALLOWED = ("arxiv.org", "aclanthology.org", "www.aclweb.org")
PDF = b"%PDF-1.7\n" + b"x" * 1000
ARXIV_IP = "151.101.1.42"
ACL_IP = "185.199.108.153"
ACLWEB_IP = "185.199.109.153"


class Resolver:
    """DNS on a script: a list of answers per host, used in turn, and every call kept."""

    def __init__(self, answers: dict[str, list[list[str]]]) -> None:
        self.answers = answers
        self.calls: list[str] = []

    async def __call__(self, host: str) -> list[str]:
        self.calls.append(host)
        script = self.answers[host]
        return script.pop(0) if len(script) > 1 else script[0]


def standard_dns() -> Resolver:
    return Resolver(
        {"arxiv.org": [[ARXIV_IP]], "aclanthology.org": [[ACL_IP]], "www.aclweb.org": [[ACLWEB_IP]]}
    )


Handler = Callable[[httpx.Request], httpx.Response]


async def streamed(data: bytes) -> AsyncIterator[bytes]:
    """A body that streams, as a network body does. Bytes given to `httpx.Response`
    directly are read on construction, and a response read already cannot stream."""
    yield data


class Network:
    """The mock transport, recording every request that reached it."""

    def __init__(self, handler: Handler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []

    def transport(self) -> httpx.MockTransport:
        def record(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return self.handler(request)

        return httpx.MockTransport(record)


def fetcher(network: Network, dns: Resolver | None = None, **kwargs: object) -> GuardedPdfFetcher:
    return GuardedPdfFetcher(
        allowed_hosts=ALLOWED,
        pacer=HostPacer(default_gap=0.0),
        resolver=dns or standard_dns(),
        transport=network.transport(),
        **kwargs,  # type: ignore[arg-type]
    )


def pdf(_: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=streamed(PDF))


class TestAFetchIsPinnedToTheCheckedAddress:
    async def test_the_request_goes_to_the_checked_ip_with_the_real_host_named(self) -> None:
        network = Network(pdf)
        result = await fetcher(network).fetch(["https://arxiv.org/pdf/2004.04906v3"])

        [request] = network.requests
        assert result.data == PDF
        assert request.url.host == ARXIV_IP, "connected by name, so DNS could be asked again"
        assert request.headers["host"] == "arxiv.org"
        assert request.extensions["sni_hostname"] == "arxiv.org", (
            "the certificate would be checked against the IP, not the host"
        )

    async def test_it_asks_for_no_encoding_names_itself_and_sends_no_cookies(self) -> None:
        network = Network(pdf)
        await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        [request] = network.requests
        assert request.headers["accept-encoding"] == "identity"
        assert request.headers["user-agent"] == USER_AGENT
        assert "cookie" not in request.headers

    async def test_a_second_dns_answer_never_redirects_the_connection(self) -> None:
        """DNS rebinding: public when checked, private when the client would ask again."""
        dns = Resolver({"arxiv.org": [[ARXIV_IP], ["169.254.169.254"]]})
        network = Network(pdf)

        await fetcher(network, dns).fetch(["https://arxiv.org/pdf/1"])

        assert dns.calls == ["arxiv.org"], "the host was resolved twice for one hop"
        assert [r.url.host for r in network.requests] == [ARXIV_IP]

    async def test_one_private_address_among_public_ones_refuses_the_host(self) -> None:
        # The order of answers is the attacker's; checking only the first is a coin toss.
        dns = Resolver({"arxiv.org": [[ARXIV_IP, "10.0.0.5"]]})
        network = Network(pdf)

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, dns).fetch(["https://arxiv.org/pdf/1"])

        assert network.requests == []
        assert caught.value.attempts[0].outcome == "refused:address"
        assert caught.value.failure_class is FailureClass.SOURCE_UNAVAILABLE

    async def test_an_unresolvable_host_is_a_transient_failure(self) -> None:
        async def broken(host: str) -> list[str]:
            raise OSError("name resolution failed")

        network = Network(pdf)
        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, broken).fetch(["https://arxiv.org/pdf/1"])  # type: ignore[arg-type]

        assert caught.value.failure_class is FailureClass.TRANSIENT
        assert network.requests == []


def redirecting(chain: dict[tuple[str, str], str]) -> Handler:
    """Answer each (host, path) with a 302 to the next URL, and a PDF where the chain ends."""

    def handle(request: httpx.Request) -> httpx.Response:
        key = (request.headers["host"], request.url.path)
        if key in chain:
            return httpx.Response(302, headers={"location": chain[key]})
        return httpx.Response(200, content=streamed(PDF))

    return handle


class TestNothingFromTheEnvironmentStepsAroundTheGuard:
    async def test_the_client_takes_no_proxy_and_follows_no_redirect_by_itself(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Structure, asserted directly: a proxy from the environment would carry the
        request past the pinned address, and a client that followed redirects would
        skip every check on the next hop. The live run showed a dead HTTPS_PROXY
        ignored; offline, a mock transport bypasses proxies whatever the setting."""
        seen: list[dict[str, object]] = []
        real = httpx.AsyncClient

        def spy(**kwargs: object) -> httpx.AsyncClient:
            seen.append(kwargs)
            return real(**kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(httpx, "AsyncClient", spy)
        await fetcher(Network(pdf)).fetch(["https://arxiv.org/pdf/1"])

        assert [(k["trust_env"], k["follow_redirects"]) for k in seen] == [(False, False)]


class TestRedirectsAreCheckedAtEveryHop:
    async def test_acls_redirect_host_hands_over_to_aclanthology(self) -> None:
        network = Network(
            redirecting({("www.aclweb.org", "/anthology/x.pdf"): "https://aclanthology.org/x.pdf"})
        )

        result = await fetcher(network).fetch(["https://www.aclweb.org/anthology/x.pdf"])

        assert [(r.url.host, r.headers["host"]) for r in network.requests] == [
            (ACLWEB_IP, "www.aclweb.org"),
            (ACL_IP, "aclanthology.org"),
        ]
        assert result.final_url == "https://aclanthology.org/x.pdf"
        assert result.attempts[0].redirects == 1

    @pytest.mark.parametrize(
        ("location", "outcome"),
        [
            ("https://169.254.169.254/latest/meta-data/", "refused:ip_literal"),
            ("https://evil.example/x.pdf", "refused:host_not_allowed"),
            ("https://arxiv.org:8443/x.pdf", "refused:port"),
            ("ftp://arxiv.org/x.pdf", "refused:scheme"),
            ("https://user:pw@arxiv.org/x.pdf", "refused:userinfo"),
        ],
        ids=["metadata service", "unlisted host", "other port", "other scheme", "credentials"],
    )
    async def test_a_redirect_that_breaks_a_rule_is_not_followed(
        self, location: str, outcome: str
    ) -> None:
        network = Network(redirecting({("arxiv.org", "/pdf/1"): location}))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert len(network.requests) == 1, "the refused hop was requested"
        assert caught.value.attempts[0].outcome == outcome

    async def test_a_redirect_to_an_allowed_host_with_a_private_address_is_refused(self) -> None:
        dns = Resolver({"arxiv.org": [[ARXIV_IP]], "aclanthology.org": [["192.168.1.10"]]})
        network = Network(redirecting({("arxiv.org", "/pdf/1"): "https://aclanthology.org/x.pdf"}))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, dns).fetch(["https://arxiv.org/pdf/1"])

        assert len(network.requests) == 1
        assert caught.value.attempts[0].outcome == "refused:address"

    async def test_an_http_redirect_is_upgraded_not_sent_in_plaintext(self) -> None:
        network = Network(redirecting({("arxiv.org", "/pdf/1"): "http://arxiv.org/pdf/1v2"}))

        result = await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert [r.url.scheme for r in network.requests] == ["https", "https"]
        assert result.final_url == "https://arxiv.org/pdf/1v2"

    async def test_a_relative_redirect_stays_on_the_host_that_sent_it(self) -> None:
        # Resolved against the hostname URL, not the IP one the request went to.
        network = Network(redirecting({("arxiv.org", "/pdf/1"): "/pdf/1v3"}))

        result = await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert result.final_url == "https://arxiv.org/pdf/1v3"
        assert network.requests[1].headers["host"] == "arxiv.org"

    async def test_five_redirects_are_followed_and_a_sixth_is_not(self) -> None:
        five = {("arxiv.org", f"/r{i}"): f"/r{i + 1}" for i in range(5)}
        network = Network(redirecting(five))
        assert (await fetcher(network).fetch(["https://arxiv.org/r0"])).attempts[0].redirects == 5

        six = {("arxiv.org", f"/r{i}"): f"/r{i + 1}" for i in range(6)}
        network = Network(redirecting(six))
        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/r0"])
        assert caught.value.attempts[0].outcome == "too_many_redirects"
        assert len(network.requests) == 6

    async def test_a_redirect_loop_ends(self) -> None:
        network = Network(redirecting({("arxiv.org", "/a"): "/b", ("arxiv.org", "/b"): "/a"}))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/a"])

        assert caught.value.attempts[0].outcome == "too_many_redirects"


class Body:
    """A streamed body that counts how much of it was ever read."""

    def __init__(self, chunks: list[bytes], delay: float = 0.0) -> None:
        self.chunks = chunks
        self.delay = delay
        self.sent = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self.chunks:
            if self.delay:
                await asyncio.sleep(self.delay)
            self.sent += len(chunk)
            yield chunk


class TestTheBodyIsReadUnderLimits:
    async def test_a_body_past_the_cap_is_abandoned_while_streaming(self) -> None:
        # The declared length lies; the cap is counted on bytes as they arrive.
        body = Body([b"%PDF-1.7\n"] + [b"x" * 1024] * 100)
        network = Network(
            lambda r: httpx.Response(200, headers={"content-length": "100"}, content=body)
        )

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, max_bytes=10_000).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].outcome == "too_large"
        assert caught.value.failure_class is FailureClass.TOO_LARGE
        assert body.sent < 10_000 + 2 * 1024, "the oversized body was read to the end"

    async def test_a_declared_size_past_the_cap_is_refused_unread(self) -> None:
        body = Body([b"%PDF-1.7\n", b"x" * 1024])
        network = Network(
            lambda r: httpx.Response(200, headers={"content-length": "99999999"}, content=body)
        )

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, max_bytes=10_000).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].outcome == "too_large"
        assert body.sent == 0

    async def test_a_compressed_response_is_refused_without_decompressing(self) -> None:
        """A gzip bomb: a few KB that inflate to gigabytes. It is never inflated."""
        import gzip

        bomb = gzip.compress(b"%PDF-" + b"\0" * 5_000_000)
        body = Body([bomb])
        network = Network(
            lambda r: httpx.Response(200, headers={"content-encoding": "gzip"}, content=body)
        )

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].outcome == "encoded"
        assert body.sent == 0

    async def test_html_where_a_pdf_was_promised_is_not_a_pdf_and_is_not_read_on(self) -> None:
        body = Body([b"<!doctype html><html>", b"x" * 1024 * 50])
        network = Network(lambda r: httpx.Response(200, content=body))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].outcome == "not_pdf"
        assert caught.value.failure_class is FailureClass.SOURCE_UNAVAILABLE
        assert body.sent < 1024 * 50

    @pytest.mark.parametrize("body", [b"", b"%PD"], ids=["empty", "shorter than the magic"])
    async def test_a_body_too_short_to_be_a_pdf_is_not_a_pdf(self, body: bytes) -> None:
        network = Network(lambda r: httpx.Response(200, content=streamed(body)))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].outcome == "not_pdf"

    async def test_a_server_that_drips_bytes_is_cut_off_at_the_total_timeout(self) -> None:
        body = Body([b"%PDF-"] + [b"x"] * 1000, delay=0.05)
        network = Network(lambda r: httpx.Response(200, content=body))

        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(FetchFailed) as caught:
            await fetcher(network, total_timeout=0.3).fetch(["https://arxiv.org/pdf/1"])

        assert loop.time() - started < 1.5
        assert caught.value.attempts[0].outcome == "timeout"
        assert caught.value.failure_class is FailureClass.TRANSIENT


class TestFailuresAreClassedByWhatHelpsNext:
    @pytest.mark.parametrize(
        ("status", "failure_class"),
        [
            (403, FailureClass.SOURCE_UNAVAILABLE),
            (404, FailureClass.SOURCE_UNAVAILABLE),
            (429, FailureClass.TRANSIENT),
            (503, FailureClass.TRANSIENT),
        ],
    )
    async def test_a_status_becomes_its_class(
        self, status: int, failure_class: FailureClass
    ) -> None:
        network = Network(lambda r: httpx.Response(status))

        with pytest.raises(FetchFailed) as caught:
            await fetcher(network).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.failure_class is failure_class

    async def test_a_connection_error_is_transient(self) -> None:
        def refuse(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(refuse)).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.failure_class is FailureClass.TRANSIENT

    async def test_links_are_tried_in_order_until_one_gives_a_pdf(self) -> None:
        def handle(request: httpx.Request) -> httpx.Response:
            return (
                httpx.Response(404)
                if request.url.path == "/pdf/1"
                else httpx.Response(200, content=streamed(PDF))
            )

        network = Network(handle)
        result = await fetcher(network).fetch(
            ["https://evil.example/x.pdf", "https://arxiv.org/pdf/1", "https://arxiv.org/pdf/1v2"]
        )

        assert result.link == "https://arxiv.org/pdf/1v2"
        assert [a.outcome for a in result.attempts] == [
            "refused:host_not_allowed",
            "status:404",
            "fetched",
        ]

    async def test_one_transient_failure_among_permanent_ones_makes_the_paper_retryable(
        self,
    ) -> None:
        # A retry may fetch it, so marking it unavailable would give up too early.
        def handle(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503 if request.url.path == "/pdf/2" else 404)

        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(handle)).fetch(
                ["https://arxiv.org/pdf/1", "https://arxiv.org/pdf/2"]
            )

        assert caught.value.failure_class is FailureClass.TRANSIENT

    async def test_no_links_at_all_is_unavailable(self) -> None:
        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(pdf)).fetch([])

        assert caught.value.failure_class is FailureClass.SOURCE_UNAVAILABLE

    def test_it_honours_the_port(self) -> None:
        assert isinstance(fetcher(Network(pdf)), PdfFetcher)


class FakeTime:
    def __init__(self) -> None:
        self.now = 100.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(round(seconds, 6))
        self.now += seconds


class TestHostsArePaced:
    async def test_arxiv_gets_its_robots_txt_gap_from_the_previous_requests_end(self) -> None:
        time = FakeTime()
        pacer = HostPacer(
            default_gap=5.0, gaps={"arxiv.org": 15.0}, clock=time.clock, sleep=time.sleep
        )

        def slow_pdf(request: httpx.Request) -> httpx.Response:
            time.now += 4.0  # the download takes 4 s
            return httpx.Response(200, content=streamed(PDF))

        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=Network(slow_pdf).transport(),
        )
        await guarded.fetch(["https://arxiv.org/pdf/1"])
        await guarded.fetch(["https://arxiv.org/pdf/2"])
        await guarded.fetch(["https://aclanthology.org/x.pdf"])

        # 15 s from the end of the first, not its start; ACL waits for nobody.
        assert time.slept == [15.0]

    async def test_two_fetches_at_once_to_one_host_still_take_turns(self) -> None:
        # Without the per-host lock both would see the slot free and go together.
        time = FakeTime()
        pacer = HostPacer(
            default_gap=5.0, gaps={"arxiv.org": 15.0}, clock=time.clock, sleep=time.sleep
        )

        async def yielding_pdf(request: httpx.Request) -> httpx.Response:
            # Yield to the loop mid-request, so the second fetch really runs while the
            # first is in flight. A mock that never yields finishes the first fetch
            # before the second starts, and the lock is never needed (a mutation run
            # caught exactly that).
            await asyncio.sleep(0)
            return httpx.Response(200, content=streamed(PDF))

        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=httpx.MockTransport(yielding_pdf),
        )

        await asyncio.gather(
            guarded.fetch(["https://arxiv.org/pdf/1"]), guarded.fetch(["https://arxiv.org/pdf/2"])
        )

        assert time.slept == [15.0]

    def test_the_configured_fetcher_paces_arxiv_at_its_terms_and_uses_the_allowlist(self) -> None:
        guarded = build_pdf_fetcher(Settings())
        assert guarded._allowed == frozenset(ALLOWED)
        assert guarded._pacer._gaps == {"arxiv.org": 15.0}

    def test_arxivs_pace_cannot_be_configured_below_its_robots_txt(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Settings(fetch_arxiv_min_gap_seconds=1.0)


class TestAHostThatSaysWhenIsBelieved:
    """D142: a 429 or 503 with Retry-After is a stated time, not a guess to beat."""

    @pytest.mark.parametrize("status", [429, 503])
    async def test_the_stated_wait_is_kept_with_the_attempt(self, status: int) -> None:
        def throttled(_: httpx.Request) -> httpx.Response:
            return httpx.Response(status, headers={"retry-after": "120"})

        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(throttled)).fetch(["https://arxiv.org/pdf/1"])

        [attempt] = caught.value.attempts
        assert attempt.outcome == f"status:{status}"
        assert attempt.retry_after == 120.0
        assert caught.value.failure_class is FailureClass.TRANSIENT
        assert caught.value.retry_after == 120.0

    async def test_the_http_date_form_is_read_too(self) -> None:
        # RFC 9110 allows either form; a proxy that rewrites one must not erase it.
        when = format_datetime(datetime.now(UTC) + timedelta(seconds=300), usegmt=True)

        def throttled(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503, headers={"retry-after": when})

        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(throttled)).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.retry_after == pytest.approx(300.0, abs=5.0)

    async def test_a_refusal_that_states_nothing_states_nothing(self) -> None:
        def throttled(_: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with pytest.raises(FetchFailed) as caught:
            await fetcher(Network(throttled)).fetch(["https://arxiv.org/pdf/1"])

        assert caught.value.attempts[0].retry_after is None
        assert caught.value.retry_after is None, "a wait nobody stated was invented"

    async def test_this_process_asks_the_host_nothing_more_until_then(self) -> None:
        time = FakeTime()
        pacer = HostPacer(
            default_gap=5.0, gaps={"arxiv.org": 15.0}, clock=time.clock, sleep=time.sleep
        )
        network = Network(lambda _: httpx.Response(429, headers={"retry-after": "60"}))
        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=network.transport(),
        )

        with pytest.raises(FetchFailed):
            await guarded.fetch(["https://arxiv.org/pdf/1"])
        with pytest.raises(FetchFailed) as caught:
            await guarded.fetch(["https://arxiv.org/pdf/2"])

        assert len(network.requests) == 1, "arXiv was asked again before the time it stated"
        [attempt] = caught.value.attempts
        assert attempt.outcome == "deferred"
        assert attempt.retry_after == pytest.approx(60.0)
        assert caught.value.failure_class is FailureClass.TRANSIENT

    async def test_a_deferred_host_holds_nothing_up(self) -> None:
        # The serial worker must not wait out a host's stated time: every upload
        # would queue behind it (D142, amended).
        time = FakeTime()
        pacer = HostPacer(
            default_gap=5.0, gaps={"arxiv.org": 15.0}, clock=time.clock, sleep=time.sleep
        )
        pacer.defer("arxiv.org", 600.0)
        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=Network(pdf).transport(),
        )

        with pytest.raises(FetchFailed):
            await guarded.fetch(["https://arxiv.org/pdf/1"])

        assert time.slept == [], "the fetch waited out the host's time instead of returning"

    async def test_a_deferred_host_does_not_stop_another_hosts_link(self) -> None:
        time = FakeTime()
        pacer = HostPacer(default_gap=0.0, clock=time.clock, sleep=time.sleep)
        pacer.defer("arxiv.org", 600.0)
        network = Network(pdf)
        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=network.transport(),
        )

        fetched = await guarded.fetch(
            ["https://arxiv.org/pdf/1", "https://aclanthology.org/2020.emnlp-main.550.pdf"]
        )

        assert fetched.link == "https://aclanthology.org/2020.emnlp-main.550.pdf"
        assert [r.headers["host"] for r in network.requests] == ["aclanthology.org"]
        assert fetched.attempts[0].outcome == "deferred"

    def test_a_later_shorter_wait_never_cuts_an_earlier_longer_one(self) -> None:
        # Two links, two refusals: the host's longest stated time stands.
        time = FakeTime()
        pacer = HostPacer(default_gap=0.0, clock=time.clock, sleep=time.sleep)
        pacer.defer("arxiv.org", 600.0)
        pacer.defer("arxiv.org", 5.0)

        assert pacer.deferred("arxiv.org") == pytest.approx(600.0)

    async def test_once_the_stated_time_passes_the_host_is_asked_again(self) -> None:
        time = FakeTime()
        pacer = HostPacer(default_gap=5.0, clock=time.clock, sleep=time.sleep)
        pacer.defer("arxiv.org", 60.0)
        network = Network(pdf)
        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=network.transport(),
        )

        time.now += 60.0
        await guarded.fetch(["https://arxiv.org/pdf/1"])

        assert len(network.requests) == 1

    async def test_a_stated_wait_shorter_than_the_gap_never_shortens_it(self) -> None:
        time = FakeTime()
        pacer = HostPacer(
            default_gap=5.0, gaps={"arxiv.org": 15.0}, clock=time.clock, sleep=time.sleep
        )
        answers = iter(
            [
                httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(200, content=streamed(PDF)),
            ]
        )
        guarded = GuardedPdfFetcher(
            allowed_hosts=ALLOWED,
            pacer=pacer,
            resolver=standard_dns(),
            transport=Network(lambda _: next(answers)).transport(),
        )

        with pytest.raises(FetchFailed):
            await guarded.fetch(["https://arxiv.org/pdf/1"])
        await guarded.fetch(["https://arxiv.org/pdf/1"])

        assert time.slept == [15.0], "arXiv's robots.txt gap was cut short"
