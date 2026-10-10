"""The provider circuit breaker (§5.11, D103, D104).

The breaker's purpose is **latency on the degradation ladder**, so the assertions
that matter are about calls *not made* and windows *not guessed*. A test that
only checked "an open breaker raises" would pass against an implementation that
dialled the dead provider first and raised afterwards — which is the entire cost
the breaker exists to remove.

The clock is injected. Sleeping for a 30 s cooldown would make the suite unusable
and would test `asyncio.sleep`; advancing a fake clock tests the window.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from scholiast.application.ports.llm import (
    GenerationChunk,
    GenerationUnavailable,
    LLMProvider,
)
from scholiast.infrastructure.circuit import Admitted, Circuit, Refused
from scholiast.infrastructure.llm.breaker import BreakerState, CircuitBreaker


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeProvider:
    """Yields, or fails, on command — and counts how often it was actually called.

    The count is the point. "Did the breaker refuse" and "did the breaker refuse
    *without dialling*" are different claims, and only the second one is worth
    anything.
    """

    name = "fake"
    model = "fake-model/1.0"

    def __init__(
        self,
        *,
        fails: bool = False,
        retry_after: float | None = None,
        chunks: int = 1,
    ) -> None:
        self.fails = fails
        self.retry_after = retry_after
        self.chunks = chunks
        self.calls = 0

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        self.calls += 1
        if self.fails:
            raise GenerationUnavailable("fake", "down", retry_after=self.retry_after)
        for index in range(self.chunks):
            yield GenerationChunk(text=f"t{index}")


async def drain(provider: object, **kwargs: str) -> str:
    text = ""
    async for chunk in provider.stream(system="s", user="u", **kwargs):  # type: ignore[attr-defined]
        text += chunk.text
    return text


def build(
    provider: FakeProvider, *, threshold: int = 3, cooldown: float = 30.0
) -> tuple[CircuitBreaker, FakeClock]:
    clock = FakeClock()
    breaker = CircuitBreaker(
        provider, failure_threshold=threshold, cooldown_seconds=cooldown, clock=clock
    )
    return breaker, clock


async def fail_times(breaker: CircuitBreaker, count: int) -> None:
    for _ in range(count):
        with pytest.raises(GenerationUnavailable):
            await drain(breaker)


class TestABreakerIsInvisibleWhileTheProviderWorks:
    async def test_chunks_pass_straight_through(self) -> None:
        provider = FakeProvider(chunks=3)
        breaker, _ = build(provider)

        assert await drain(breaker) == "t0t1t2"
        assert breaker.state is BreakerState.CLOSED

    async def test_it_satisfies_the_provider_port(self) -> None:
        # The chain treats a breaker and a bare provider identically; if the
        # wrapper did not satisfy the port, that would only surface at runtime.
        breaker, _ = build(FakeProvider())
        assert isinstance(breaker, LLMProvider)

    async def test_identity_is_the_wrapped_providers(self) -> None:
        # §6.10 persists provider and model per message so a quality difference
        # can be attributed. Reporting "circuitbreaker" would make every answer
        # attributable to the wrapper instead of the model that wrote it.
        breaker, _ = build(FakeProvider())
        assert breaker.name == "fake"
        assert breaker.model == "fake-model/1.0"


class TestTheThresholdMeasuresAnOutageNotALifetime:
    async def test_it_opens_after_the_configured_failures(self) -> None:
        provider = FakeProvider(fails=True)
        breaker, _ = build(provider, threshold=3)

        await fail_times(breaker, 3)

        assert breaker.state is BreakerState.OPEN
        assert provider.calls == 3

    async def test_it_stays_closed_below_the_threshold(self) -> None:
        provider = FakeProvider(fails=True)
        breaker, _ = build(provider, threshold=3)

        await fail_times(breaker, 2)

        assert breaker.state is BreakerState.CLOSED

    async def test_a_success_resets_the_count(self) -> None:
        # Consecutive, not cumulative. A provider that fails once an hour would
        # otherwise trip eventually and be refused for reasons long past.
        provider = FakeProvider(fails=True)
        breaker, _ = build(provider, threshold=3)
        await fail_times(breaker, 2)

        provider.fails = False
        await drain(breaker)
        assert breaker.failures == 0

        provider.fails = True
        await fail_times(breaker, 2)

        assert breaker.state is BreakerState.CLOSED


class TestAnOpenBreakerCostsNothing:
    """The latency claim, which is the whole reason the class exists (D104)."""

    async def test_it_refuses_without_dialling_the_provider(self) -> None:
        # Not "does it raise" — "does it raise instead of spending the 60 s
        # timeout". The call count is the only thing that distinguishes them.
        provider = FakeProvider(fails=True)
        breaker, _ = build(provider, threshold=1)
        await fail_times(breaker, 1)
        calls_when_opened = provider.calls

        await fail_times(breaker, 5)

        assert provider.calls == calls_when_opened, (
            "the breaker called a provider it had given up on"
        )

    async def test_the_refusal_still_looks_like_every_other_failure(self) -> None:
        # The use case discloses `provider_failover` on `GenerationUnavailable`.
        # A different exception type would make a fast failover silently stop
        # being disclosed — the breaker changing what the user is told, which is
        # precisely what it must not do.
        provider = FakeProvider(fails=True)
        breaker, _ = build(provider, threshold=1)
        await fail_times(breaker, 1)

        with pytest.raises(GenerationUnavailable) as caught:
            await drain(breaker)

        assert caught.value.provider == "fake"

    async def test_the_refusal_carries_the_remaining_window(self) -> None:
        provider = FakeProvider(fails=True)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(10.0)

        with pytest.raises(GenerationUnavailable) as caught:
            await drain(breaker)

        assert caught.value.retry_after == pytest.approx(20.0)


class TestTheServerSetsTheWindowWhenTheServerSaidSo:
    """D103: the authority on a limit is the thing enforcing it."""

    async def test_a_retry_after_opens_the_breaker_immediately(self) -> None:
        # Before the threshold, and deliberately. D103's stated saving is that
        # "every subsequent request inside the window is refused locally at no
        # cost"; waiting for three failures would spend two more calls inside a
        # window the server had already declared.
        provider = FakeProvider(fails=True, retry_after=17.0)
        breaker, _ = build(provider, threshold=3)

        await fail_times(breaker, 1)

        assert breaker.state is BreakerState.OPEN
        assert breaker.failures == 1

    async def test_the_window_is_exactly_what_was_stated(self) -> None:
        provider = FakeProvider(fails=True, retry_after=17.0)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)

        clock.advance(16.9)
        assert breaker.state is BreakerState.OPEN, "reopened before the stated window elapsed"

        clock.advance(0.2)
        assert breaker.state is BreakerState.HALF_OPEN

    async def test_a_silent_failure_falls_back_to_the_cooldown(self) -> None:
        # None means "nothing was said", not "come back now". Treating it as zero
        # would make the breaker useless for exactly the case it was built for:
        # a timeout, which carries no header at all.
        provider = FakeProvider(fails=True, retry_after=None)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)

        clock.advance(29.0)
        assert breaker.state is BreakerState.OPEN

        clock.advance(2.0)
        assert breaker.state is BreakerState.HALF_OPEN

    async def test_a_stated_window_overrides_a_longer_cooldown(self) -> None:
        # The direction that matters: a provider saying "2 seconds" must not be
        # locked out for the configured 30, or a brief limit becomes an outage.
        provider = FakeProvider(fails=True, retry_after=2.0)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)

        clock.advance(2.5)

        assert breaker.state is BreakerState.HALF_OPEN


class TestTheProbeBoundsTheBreakersOwnRisk:
    """With one provider configured, an open breaker IS the outage (D104)."""

    async def test_a_successful_probe_closes_the_breaker(self) -> None:
        provider = FakeProvider(fails=True)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(31.0)

        provider.fails = False
        assert await drain(breaker) == "t0"

        assert breaker.state is BreakerState.CLOSED
        assert breaker.failures == 0

    async def test_a_failed_probe_reopens_at_once(self) -> None:
        """Re-opening on the strength of being a probe, and nothing else.

        A mutation run caught the first version of this test: the failing probe
        also carried a `Retry-After`, so the breaker re-opened through the header
        branch and the test passed with the probe logic deleted. The header is
        therefore removed before the probe — threshold is 3 and only two failures
        have happened, so `was_probing` is now the only thing that can re-open it.
        """
        provider = FakeProvider(fails=True, retry_after=5.0)
        breaker, clock = build(provider, threshold=3, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(6.0)
        assert breaker.state is BreakerState.HALF_OPEN

        provider.retry_after = None
        await fail_times(breaker, 1)

        assert breaker.failures == 2, "below the threshold, so only the probe can explain this"
        assert breaker.state is BreakerState.OPEN

    async def test_only_one_request_probes_at_a_time(self) -> None:
        # D104 says "a single HALF-OPEN probe". Without the in-flight flag every
        # concurrent request becomes a probe and every one pays the timeout —
        # the breaker would be open and still cost what being closed cost.
        provider = FakeProvider(fails=True)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(31.0)

        provider.fails = False
        probe = breaker.stream(system="s", user="u")
        assert (await probe.__anext__()).text == "t0"  # the probe is now in flight

        with pytest.raises(GenerationUnavailable, match="probe in flight"):
            await drain(breaker)

        await probe.aclose()

    async def test_the_probe_reaches_the_provider(self) -> None:
        # The non-vacuity check for the class above: if HALF-OPEN refused
        # everything, every test here would pass against a two-state breaker
        # that never recovers.
        provider = FakeProvider(fails=True)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)
        calls_when_opened = provider.calls
        clock.advance(31.0)

        provider.fails = False
        await drain(breaker)

        assert provider.calls == calls_when_opened + 1


class TestAnAbandonedProbeDoesNotWedgeTheBreaker:
    """A probe that ends with no verdict must not hold the probe slot forever (D122).

    Found extracting the breaker's core for discovery, where the deadline cancels
    slow calls routinely. Before the fix, a HALF-OPEN probe whose caller stopped
    reading, or that was cancelled, never cleared the in-flight flag, so every
    later call was refused as "probe in flight" until the process restarted: one
    abandoned stream turned a recoverable provider into a permanent outage.
    """

    async def test_a_probe_its_caller_stops_reading_lets_the_next_call_probe(self) -> None:
        provider = FakeProvider(fails=True)
        breaker, clock = build(provider, threshold=1, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(31.0)

        provider.fails = False
        probe = breaker.stream(system="s", user="u")
        await probe.__anext__()  # the probe is in flight...
        await probe.aclose()  # ...and its caller walks away

        assert await drain(breaker) == "t0", (
            "an abandoned probe left the breaker refusing every call"
        )

    async def test_a_cancelled_probe_lets_the_next_call_probe(self) -> None:
        class Moody:
            name = "fake"
            model = "m"

            def __init__(self) -> None:
                self.mode = "fail"

            async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
                if self.mode == "fail":
                    raise GenerationUnavailable("fake", "down", retry_after=5.0)
                if self.mode == "hang":
                    await asyncio.Event().wait()
                yield GenerationChunk(text="back")

        provider = Moody()
        breaker, clock = build(provider)  # type: ignore[arg-type]
        await fail_times(breaker, 1)
        clock.advance(6.0)

        provider.mode = "hang"
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(drain(breaker), timeout=0.05)

        provider.mode = "ok"
        assert await drain(breaker) == "back", (
            "a cancelled probe left the breaker refusing every call"
        )

    async def test_an_abandoned_probe_is_not_counted_as_a_failure(self) -> None:
        # M3's rule stands: only `GenerationUnavailable` is a provider failure. A
        # caller walking away says nothing about the provider's health.
        provider = FakeProvider(fails=True, retry_after=1.0)
        breaker, clock = build(provider, threshold=3, cooldown=30.0)
        await fail_times(breaker, 1)
        clock.advance(2.0)

        provider.fails = False
        probe = breaker.stream(system="s", user="u")
        await probe.__anext__()
        await probe.aclose()

        assert breaker.failures == 1
        assert breaker.state is BreakerState.HALF_OPEN


class TestAMidStreamFailureIsStillRecorded:
    async def test_a_provider_that_dies_after_yielding_counts_as_a_failure(self) -> None:
        # The truncation case (§5.11's `generation_truncated`). The breaker must
        # not treat "yielded something" as success, or a provider failing halfway
        # through every answer would keep its breaker closed forever.
        class DiesMidway:
            name = "fake"
            model = "m"

            async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
                yield GenerationChunk(text="half an ans")
                raise GenerationUnavailable("fake", "died mid-stream")

        breaker, _ = build(DiesMidway())  # type: ignore[arg-type]

        with pytest.raises(GenerationUnavailable):
            await drain(breaker)

        assert breaker.failures == 1


class TestOnlyTheProbeCanReleaseTheProbeSlot:
    """The shared core's ticket (D122): a late call cannot free a probe it does not hold."""

    def test_a_call_admitted_before_the_breaker_opened_cannot_free_the_probe(self) -> None:
        clock = FakeClock()
        circuit = Circuit(failure_threshold=1, cooldown_seconds=30.0, clock=clock)
        early = circuit.admit()  # admitted while CLOSED, still in flight
        assert isinstance(early, Admitted)

        circuit.failed(retry_after=None)  # another call fails: the breaker opens
        clock.advance(31.0)
        probe = circuit.admit()
        assert isinstance(probe, Admitted) and probe.probe

        circuit.abandoned(early)  # the early call finishes without a verdict

        refused = circuit.admit()
        assert isinstance(refused, Refused), "two probes in flight at once"
        assert refused.reason == "circuit half-open, probe in flight"

        circuit.abandoned(probe)
        assert isinstance(circuit.admit(), Admitted), "the real probe's release was lost"
