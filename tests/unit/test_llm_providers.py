"""Provider adapters, hermetic (D23, D26, D29, D58).

`respx` intercepts every call, so the default test run makes zero network calls
(NFR-T3). What is worth testing here is not "does httpx work" but the three
things that would fail silently: a rate limit read as a bug, usage counted as
zero, and — the one M0 actually lost time to — a CPU-placed model reported as a
GPU one.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest
import respx

from scholiast.application.ports.llm import GenerationUnavailable
from scholiast.config.settings import Settings
from scholiast.infrastructure.llm.breaker import BreakerState
from scholiast.infrastructure.llm.chain import build_provider_chain
from scholiast.infrastructure.llm.groq import GroqProvider
from scholiast.infrastructure.llm.ollama import OllamaProvider

OLLAMA = "http://ollama.test:11434"
GROQ = "https://groq.test/openai/v1"


def ndjson(*frames: dict[str, object]) -> str:
    return "\n".join(json.dumps(f) for f in frames)


def sse(*frames: str) -> str:
    return "\n".join(f"data: {f}" for f in frames) + "\ndata: [DONE]\n"


async def drain(provider: object, **kwargs: str) -> tuple[str, object]:
    text = ""
    usage = None
    async for chunk in provider.stream(**kwargs):  # type: ignore[attr-defined]
        text += chunk.text
        if chunk.usage is not None:
            usage = chunk.usage
    return text, usage


class TestOllamaStreams:
    @respx.mock
    async def test_ndjson_frames_become_tokens(self) -> None:
        respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(
                200,
                text=ndjson(
                    {"message": {"content": "Attention "}},
                    {"message": {"content": "scales."}},
                    {"done": True, "prompt_eval_count": 1200, "eval_count": 42},
                ),
            )
        )
        text, usage = await drain(OllamaProvider(OLLAMA), system="rules", user="question")
        assert text == "Attention scales."
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (1200, 42)

    @respx.mock
    async def test_num_gpu_is_never_sent(self) -> None:
        # D23: one `num_gpu` request pins the model to that placement for every
        # later request inside `keep_alive`. The safe number of places that set
        # it is zero, and this is what keeps it there.
        route = respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(200, text=ndjson({"done": True}))
        )
        await drain(OllamaProvider(OLLAMA), system="rules", user="question")
        body = json.loads(route.calls[0].request.content)
        assert "num_gpu" not in json.dumps(body)
        assert body["stream"] is True

    @respx.mock
    async def test_ollama_being_down_is_unavailable_not_a_crash(self) -> None:
        # Means "try the next rung", which is the whole point of the ladder.
        respx.post(f"{OLLAMA}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(GenerationUnavailable):
            await drain(OllamaProvider(OLLAMA), system="s", user="u")

    @respx.mock
    async def test_an_error_frame_mid_stream_is_unavailable(self) -> None:
        respx.post(f"{OLLAMA}/api/chat").mock(
            return_value=httpx.Response(200, text=ndjson({"error": "model not found"}))
        )
        with pytest.raises(GenerationUnavailable):
            await drain(OllamaProvider(OLLAMA), system="s", user="u")


class TestOllamaPlacementIsReadNotAssumed:
    """D23's rule, and the reason it exists: a silent CPU fallback at ~7x slower
    looks exactly like a working GPU path, which is the undisclosed degradation
    NFR-R4 forbids."""

    @respx.mock
    @pytest.mark.parametrize(
        ("size", "size_vram", "expected"),
        [
            (2_000_000_000, 2_000_000_000, "gpu"),
            (2_000_000_000, 0, "cpu"),
            (2_000_000_000, 1_000_000_000, "partial"),
        ],
    )
    async def test_placement_reflects_what_ollama_reports(
        self, size: int, size_vram: int, expected: str
    ) -> None:
        respx.get(f"{OLLAMA}/api/ps").mock(
            return_value=httpx.Response(
                200,
                json={
                    "models": [{"name": "llama3.2:latest", "size": size, "size_vram": size_vram}]
                },
            )
        )
        assert await OllamaProvider(OLLAMA).placement() == expected

    @respx.mock
    async def test_an_unloaded_model_says_so(self) -> None:
        respx.get(f"{OLLAMA}/api/ps").mock(return_value=httpx.Response(200, json={"models": []}))
        assert await OllamaProvider(OLLAMA).placement() == "unloaded"

    @respx.mock
    async def test_an_unreachable_ollama_is_unknown_not_gpu(self) -> None:
        # The failure that matters: defaulting to "gpu" when we cannot tell would
        # make the system claim a property it never checked.
        respx.get(f"{OLLAMA}/api/ps").mock(side_effect=httpx.ConnectError("refused"))
        assert await OllamaProvider(OLLAMA).placement() == "unknown"


class TestGroqStreams:
    @respx.mock
    async def test_sse_frames_become_tokens_and_usage(self) -> None:
        respx.post(f"{GROQ}/chat/completions").mock(
            return_value=httpx.Response(
                200,
                text=sse(
                    json.dumps({"choices": [{"delta": {"content": "Hybrid "}}]}),
                    json.dumps({"choices": [{"delta": {"content": "retrieval."}}]}),
                    json.dumps(
                        {"choices": [], "usage": {"prompt_tokens": 900, "completion_tokens": 30}}
                    ),
                ),
            )
        )
        text, usage = await drain(
            GroqProvider("key", base_url=GROQ), system="rules", user="question"
        )
        assert text == "Hybrid retrieval."
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (900, 30)

    @respx.mock
    async def test_a_rate_limit_is_unavailable_and_records_retry_after(self) -> None:
        # M0 measured 12,000 TPM ≈ 2 RAG requests/minute, so this is the ordinary
        # case rather than an exotic one — and the Retry-After value is what M3's
        # limiter gets tuned against.
        respx.post(f"{GROQ}/chat/completions").mock(
            return_value=httpx.Response(429, headers={"retry-after": "17"}, text="slow down")
        )
        with pytest.raises(GenerationUnavailable) as caught:
            await drain(GroqProvider("key", base_url=GROQ), system="s", user="u")
        assert "retry-after=17" in caught.value.detail
        # D103: and as a NUMBER, not only as prose. The string is for logs; the
        # float is what the breaker opens its window from. Before this the value
        # was parsed and then thrown away into a message.
        assert caught.value.retry_after == pytest.approx(17.0)

    @respx.mock
    async def test_the_api_key_is_sent_as_a_bearer_token(self) -> None:
        route = respx.post(f"{GROQ}/chat/completions").mock(
            return_value=httpx.Response(200, text=sse(json.dumps({"choices": []})))
        )
        await drain(GroqProvider("secret-key", base_url=GROQ), system="s", user="u")
        assert route.calls[0].request.headers["authorization"] == "Bearer secret-key"

    @respx.mock
    async def test_usage_is_requested_explicitly(self) -> None:
        # Without `include_usage` the stream ends with no counts and NFR-C2's
        # requests-per-answer accounting silently records zeros.
        route = respx.post(f"{GROQ}/chat/completions").mock(
            return_value=httpx.Response(200, text=sse(json.dumps({"choices": []})))
        )
        await drain(GroqProvider("key", base_url=GROQ), system="s", user="u")
        body = json.loads(route.calls[0].request.content)
        assert body["stream_options"] == {"include_usage": True}
        assert body["temperature"] == 0


class TestTheChainIsOrderedAndHonest:
    def test_ollama_comes_first(self) -> None:
        # D26, and it inverts §5.11's obvious reading: Groq's free tier is ~2 RAG
        # requests per minute, so hosted-first makes the dev default unusable
        # after two questions.
        chain = build_provider_chain(Settings(groq_api_key="key"))
        assert [p.name for p in chain] == ["ollama", "groq"]

    def test_a_missing_credential_omits_the_provider(self) -> None:
        # Absent, not broken. A Groq client around an empty key turns every
        # request into a 401 that the chain reads as a transient outage — a
        # misconfiguration wearing a failure's clothes.
        chain = build_provider_chain(Settings(groq_api_key=None))
        assert [p.name for p in chain] == ["ollama"]

    def test_the_pinned_model_is_reported(self) -> None:
        # Doc 7 §7.5: with `temperature` unavailable on some providers,
        # reproducibility rests on a pinned model id plus a pinned prompt version.
        chain = build_provider_chain(Settings(groq_api_key="key"))
        assert all(p.model for p in chain)

    def test_every_rung_sits_behind_a_breaker(self) -> None:
        # D104, including the local one. An Ollama that is not running refuses
        # fast; an Ollama that is HUNG does not, and that is the case that costs
        # a timeout on every question.
        chain = build_provider_chain(Settings(groq_api_key="key"))
        assert [type(p).__name__ for p in chain] == ["CircuitBreaker", "CircuitBreaker"]

    def test_the_breakers_are_independent(self) -> None:
        # One breaker per provider instance, not one per chain: a rate-limited
        # Groq must not stop the chain trying a healthy Ollama.
        chain = build_provider_chain(Settings(groq_api_key="key"))
        first, second = chain
        assert first is not second
        assert first.state is BreakerState.CLOSED  # type: ignore[attr-defined]

    @respx.mock
    async def test_the_threshold_comes_from_settings_not_a_hardcoded_default(self) -> None:
        # Asserted through behaviour rather than by reading `_threshold`: a
        # breaker built with the default 3 would still be CLOSED after one
        # failure, so one failure is what distinguishes wired from ignored.
        respx.post(f"{OLLAMA}/api/chat").mock(return_value=httpx.Response(503, text="down"))
        chain = build_provider_chain(
            Settings(
                ollama_url=OLLAMA,
                groq_api_key=None,
                breaker_failure_threshold=1,
                breaker_cooldown_seconds=99.0,
            )
        )
        breaker = chain[0]

        with pytest.raises(GenerationUnavailable):
            await drain(breaker, system="s", user="u")

        assert breaker.state is BreakerState.OPEN  # type: ignore[attr-defined]


class TestRetryAfterIsParsedNotGuessed:
    """D103's header, in every form a server may legally send it.

    An unparseable header must become None, not zero: None means "nothing was
    said, use the cooldown" and zero means "come back now". Reading a malformed
    header as zero would silently defeat the breaker on exactly the provider that
    sent it.
    """

    async def _retry_after_for(self, headers: dict[str, str]) -> float | None:
        respx.post(f"{GROQ}/chat/completions").mock(
            return_value=httpx.Response(429, headers=headers, text="slow down")
        )
        with pytest.raises(GenerationUnavailable) as caught:
            await drain(GroqProvider("key", base_url=GROQ), system="s", user="u")
        return caught.value.retry_after

    @respx.mock
    async def test_a_fractional_value_survives(self) -> None:
        # Groq sends these. Truncating to int would round a 7.66 s window down and
        # the next request would 429 again.
        assert await self._retry_after_for({"retry-after": "7.66"}) == pytest.approx(7.66)

    @respx.mock
    async def test_surrounding_whitespace_is_tolerated(self) -> None:
        assert await self._retry_after_for({"retry-after": " 12 "}) == pytest.approx(12.0)

    @respx.mock
    async def test_an_http_date_is_converted_to_seconds(self) -> None:
        # RFC 9110 §10.2.3 allows either form. Groq sends a number, but a proxy
        # that rewrites the header would otherwise have it silently ignored — the
        # breaker guessing while the server had stated the answer.
        when = datetime.now(UTC) + timedelta(seconds=45)
        seconds = await self._retry_after_for({"retry-after": format_datetime(when, usegmt=True)})
        assert seconds is not None
        assert 40 <= seconds <= 50

    @respx.mock
    async def test_a_date_in_the_past_is_zero_not_negative(self) -> None:
        # A negative window would put `_opened_until` behind the clock and make
        # the breaker read as HALF-OPEN forever.
        when = datetime.now(UTC) - timedelta(seconds=60)
        assert await self._retry_after_for(
            {"retry-after": format_datetime(when, usegmt=True)}
        ) == pytest.approx(0.0)

    @respx.mock
    async def test_a_negative_number_is_clamped(self) -> None:
        assert await self._retry_after_for({"retry-after": "-5"}) == pytest.approx(0.0)

    @respx.mock
    async def test_an_unparseable_header_is_none(self) -> None:
        assert await self._retry_after_for({"retry-after": "soon-ish"}) is None

    @respx.mock
    async def test_an_absent_header_is_none(self) -> None:
        assert await self._retry_after_for({}) is None

    @respx.mock
    async def test_a_500_carries_no_window(self) -> None:
        # Nothing was stated, so the breaker must use its cooldown rather than
        # inherit a number from an unrelated response.
        respx.post(f"{GROQ}/chat/completions").mock(return_value=httpx.Response(500, text="boom"))
        with pytest.raises(GenerationUnavailable) as caught:
            await drain(GroqProvider("key", base_url=GROQ), system="s", user="u")
        assert caught.value.retry_after is None
