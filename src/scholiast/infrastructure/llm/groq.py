"""Hosted generation via Groq (D29, D26).

**Preferred hosted provider on data-terms grounds, not capability grounds**
(D29). Doc 7 called Gemini the strong first candidate; reading both policies
reversed it. Groq's Services Agreement prohibits training on Inputs and Outputs;
Google's *free* tier explicitly uses submitted content to improve its products
and permits human reviewers to read it. This system ingests possibly-unpublished
manuscripts, so the free tier is the one that would be used and the one that
disqualifies itself.

**Secondary, not primary** (D26). Measured in M0: 12,000 tokens/minute, and a
RAG request is ~6k tokens — about two sustained requests per minute. That is
unusable as a development default, so quota is spent deliberately rather than by
accident.

Rate limits arrive as 429 with `Retry-After`. This adapter surfaces them as
`GenerationUnavailable` **carrying that header as a number** (D103), which is
what lets the circuit breaker in `breaker.py` open for exactly the window the
server stated instead of guessing at it.

**No token bucket, and that deviates from doc 8 deliberately** (D103). Groq
meters *tokens* — 12,000 per minute against a ~6k-token RAG request — and a
token count is not known until after the call, so a bucket sized in requests
limits the wrong quantity and one sized in estimated tokens limits the right
quantity with an invented number. The 429 already answers the question exactly.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from scholiast.application.ports.llm import (
    GenerationChunk,
    GenerationUnavailable,
    TokenUsage,
)
from scholiast.infrastructure.retry_after import retry_after_seconds

PROVIDER_NAME = "groq"
DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
_CHAT_ENDPOINT = "/chat/completions"

#: Server-sent events from an OpenAI-compatible endpoint. The terminator is a
#: literal sentinel rather than JSON, so it is checked before parsing.
_SSE_PREFIX = "data: "
_SSE_DONE = "[DONE]"


class GroqProvider:
    """The hosted `LLMProvider`."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def model(self) -> str:
        return self._model

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
            # Groq accepts temperature, so determinism is available here — doc 7
            # §7.5's correction is that it is provider-dependent, not universal,
            # which is exactly why it is set in the adapter and not the use case.
            "temperature": 0,
            # Ask for usage on the final SSE frame; without it the stream ends
            # with no token counts and NFR-C2's requests-per-answer accounting
            # silently records zeros.
            "stream_options": {"include_usage": True},
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST", f"{self._base_url}{_CHAT_ENDPOINT}", json=payload, headers=headers
                ) as response,
            ):
                if response.status_code != 200:
                    await response.aread()
                    raise GenerationUnavailable(
                        PROVIDER_NAME,
                        _describe(response.status_code, response),
                        retry_after=retry_after_seconds(response),
                    )
                async for line in response.aiter_lines():
                    chunk = _parse_sse(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            raise GenerationUnavailable(
                PROVIDER_NAME, f"{type(exc).__name__}: {exc}"[:200]
            ) from exc


def _describe(status: int, response: httpx.Response) -> str:
    """A failure message that says what to do about it.

    Still formats `Retry-After` for a human reader, but it is no longer the only
    place the value goes — `retry_after_seconds` returns it as a number the breaker can
    act on (D103). The string is for logs; the float is for behaviour.
    """
    if status == 429:
        retry_after = response.headers.get("retry-after", "unknown")
        return f"rate limited, retry-after={retry_after}"
    return f"HTTP {status}: {response.text[:200]}"


def _parse_sse(line: str) -> GenerationChunk | None:
    """One SSE frame to a chunk, or None for blanks and the terminator."""
    if not line.startswith(_SSE_PREFIX):
        return None
    body = line[len(_SSE_PREFIX) :].strip()
    if not body or body == _SSE_DONE:
        return None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None

    choices = payload.get("choices") or []
    text = ""
    done = False
    if choices:
        text = str(choices[0].get("delta", {}).get("content") or "")
        done = choices[0].get("finish_reason") is not None

    usage_payload = payload.get("usage")
    usage = None
    if usage_payload:
        usage = TokenUsage(
            prompt_tokens=int(usage_payload.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage_payload.get("completion_tokens", 0) or 0),
        )
        # The usage frame arrives after the last content frame and carries no
        # choices, so it is the real end of the stream.
        done = True
    return GenerationChunk(text=text, done=done, usage=usage)
