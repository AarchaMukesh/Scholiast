"""Local generation via Ollama (D22, D23, D26).

The **throughput path**, not the emergency one. D26 inverted D14's framing on
measured evidence: Groq's free tier is 12,000 tokens/minute, which is roughly two
RAG requests per minute, while this 3B runs at 125 tok/s fully GPU-resident with
no quota at all. Hosted capacity becomes a deliberate spend; this carries the
load.

**Placement is read, never assumed** (D23). M0 lost an afternoon to this: one
request with `num_gpu: 0` pinned the model to CPU, and every later request inside
the 5-minute `keep_alive` window reused that placement — a 7x slowdown that
looked exactly like a working GPU path. Two rules follow, and both are
implemented here rather than left to the operator:

  1. This adapter **never sends `num_gpu`**. Mixing the setting against one
     loaded model is what poisons it, so the safe number of places that set it
     is zero.
  2. `placement()` asks `/api/ps` what Ollama actually did. An undisclosed
     degradation is what NFR-R4 forbids, and "we assumed GPU" is how one hides.
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

PROVIDER_NAME = "ollama"

#: Ollama reports sizes in bytes; `size_vram` is how much of the model is on the
#: GPU. The ratio is the placement, and it is the number D23 wants surfaced.
_PS_ENDPOINT = "/api/ps"
_CHAT_ENDPOINT = "/api/chat"


class OllamaProvider:
    """The local `LLMProvider`."""

    def __init__(
        self,
        base_url: str,
        *,
        model: str = "llama3.2",
        timeout: float = 120.0,
    ) -> None:
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
        """Stream `/api/chat`, newline-delimited JSON.

        No `options` block at all: `temperature` would be reasonable to set here,
        but `num_gpu` must never be, and the cheapest way to guarantee the second
        is not to build the dict. Determinism for eval comes from a pinned model
        id and a pinned prompt version instead (doc 7 §7.5).
        """
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": True,
        }

        try:
            async with (
                httpx.AsyncClient(timeout=self._timeout) as client,
                client.stream(
                    "POST", f"{self._base_url}{_CHAT_ENDPOINT}", json=payload
                ) as response,
            ):
                if response.status_code != 200:
                    await response.aread()
                    raise GenerationUnavailable(
                        PROVIDER_NAME, f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                async for line in response.aiter_lines():
                    chunk = _parse_line(line)
                    if chunk is not None:
                        yield chunk
        except httpx.HTTPError as exc:
            # Ollama not running, or the model not pulled. Both mean "try the
            # next rung", which is what GenerationUnavailable signals.
            raise GenerationUnavailable(
                PROVIDER_NAME, f"{type(exc).__name__}: {exc}"[:200]
            ) from exc

    async def placement(self) -> str:
        """Where the model actually is: `gpu`, `cpu`, `partial`, or `unloaded`.

        Reported rather than inferred (D23). Called by the readiness probe and by
        the eval harness, not on the request path — this is a claim the system
        makes about itself, and it should be checkable on demand rather than
        guessed from throughput after the fact.
        """
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self._base_url}{_PS_ENDPOINT}")
                response.raise_for_status()
                loaded = response.json().get("models", [])
        except (httpx.HTTPError, ValueError):
            return "unknown"

        for entry in loaded:
            if not str(entry.get("name", "")).startswith(self._model):
                continue
            total = int(entry.get("size", 0) or 0)
            on_gpu = int(entry.get("size_vram", 0) or 0)
            if total <= 0:
                return "unknown"
            if on_gpu == 0:
                return "cpu"
            # Exact equality rather than a ratio threshold: M0's failure mode was
            # a *fully* CPU-placed model masquerading as a GPU one, and a partial
            # offload is a third state that deserves its own name rather than
            # being rounded into either neighbour.
            return "gpu" if on_gpu >= total else "partial"
        return "unloaded"


def _parse_line(line: str) -> GenerationChunk | None:
    """One NDJSON line to a chunk, or None for keep-alive blanks."""
    if not line.strip():
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        # A malformed line mid-stream is not worth failing the whole answer
        # over; the tokens already delivered stand and the stream continues.
        return None

    if payload.get("error"):
        raise GenerationUnavailable(PROVIDER_NAME, str(payload["error"])[:200])

    text = str(payload.get("message", {}).get("content", ""))
    done = bool(payload.get("done"))
    usage = None
    if done:
        usage = TokenUsage(
            prompt_tokens=int(payload.get("prompt_eval_count", 0) or 0),
            completion_tokens=int(payload.get("eval_count", 0) or 0),
            reported="eval_count" in payload,
        )
    return GenerationChunk(text=text, done=done, usage=usage)
