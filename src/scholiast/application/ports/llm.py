"""Generation port (§5.11, D14, D26, D58).

One interface over "a model that streams text back", because the thing behind it
changes: a local 3B on the GPU, a hosted 70B on someone's free tier, or nothing
at all. §5.11's ladder is only implementable if those are interchangeable at the
call site, and the ladder is what keeps a degraded answer from being served as a
normal one.

Three things are in the contract rather than left to adapters:

  streaming    FR-Q5 streams token-by-token, and NFR-P3 budgets *time to first
               token* rather than total time — so an adapter that buffered the
               whole answer and yielded it at the end would satisfy the type and
               miss the requirement entirely.

  attribution  every adapter reports the provider and model that actually
               answered. §6.10 requires provider and model on every message so a
               quality difference can be attributed to the path that produced it,
               which is impossible if the chain silently substitutes.

  usage        tokens in and out, because with no paid inference the scarce
               resource is quota, not dollars (D26) — and requests-per-answer is
               the number that replaces cost-per-answer (NFR-C2).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class GenerationUnavailable(Exception):
    """This provider cannot serve the request right now.

    Distinct from a bad request: it means *try the next rung of the ladder*
    (§5.11), so the chain catches it and a caller never sees it unless every
    provider raised. Rate limits, timeouts, 5xx and a model that is not pulled
    all arrive here; a malformed prompt does not.

    **`retry_after` is structured rather than formatted into the message (D103).**
    A 429 carries `Retry-After`, which is the server stating when it will accept
    work again — the authority on the limit is the thing enforcing it, so no
    estimate is needed and none can be wrong. The adapter used to parse that
    header straight into a human-readable string, which meant the number was
    read and then thrown away; a field is what lets the circuit breaker open for
    exactly the stated window instead of guessing at it (D104).

    None means the provider said nothing about when to come back, not "come back
    immediately" — the breaker then falls back to its configured cooldown.
    """

    def __init__(self, provider: str, detail: str, *, retry_after: float | None = None) -> None:
        super().__init__(f"{provider}: {detail}")
        self.provider = provider
        self.detail = detail
        #: Seconds the provider asked us to wait, when it said so at all.
        self.retry_after = retry_after


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """What one generation call consumed.

    Zeros rather than None when a provider does not report usage: the field is
    an accounting total, and a null that means "unknown" would be summed as if
    it meant "nothing". `reported` says which it was.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    reported: bool = True

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    """One streamed fragment.

    The final chunk carries `usage` and `done`; earlier ones carry text. Keeping
    them in one type means a caller consumes a single stream rather than
    reconciling a stream with a separate summary it has to wait for.
    """

    text: str = ""
    done: bool = False
    usage: TokenUsage | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """A model that streams an answer."""

    @property
    def name(self) -> str:
        """Stable provider identifier, persisted per message (§6.10)."""
        ...

    @property
    def model(self) -> str:
        """Pinned model id. Part of reproducibility: doc 7 §7.5 notes that some
        current models reject `temperature`, so a pinned id plus a pinned prompt
        version is what determinism rests on (FR-E4)."""
        ...

    def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        """Stream an answer, or raise `GenerationUnavailable`.

        Not `async def` returning an iterator: the call itself must be cheap, so
        the chain can hand the stream to the caller and let failures surface on
        first iteration — which is where a rate limit actually appears.
        """
        ...
