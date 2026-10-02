"""Assembling the provider chain from configuration (D26, D58, §5.11).

Not a provider itself. The chain's *behaviour* — try each rung, disclose every
transition — lives in the use case, because "which degradation do we report" is
a product decision that §5.11 specifies and not something an adapter should own.
What lives here is the ordering and the question of which rungs exist at all,
both of which are configuration.

**Ollama first** (D26). The obvious reading of §5.11's ladder is hosted-then-
local; M0's measurements inverted it. Groq's free tier is ~2 RAG requests per
minute, so hosted-first makes the development default unusable after two
questions, while the local 3B sustains 125 tok/s with no quota.

A provider with no credential is **absent, not broken**. Building a Groq client
around an empty API key would turn every request into a 401 that the chain reads
as a transient outage — a misconfiguration wearing a failure's clothes.

**Every rung sits behind a circuit breaker, the local one included** (D104). It
is tempting to exempt Ollama on the grounds that localhost cannot be slow, but
an Ollama that is not running refuses fast while one that is *hung* does not, and
the second case is the one that costs a timeout on every question. The breaker
does not change what the user is told — the use case still sees
`GenerationUnavailable` and still discloses `provider_failover` — it changes what
discovering an outage costs.
"""

from __future__ import annotations

from scholiast.application.ports.llm import LLMProvider
from scholiast.config.settings import Settings
from scholiast.infrastructure.llm.breaker import CircuitBreaker
from scholiast.infrastructure.llm.groq import GroqProvider
from scholiast.infrastructure.llm.ollama import OllamaProvider


def build_provider_chain(settings: Settings) -> list[LLMProvider]:
    """The ordered chain this deployment can actually serve.

    Returns a list rather than a single composed object so the caller can see —
    and a test can assert — exactly which rungs exist. A chain that silently
    contained one provider would look identical to one that contained three
    until the day the first fails.
    """
    providers: list[LLMProvider] = [
        OllamaProvider(settings.ollama_url, model=settings.ollama_model)
    ]
    if settings.groq_api_key:
        providers.append(GroqProvider(settings.groq_api_key, model=settings.groq_model))

    # One breaker per provider INSTANCE, not one per chain (D104): the states are
    # independent facts, and a shared breaker would let a rate-limited Groq stop
    # the chain from trying a healthy Ollama.
    return [
        CircuitBreaker(
            provider,
            failure_threshold=settings.breaker_failure_threshold,
            cooldown_seconds=settings.breaker_cooldown_seconds,
        )
        for provider in providers
    ]
