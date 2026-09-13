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
"""

from __future__ import annotations

from scholiast.application.ports.llm import LLMProvider
from scholiast.config.settings import Settings
from scholiast.infrastructure.llm.groq import GroqProvider
from scholiast.infrastructure.llm.ollama import OllamaProvider


def build_provider_chain(settings: Settings) -> list[LLMProvider]:
    """The ordered chain this deployment can actually serve.

    Returns a list rather than a single composed object so the caller can see —
    and a test can assert — exactly which rungs exist. A chain that silently
    contained one provider would look identical to one that contained three
    until the day the first fails.
    """
    chain: list[LLMProvider] = [OllamaProvider(settings.ollama_url, model=settings.ollama_model)]
    if settings.groq_api_key:
        chain.append(GroqProvider(settings.groq_api_key, model=settings.groq_model))
    return chain
