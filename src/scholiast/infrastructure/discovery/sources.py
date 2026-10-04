"""Assembling the discovery sources from configuration (D118, D122, D123, D125).

Not a source itself: which sources exist, in which order, and what each is wrapped
in. Like `build_provider_chain`, it returns a list the caller can see and a test can
assert, because a fan-out that silently contained one source would look identical
to one that contained two until the day the first failed.

**OpenAlex first, and only with a key** (D118). A source with no credential is
absent, not broken: an OpenAlex client built around an empty key would turn every
search into a 401 that the fan-out reads as an outage. First, because the order is
the interleave order until phase 8 ranks (D122), and OpenAlex is the primary.

**Every source sits behind its own breaker** (D104, D122), one per instance, so a
rate-limited OpenAlex never stops the fan-out asking a healthy arXiv.

**arXiv's pacer sits INSIDE its breaker** (D123). The breaker refuses an open
source before the pacer is consulted, so an open breaker spends no pacing slot; and
the pacer's refusals are marked unsent, so the breaker does not count them.
"""

from __future__ import annotations

from scholiast.application.use_cases.discover_papers import ConfiguredSource
from scholiast.config.settings import Settings
from scholiast.infrastructure.discovery.arxiv import ArxivSource
from scholiast.infrastructure.discovery.breaker import SourceBreaker
from scholiast.infrastructure.discovery.openalex import OpenAlexSource
from scholiast.infrastructure.discovery.pacing import PacedSource


def build_discovery_sources(settings: Settings) -> list[ConfiguredSource]:
    """The sources this deployment can actually ask, in interleave order."""

    def guarded(source: OpenAlexSource | PacedSource) -> SourceBreaker:
        return SourceBreaker(
            source,
            failure_threshold=settings.breaker_failure_threshold,
            cooldown_seconds=settings.breaker_cooldown_seconds,
        )

    sources: list[ConfiguredSource] = []
    if settings.openalex_api_key:
        sources.append(
            ConfiguredSource(
                guarded(OpenAlexSource(settings.openalex_api_key)),
                settings.discovery_openalex_page_size,
            )
        )
    arxiv = PacedSource(ArxivSource(), min_gap_seconds=settings.discovery_arxiv_min_gap_seconds)
    sources.append(ConfiguredSource(guarded(arxiv), settings.discovery_arxiv_page_size))
    return sources
