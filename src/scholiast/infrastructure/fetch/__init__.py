"""Fetching PDFs from links a third party supplied, behind D115's guard (D129)."""

from __future__ import annotations

from scholiast.config.settings import Settings
from scholiast.infrastructure.fetch.guarded import GuardedPdfFetcher, HostPacer

__all__ = ["build_pdf_fetcher"]


def build_pdf_fetcher(settings: Settings) -> GuardedPdfFetcher:
    """The fetcher this deployment would use. Phase 6's FETCHING stage is its caller."""
    return GuardedPdfFetcher(
        allowed_hosts=settings.fetch_allowed_hosts,
        pacer=HostPacer(
            default_gap=settings.fetch_min_gap_seconds,
            gaps={"arxiv.org": settings.fetch_arxiv_min_gap_seconds},
        ),
        total_timeout=settings.fetch_total_timeout_seconds,
    )
