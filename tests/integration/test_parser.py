"""Parser validation against real PDFs and a real GROBID.

Skipped when either is unavailable — the fixture PDFs are gitignored (45 MB of
re-downloadable arXiv papers) and CI has no GROBID container, so this cannot be
a hard requirement of the default test run. `tests/unit/test_parsing.py` carries
the hermetic half that always runs.

The 20-paper sweep is one test rather than twenty parameterised ones on purpose:
it costs ~55 s of GROBID time, and the interesting result is the *aggregate*
(ASM-3's ≥85% bar, D39's coverage claim), not which individual paper failed.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from scholiast.application.ports.parsing import DocumentUnparseable
from scholiast.domain.chunking import BlockKind, StructureAwareChunker
from scholiast.domain.errors import FailureClass
from scholiast.infrastructure.parsing.grobid import GrobidParser

pytestmark = [pytest.mark.integration]

FIXTURES = Path(__file__).resolve().parents[2] / "spike" / "fixtures"
GROBID_URL = "http://localhost:8070"


def _grobid_alive() -> bool:
    try:
        with httpx.Client(timeout=5.0) as client:
            return client.get(f"{GROBID_URL}/api/isalive").status_code == 200
    except httpx.HTTPError:
        return False


needs_grobid = pytest.mark.skipif(not _grobid_alive(), reason="GROBID is not running")
needs_fixtures = pytest.mark.skipif(
    not FIXTURES.exists(), reason="fixture PDFs are gitignored; run spike/scripts/02 to fetch"
)


@pytest.fixture(scope="module")
def parser() -> GrobidParser:
    return GrobidParser(GROBID_URL)


def _papers() -> list[Path]:
    return sorted(p for p in FIXTURES.glob("*.pdf") if not p.name.startswith("_"))


@needs_grobid
@needs_fixtures
class TestBrokenPdfsAreClassified:
    """FR-I9: the three synthetic pathological cases, reproduced through the adapter.

    M0 measured these with a standalone script; this asserts the shipped code
    reaches the same verdicts, so the taxonomy is a property of the system rather
    than of a spike.
    """

    async def test_an_encrypted_pdf_is_unparseable(self, parser: GrobidParser) -> None:
        # Not TRANSIENT: no amount of retrying supplies the password, and telling
        # the user "retrying..." forever is worse than telling them it cannot work.
        with pytest.raises(DocumentUnparseable) as caught:
            await parser.parse((FIXTURES / "_broken-encrypted.pdf").read_bytes())
        assert caught.value.failure_class is FailureClass.UNPARSEABLE
        assert not caught.value.failure_class.retryable

    async def test_a_scanned_pdf_is_reported_as_a_scan(self, parser: GrobidParser) -> None:
        # SCANNED_IMAGE, not UNPARSEABLE: OCR would fix it (FR-I12), so the user
        # gets an accurate reason instead of "corrupted", which is both wrong and
        # unactionable.
        with pytest.raises(DocumentUnparseable) as caught:
            await parser.parse((FIXTURES / "_broken-scanned.pdf").read_bytes())
        assert caught.value.failure_class is FailureClass.SCANNED_IMAGE
        assert "scan" in caught.value.failure_class.user_message.lower()

    async def test_a_pdf_with_no_text_layer_is_a_scan(self, parser: GrobidParser) -> None:
        with pytest.raises(DocumentUnparseable) as caught:
            await parser.parse((FIXTURES / "_broken-empty.pdf").read_bytes())
        assert caught.value.failure_class is FailureClass.SCANNED_IMAGE

    async def test_an_oversized_upload_is_rejected_before_grobid(
        self, parser: GrobidParser
    ) -> None:
        # NFR-P8, and the check is local: spending a GROBID round-trip to learn
        # a file is too big wastes the worker and the user's time.
        with pytest.raises(DocumentUnparseable) as caught:
            await parser.parse(b"%PDF-1.4" + b"\0" * (51 * 1024 * 1024))
        assert caught.value.failure_class is FailureClass.TOO_LARGE


@needs_grobid
@needs_fixtures
class TestTheFixtureCorpusParses:
    """The aggregate claims: ASM-3's bar, D39's coverage, D42's round trip."""

    @pytest.fixture(scope="class")
    @classmethod
    async def parsed(cls, parser: GrobidParser) -> dict[str, object]:
        results: dict[str, object] = {}
        for pdf in _papers():
            try:
                results[pdf.stem] = await parser.parse(pdf.read_bytes(), filename=pdf.name)
            except DocumentUnparseable as exc:
                results[pdf.stem] = exc
        return results

    async def test_every_fixture_yields_usable_structure(self, parsed: dict[str, object]) -> None:
        # ASM-3 sets the bar at 85%. D19 established that "usable" means section
        # *bodies*, not headings — the distinction that reversed the M0 verdict.
        docs = {k: v for k, v in parsed.items() if not isinstance(v, DocumentUnparseable)}
        assert len(parsed) == 20, f"expected 20 fixtures, found {len(parsed)}"
        rate = len(docs) / len(parsed)
        assert rate >= 0.85, f"parse rate {rate:.0%} is below ASM-3's 85% bar"

        with_body = [d for d in docs.values() if d.has_body]  # type: ignore[union-attr]
        assert len(with_body) == len(docs), "a parsed paper with no body is not usable"

    async def test_every_block_indexes_its_section_text(self, parsed: dict[str, object]) -> None:
        # D42/D45 on real data. ParsedSection's constructor already enforces it,
        # so this passing across 20 real papers is evidence the parser never has
        # to be trusted about it — every one of these went through the check.
        checked = 0
        for doc in parsed.values():
            if isinstance(doc, DocumentUnparseable):
                continue
            for section in doc.sections:  # type: ignore[union-attr]
                for block in section.blocks:
                    assert section.text[block.char_start : block.char_end] == block.text
                    checked += 1
        assert checked > 1000, f"only {checked} blocks checked — corpus looks too small"

    async def test_pages_are_real_and_one_indexed(self, parsed: dict[str, object]) -> None:
        # D39's payoff. If coordinates were silently absent, every block would
        # fall back to page 1 — which is exactly what the comma-joined
        # teiCoordinates bug produced, and it looks like success.
        pages: set[int] = set()
        for doc in parsed.values():
            if isinstance(doc, DocumentUnparseable):
                continue
            for section in doc.sections:  # type: ignore[union-attr]
                for block in section.blocks:
                    assert block.page >= 1
                    pages.add(block.page)
        assert max(pages) > 5, "no block past page 5 — coordinates are probably missing"

    async def test_tables_survive_with_their_captions(self, parsed: dict[str, object]) -> None:
        # D41: what is honestly claimable. Not that cells are faithful — 28.6% of
        # tables are ragged — but that a table is one atomic block and never
        # loses its caption, so it still retrieves.
        tables = [
            block
            for doc in parsed.values()
            if not isinstance(doc, DocumentUnparseable)
            for section in doc.sections  # type: ignore[union-attr]
            for block in section.blocks
            if block.kind is BlockKind.TABLE
        ]
        assert len(tables) >= 50, f"only {len(tables)} tables across the corpus"
        assert all(block.is_atomic for block in tables)
        assert all(block.text.strip() for block in tables)

    async def test_the_corpus_chunks_end_to_end(self, parsed: dict[str, object]) -> None:
        # The handoff Phase 5 depends on: parser output feeds the chunker without
        # translation. D19's whole argument was that PyMuPDF could not do this.
        chunker = StructureAwareChunker()
        total = 0
        for name, doc in parsed.items():
            if isinstance(doc, DocumentUnparseable):
                continue
            result = chunker.chunk(name, doc.sections)  # type: ignore[union-attr]
            assert result.chunks, f"{name} produced no chunks"
            for chunk in result.chunks:
                assert chunk.provenance.char_end > chunk.provenance.char_start
                assert chunk.provenance.page >= 1
            total += len(result.chunks)
        assert total > 500, f"only {total} chunks from 20 papers"

    async def test_metadata_is_extracted_for_the_corpus(self, parsed: dict[str, object]) -> None:
        # Feeds FR-D3 dedup and the contextual chunk header (ML-3), which needs a
        # real paper title to be worth its ~20 tokens.
        docs = [d for d in parsed.values() if not isinstance(d, DocumentUnparseable)]
        titled = [d for d in docs if d.title]  # type: ignore[union-attr]
        assert len(titled) / len(docs) >= 0.9
        assert all(d.raw_tei for d in docs)  # type: ignore[union-attr]
        assert all(d.parser_version for d in docs)  # type: ignore[union-attr]
