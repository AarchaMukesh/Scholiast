"""Parser adapter tests, hermetic — no GROBID, no fixture PDFs.

The TEI here is hand-written so each test isolates one decision. The 20-fixture
validation lives in `tests/integration/test_parser.py`; this file is what runs in
CI, where neither a GROBID container nor the gitignored fixture PDFs exist.
"""

from __future__ import annotations

import fitz
import httpx
import pytest
import respx

from scholiast.application.ports.parsing import DocumentUnparseable
from scholiast.domain.chunking import BlockKind
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import SectionKind
from scholiast.infrastructure.parsing.grobid import GrobidParser
from scholiast.infrastructure.parsing.preflight import MAX_PDF_BYTES

GROBID = "http://grobid.test:8070"


def tei(body: str, *, header: str = "") -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc>
      <titleStmt><title>Lost in the Middle</title></titleStmt>
      <sourceDesc><biblStruct><analytic>
        <author><persName>Nelson Liu</persName></author>
      </analytic></biblStruct></sourceDesc>
    </fileDesc>
    <profileDesc><abstract><p>We analyse long-context behaviour.</p></abstract></profileDesc>
    {header}
  </teiHeader>
  <text><body>{body}</body></text>
</TEI>"""


def parse(body: str) -> object:
    """Run the TEI mapping directly, skipping HTTP and the PDF pre-flight."""
    return GrobidParser(GROBID)._to_document(tei(body), page_count=12)


def minimal_pdf(
    *, lines: int = 20, text_line: str = "Retrieval quality is measured here."
) -> bytes:
    """A real PDF built at test time, with a genuine text layer.

    Generated rather than committed so the composition test runs in CI, where the
    fixture corpus is gitignored. It must carry more than
    `MIN_EXTRACTABLE_CHARS`, or pre-flight would classify it as a scan and the
    test would pass for the wrong reason.
    """
    document = fitz.open()
    page = document.new_page()
    for index in range(lines):
        page.insert_text((72, 72 + index * 14), f"{index}. {text_line}")
    pdf: bytes = document.tobytes()
    document.close()
    return pdf


class TestParseComposesPreflightFetchAndMapping:
    """CI coverage for the public entry point, not just its private parts.

    The other tests exercise `_to_document` and `_fetch_tei` separately — a
    useful seam, but it leaves the composition untested everywhere except the
    integration suite, which skips without GROBID and without the fixture PDFs.
    A dropped pre-flight call or a mis-threaded page_count would keep CI green.
    """

    @respx.mock
    async def test_a_real_pdf_parses_end_to_end(self) -> None:
        route = respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            return_value=httpx.Response(
                200,
                text=tei(
                    '<div><head n="2">Method</head>'
                    '<p coords="3,0,0,1,1"><s coords="3,0,0,1,1">We train on GPUs.</s></p>'
                    "</div>"
                ),
            )
        )
        pdf = minimal_pdf()
        doc = await GrobidParser(GROBID).parse(pdf, filename="paper.pdf")

        assert route.called, "GROBID was never called"
        (section,) = doc.sections
        assert section.kind is SectionKind.METHOD
        assert section.blocks[0].page == 3
        assert doc.title == "Lost in the Middle"
        assert doc.raw_tei is not None
        # page_count comes from pre-flight, not from TEI — proving the two
        # halves were actually composed rather than each working alone.
        assert doc.page_count == 1

    @respx.mock
    async def test_the_pdf_bytes_reach_grobid(self) -> None:
        # Prevents the composition sending the wrong thing — an empty body, or
        # the filename instead of the file — which the separate tests cannot see.
        route = respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            return_value=httpx.Response(
                200, text=tei("<div><head>M</head><p><s>Body.</s></p></div>")
            )
        )
        pdf = minimal_pdf()
        await GrobidParser(GROBID).parse(pdf, filename="upload.pdf")

        sent = route.calls.last.request.content
        assert pdf in sent, "the PDF bytes were not the upload body"
        assert b'filename="upload.pdf"' in sent

    @respx.mock
    async def test_preflight_runs_before_grobid_is_called(self) -> None:
        # The composition's whole point. A scanned PDF must be rejected locally,
        # so a corpus of scans cannot spend a GROBID round-trip each. If the
        # pre-flight call were dropped, this route would be called and the
        # failure would arrive later and less precisely classified.
        route = respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            return_value=httpx.Response(200, text=tei("<div><head>M</head><p><s>x</s></p></div>"))
        )
        blank = fitz.open()
        blank.new_page()
        scanned = blank.tobytes()
        blank.close()

        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID).parse(scanned, filename="scan.pdf")

        assert caught.value.failure_class is FailureClass.SCANNED_IMAGE
        assert not route.called, "pre-flight was skipped; GROBID was called on a scan"

    @respx.mock
    async def test_an_oversized_upload_never_reaches_grobid(self) -> None:
        route = respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            return_value=httpx.Response(200, text=tei("<div><head>M</head><p><s>x</s></p></div>"))
        )
        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID).parse(b"%PDF-1.4" + bytes(MAX_PDF_BYTES + 1))
        assert caught.value.failure_class is FailureClass.TOO_LARGE
        assert not route.called

    @respx.mock
    async def test_a_transient_grobid_failure_propagates_through_parse(self) -> None:
        # The retry decision has to survive the composition, not just _fetch_tei.
        respx.post(f"{GROBID}/api/processFulltextDocument").mock(return_value=httpx.Response(503))
        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID).parse(minimal_pdf(), filename="paper.pdf")
        assert caught.value.failure_class is FailureClass.TRANSIENT
        assert caught.value.failure_class.retryable


class TestPageNumbersComeFromCoordinates:
    """D39: TEI supplies pages; the M0 conclusion that it could not was an artifact."""

    def test_a_sentence_reports_the_page_its_coords_name(self) -> None:
        doc = parse(
            '<div><head n="2">Method</head>'
            '<p coords="4,10,10,100,10">'
            '<s coords="7,10,10,100,10">We train on eight GPUs.</s>'
            "</p></div>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        # 7, not 4: the sentence's own coordinates win over the paragraph's.
        assert section.blocks[0].page == 7

    def test_a_paragraph_without_coords_falls_back_to_page_one(self) -> None:
        # Never raise on missing coordinates — a page is required by Provenance,
        # and refusing to parse a paper over one absent attribute trades a whole
        # document for a slightly wrong page number.
        doc = parse("<div><head>Method</head><p><s>No coordinates here.</s></p></div>")
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert section.blocks[0].page == 1


class TestParagraphsSplitAtPageBoundaries:
    """D40: 6.81% of sentences would otherwise report a page that is too low."""

    def test_a_straddling_paragraph_becomes_two_blocks(self) -> None:
        doc = parse(
            '<div><head n="3">Experiments</head><p coords="4,0,0,1,1">'
            '<s coords="4,0,0,1,1">First sentence on page four.</s>'
            '<s coords="4,0,0,1,1">Still page four.</s>'
            '<s coords="5,0,0,1,1">This one spilled onto page five.</s>'
            "</p></div>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert [b.page for b in section.blocks] == [4, 5]
        assert "Still page four" in section.blocks[0].text
        assert "spilled onto page five" in section.blocks[1].text

    def test_a_paragraph_on_one_page_stays_one_block(self) -> None:
        # The split must be driven by the data, not applied per sentence — one
        # block per sentence would destroy the paragraph coherence the chunker
        # relies on.
        doc = parse(
            '<div><head>Method</head><p coords="2,0,0,1,1">'
            '<s coords="2,0,0,1,1">One.</s><s coords="2,0,0,1,1">Two.</s>'
            "</p></div>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert len(section.blocks) == 1
        assert section.blocks[0].text == "One. Two."


class TestSectionTextAndOffsetsAgree:
    """D42/D45: the round trip holds on real parser output, not just in tests."""

    def test_every_block_indexes_the_section_text(self) -> None:
        # ParsedSection's constructor enforces this, so reaching here at all
        # means the parser assembled text and offsets consistently. Asserting it
        # again states the property the constructor is protecting.
        doc = parse(
            '<div><head n="3">Experiments</head>'
            '<p coords="1,0,0,1,1"><s coords="1,0,0,1,1">First paragraph text.</s></p>'
            '<p coords="2,0,0,1,1"><s coords="2,0,0,1,1">Second paragraph text.</s></p>'
            "</div>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert len(section.blocks) == 2
        for block in section.blocks:
            assert section.text[block.char_start : block.char_end] == block.text

    def test_offsets_restart_at_zero_in_each_section(self) -> None:
        # The specific thing D42 changed. If offsets accumulated across sections,
        # the second section's blocks would start where the first ended and index
        # the wrong string entirely.
        doc = parse(
            '<div><head n="1">Introduction</head><p><s>Intro sentence here.</s></p></div>'
            '<div><head n="2">Method</head><p><s>Method sentence here.</s></p></div>'
        )
        first, second = doc.sections  # type: ignore[attr-defined]
        assert first.blocks[0].char_start == 0
        assert second.blocks[0].char_start == 0


class TestSectionHierarchy:
    """Breadcrumb paths (doc 6 §6.3), rebuilt from numbering because TEI is flat."""

    def test_a_numbered_subsection_names_its_parent(self) -> None:
        doc = parse(
            '<div><head n="4">Experiments</head><p><s>Setup described here.</s></p></div>'
            '<div><head n="4.1">Multi-Document QA</head><p><s>Details here now.</s></p></div>'
        )
        parent, child = doc.sections  # type: ignore[attr-defined]
        assert parent.path == "Experiments"
        assert child.path == "Experiments > Multi-Document QA"

    def test_an_unnumbered_head_gets_a_flat_path(self) -> None:
        # Honest degradation: an unknown parent is better than an invented one,
        # and about two thirds of GROBID's heads carry no numbering.
        doc = parse("<div><head>Acknowledgements</head><p><s>Thanks to everyone.</s></p></div>")
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert section.path == "Acknowledgements"

    def test_section_kind_is_normalised_not_raw(self) -> None:
        # Everything downstream switches on the canonical kind — attribution
        # most of all — so a raw heading reaching the domain would silently
        # disable the related-work marker.
        doc = parse(
            '<div><head n="2">Related Work</head><p><s>Others did this first.</s></p></div>'
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert section.kind is SectionKind.RELATED_WORK
        assert section.kind.describes_other_work


class TestSectionPathsAreUniqueWithinADocument:
    """D51 — the property chunk-to-section mapping silently depends on.

    Chunks carry `provenance.section_path` and the repository resolves it to a
    `sections.id`. Two sections sharing a path do not collide loudly: one
    section's chunks are attributed to the other's text, and the citation
    highlights a span from the wrong part of the paper with nothing raising.
    """

    def test_two_headless_divs_do_not_share_a_path(self) -> None:
        # The common case: GROBID emits a head-less <div> for orphan prose, and
        # every one of them would otherwise be called "Untitled section".
        doc = parse(
            "<div><p><s>First orphan paragraph here.</s></p></div>"
            "<div><p><s>Second orphan paragraph here.</s></p></div>"
        )
        first, second = doc.sections  # type: ignore[attr-defined]
        assert first.path == "Untitled section"
        assert second.path == "Untitled section (2)"

    def test_repeated_headings_are_disambiguated_in_order(self) -> None:
        doc = parse(
            "<div><head>Results</head><p><s>Main results here.</s></p></div>"
            "<div><head>Results</head><p><s>Appendix results here.</s></p></div>"
            "<div><head>Results</head><p><s>Further results here.</s></p></div>"
        )
        assert [s.path for s in doc.sections] == [  # type: ignore[attr-defined]
            "Results",
            "Results (2)",
            "Results (3)",
        ]

    def test_the_raw_heading_is_kept_unsuffixed(self) -> None:
        # The suffix belongs to the path, which is an identifier. A reader view
        # showing "Results (2)" as the *heading* would report a document that
        # does not exist.
        doc = parse(
            "<div><head>Results</head><p><s>Main results here.</s></p></div>"
            "<div><head>Results</head><p><s>Appendix results here.</s></p></div>"
        )
        assert [s.raw_title for s in doc.sections] == ["Results", "Results"]  # type: ignore[attr-defined]

    def test_a_dropped_section_does_not_consume_a_path(self) -> None:
        # A head-only div is never persisted, so reserving its path would push a
        # counter onto the next real section for no reason.
        doc = parse(
            "<div><head>Results</head></div>"
            "<div><head>Results</head><p><s>The real results are here.</s></p></div>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert section.path == "Results"

    def test_distinct_headings_are_left_alone(self) -> None:
        # Positive control: a disambiguator that suffixed everything would pass
        # every test above while making all breadcrumbs wrong.
        doc = parse(
            "<div><head>Method</head><p><s>We propose something.</s></p></div>"
            "<div><head>Results</head><p><s>It works well enough.</s></p></div>"
        )
        assert [s.path for s in doc.sections] == ["Method", "Results"]  # type: ignore[attr-defined]

    def test_subsections_inherit_the_disambiguated_parent(self) -> None:
        doc = parse(
            '<div><head n="3">Results</head><p><s>First results block.</s></p></div>'
            '<div><head n="3">Results</head><p><s>Second results block.</s></p></div>'
            '<div><head n="3.1">Ablations</head><p><s>Ablation details here.</s></p></div>'
        )
        assert doc.sections[2].path == "Results (2) > Ablations"  # type: ignore[attr-defined]


class TestTables:
    """D41: caption-first, because captions are what survive and what people search."""

    def test_a_table_leads_with_its_caption(self) -> None:
        doc = parse(
            "<div><head>Results</head>"
            "<p><s>We report results below.</s></p></div>"
            # Sibling of the div, not nested — GROBID never nests floats.
            '<figure type="table" coords="6,0,0,1,1">'
            "<figDesc>Table 3: accuracy on MMLU</figDesc>"
            "<table><row><cell>Model</cell><cell>Acc</cell></row>"
            "<row><cell>GPT-4</cell><cell>86.4</cell></row></table>"
            "</figure>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        table = next(b for b in section.blocks if b.kind is BlockKind.TABLE)
        assert table.text.startswith("Table 3: accuracy on MMLU")
        assert "| GPT-4 | 86.4 |" in table.text
        assert table.page == 6

    def test_a_table_is_an_atomic_block(self) -> None:
        # The chunker keys off is_atomic to keep tables whole; classifying a
        # table as prose would let it be split mid-row.
        doc = parse(
            "<div><head>Results</head><p><s>Prose sentence here.</s></p></div>"
            '<figure type="table"><figDesc>Cap</figDesc>'
            "<table><row><cell>a</cell></row></table></figure>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert any(b.is_atomic for b in section.blocks)

    def test_a_table_outside_its_div_still_reaches_a_section(self) -> None:
        # The exact bug the corpus test caught: GROBID emits <figure> as a
        # SIBLING of <div>, so looking for tables inside a div finds zero. That
        # is a silent miss — the paper parses, sections have text, and tables
        # simply never enter the index. Document order decides the owner.
        doc = parse(
            "<div><head>Method</head><p><s>Method text here.</s></p></div>"
            '<figure type="table"><figDesc>Table 1: setup</figDesc>'
            "<table><row><cell>lr</cell><cell>3e-4</cell></row></table></figure>"
            "<div><head>Results</head><p><s>Results text here.</s></p></div>"
        )
        method, results = doc.sections  # type: ignore[attr-defined]
        assert any(b.kind is BlockKind.TABLE for b in method.blocks), (
            "the table belongs to the section it follows"
        )
        assert all(b.kind is not BlockKind.TABLE for b in results.blocks)

    def test_a_non_table_figure_is_ignored(self) -> None:
        # Figure images carry no retrievable text; indexing their captions as
        # prose blocks would dilute the section text with fragments.
        doc = parse(
            "<div><head>Results</head><p><s>Prose sentence here.</s></p></div>"
            "<figure><figDesc>Figure 1: architecture</figDesc></figure>"
        )
        (section,) = doc.sections  # type: ignore[attr-defined]
        assert all(b.kind is BlockKind.PARAGRAPH for b in section.blocks)


class TestMetadata:
    def test_header_metadata_is_extracted(self) -> None:
        doc = parse("<div><head>Method</head><p><s>Some method text.</s></p></div>")
        assert doc.title == "Lost in the Middle"  # type: ignore[attr-defined]
        assert doc.authors == ("Nelson Liu",)  # type: ignore[attr-defined]
        assert doc.abstract is not None  # type: ignore[attr-defined]

    def test_raw_tei_is_retained(self) -> None:
        # D43: kept so re-chunking ablations are reproducible across a GROBID
        # version change rather than silently re-baselined.
        doc = parse("<div><head>Method</head><p><s>Some method text.</s></p></div>")
        assert doc.raw_tei is not None  # type: ignore[attr-defined]
        assert "Lost in the Middle" in doc.raw_tei  # type: ignore[attr-defined]

    def test_the_parser_version_is_stamped(self) -> None:
        doc = parse("<div><head>Method</head><p><s>Some method text.</s></p></div>")
        assert doc.parser_version  # type: ignore[attr-defined]
        assert doc.parser_version == GrobidParser(GROBID).version  # type: ignore[attr-defined]


class TestFailureClassification:
    """FR-I8/I9: errors are classified, because retry and message both depend on it."""

    def test_tei_with_no_body_sections_is_unparseable(self) -> None:
        # D19's whole point: the chunker consumes section *content*. TEI that
        # parses but yields no bodies is a successful request and a failed parse.
        with pytest.raises(DocumentUnparseable) as caught:
            parse("")
        assert caught.value.failure_class is FailureClass.UNPARSEABLE
        assert not caught.value.failure_class.retryable

    def test_a_section_with_only_a_heading_is_dropped(self) -> None:
        # A heading with no body cannot feed the chunker, so it must not become
        # an empty section that later looks like missing content.
        doc = parse(
            "<div><head>Empty Section</head></div>"
            "<div><head>Method</head><p><s>Real content here.</s></p></div>"
        )
        assert [s.path for s in doc.sections] == ["Method"]  # type: ignore[attr-defined]

    def test_malformed_tei_is_unparseable(self) -> None:
        with pytest.raises(DocumentUnparseable) as caught:
            GrobidParser(GROBID)._to_document("<TEI><unclosed>", page_count=1)
        assert caught.value.failure_class is FailureClass.UNPARSEABLE

    @respx.mock
    async def test_grobid_being_down_is_transient_not_unparseable(self) -> None:
        # The distinction that decides whether the paper is retried or abandoned:
        # a 503 says try later, and misclassifying it as UNPARSEABLE would tell
        # the user their valid PDF is corrupt and offer no retry.
        respx.post(f"{GROBID}/api/processFulltextDocument").mock(return_value=httpx.Response(503))
        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID)._fetch_tei(b"%PDF-1.4", filename="x.pdf")
        assert caught.value.failure_class is FailureClass.TRANSIENT
        assert caught.value.failure_class.retryable

    @respx.mock
    async def test_a_connection_error_is_transient(self) -> None:
        respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID)._fetch_tei(b"%PDF-1.4", filename="x.pdf")
        assert caught.value.failure_class is FailureClass.TRANSIENT

    @respx.mock
    async def test_grobid_204_is_unparseable(self) -> None:
        # GROBID answers 204 when it extracted nothing. Retrying is pointless.
        respx.post(f"{GROBID}/api/processFulltextDocument").mock(return_value=httpx.Response(204))
        with pytest.raises(DocumentUnparseable) as caught:
            await GrobidParser(GROBID)._fetch_tei(b"%PDF-1.4", filename="x.pdf")
        assert caught.value.failure_class is FailureClass.UNPARSEABLE

    @respx.mock
    async def test_coordinates_are_requested_as_repeated_fields(self) -> None:
        # The bug that made M0 conclude TEI has no page numbers (D39): a
        # comma-joined teiCoordinates returns 200 with valid TEI and zero coords.
        # Nothing downstream can detect that, so the request shape is asserted
        # here rather than trusted.
        route = respx.post(f"{GROBID}/api/processFulltextDocument").mock(
            return_value=httpx.Response(200, text=tei("<div><head>M</head></div>"))
        )
        await GrobidParser(GROBID)._fetch_tei(b"%PDF-1.4", filename="x.pdf")

        body = route.calls.last.request.content.decode("utf-8", "replace")
        assert body.count('name="teiCoordinates"') >= 5, "coords must be repeated fields"
        assert "p,s,head" not in body, "comma-joined teiCoordinates returns no coords at all"
        assert 'name="segmentSentences"' in body
