"""GROBID TEI -> ParsedSection/ParsedBlock (D19, D31, D39-D41, D45).

Four measured decisions are encoded here rather than left to a reader:

**Coordinates must be requested, and as repeated fields.** `teiCoordinates` sent
as a comma-joined string returns perfectly valid TEI with zero `coords`
attributes — no error, no warning. That silent-success mode is why M0 concluded
"TEI lacks page offsets" and why D39 had to reverse it. Sent as repeated form
fields, coverage is 10,621/10,621 sentences across 20 fixtures at ~1.5% latency.

**Blocks split at page boundaries** (D40). 6.1% of paragraphs straddle a page
break, putting 6.81% of sentences on a later page than their paragraph starts on.
`ParsedBlock` carries one page, so a whole-paragraph block would send roughly one
citation in fifteen to the wrong page.

**Tables lead with their caption** (D41). Cell segmentation is imperfect — 28.6%
of tables are ragged, 7.6% extract almost nothing — but captions are present for
100% of them and are what people actually search ("Table 3: results on MMLU"). A
mangled table still retrieves on its caption instead of becoming index ballast.

**Section hierarchy comes from `head/@n`, not from XML nesting.** GROBID's body
is flat: 158 top-level `<div>` and zero nested across the fixture sample. Numbered
heads ("3.1") are what makes the breadcrumb paths of doc 6 §6.3 possible, and only
about a third of heads carry numbering — so hierarchy is partial by construction,
and reported as such rather than implied to be complete.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from scholiast.application.ports.parsing import (
    DocumentUnparseable,
    ParsedDocument,
)
from scholiast.domain.chunking import (
    SECTION_PATH_SEPARATOR,
    UNTITLED_SECTION,
    BlockKind,
    ParsedBlock,
    ParsedSection,
    unique_section_path,
)
from scholiast.domain.errors import FailureClass
from scholiast.domain.paper import SectionKind, normalise_section
from scholiast.infrastructure.parsing.preflight import preflight

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

#: Bump whenever anything here changes what offsets or sections come out.
#: Stamped onto the paper, and offsets are only meaningful with it (FR-I5).
# 1.1.0: section paths are made unique within a document (D51) and the raw
# heading is carried alongside the path. Paths reach the contextual header, so
# this changes embedding input for colliding sections — free today because
# nothing is indexed yet, a re-index if it had been deferred.
PARSER_VERSION = "grobid-crf-0.8.2.1/tei/1.1.0"

#: Structures worth coordinates. `s` is the load-bearing one — a child chunk is a
#: run of sentences, so sentence-level pages are what make per-chunk pages exact.
COORD_ELEMENTS = ("p", "s", "head", "figure", "table")

#: Blocks are joined by this, and the section text is built from the same join,
#: so `ParsedSection`'s invariant holds by construction rather than by care.
BLOCK_JOINER = "\n\n"

_WHITESPACE = re.compile(r"\s+")


def _flatten(element: ET.Element | None) -> str:
    """All descendant text, whitespace-normalised.

    Normalising here rather than later is deliberate: offsets are computed from
    these strings, so any later normalisation would shift every span (the
    failure `ParsedBlock`'s length invariant exists to catch).
    """
    if element is None:
        return ""
    return _WHITESPACE.sub(" ", "".join(element.itertext())).strip()


def _first_page(coords: str | None) -> int | None:
    """The page a `coords` attribute starts on.

    Format is `page,x,y,w,h` per line box, semicolon-separated. The first box is
    where the element begins, which is where a reader should be sent.
    """
    if not coords:
        return None
    head = coords.split(";")[0].split(",")[0]
    return int(head) if head.isdigit() else None


def _table_markdown(figure: ET.Element) -> str:
    """Serialise a TEI table caption-first (D41)."""
    caption = _flatten(figure.find("tei:figDesc", TEI_NS))
    lines: list[str] = []
    if caption:
        lines.append(caption)

    table = figure.find("tei:table", TEI_NS)
    if table is not None:
        for row in table.findall("tei:row", TEI_NS):
            cells = [_flatten(c) for c in row.findall("tei:cell", TEI_NS)]
            if any(cells):
                lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


class GrobidParser:
    """The `DocumentParser` adapter."""

    def __init__(self, base_url: str, *, timeout: float = 300.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def version(self) -> str:
        return PARSER_VERSION

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument:
        # Local checks first: a scan or an encrypted file is decided in
        # milliseconds here versus ~3 s and an un-typed 4xx from GROBID.
        page_count = preflight(pdf, filename=filename)
        tei = await self._fetch_tei(pdf, filename=filename)
        return self._to_document(tei, page_count=page_count)

    async def _fetch_tei(self, pdf: bytes, *, filename: str) -> str:
        data: dict[str, str | list[str]] = {
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            # Sentence segmentation is what carries per-sentence coordinates.
            "segmentSentences": "1",
            # A LIST, not a comma-joined string. httpx renders a list value as
            # repeated form fields, which is what GROBID needs; the comma form
            # returns 200 with valid TEI and no coords at all — a silent success
            # indistinguishable from the feature being unavailable.
            "teiCoordinates": list(COORD_ELEMENTS),
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/api/processFulltextDocument",
                    files={"input": (filename or "paper.pdf", pdf, "application/pdf")},
                    data=data,
                )
        except httpx.HTTPError as exc:
            # The service being down is TRANSIENT — retrying is the right
            # response, unlike every other class this module raises.
            raise DocumentUnparseable(
                FailureClass.TRANSIENT, f"GROBID unreachable: {type(exc).__name__}: {exc}"[:200]
            ) from exc

        if response.status_code == 204:
            raise DocumentUnparseable(FailureClass.UNPARSEABLE, "GROBID extracted no content")
        if response.status_code >= 500:
            raise DocumentUnparseable(FailureClass.TRANSIENT, f"GROBID {response.status_code}")
        if response.status_code != 200:
            raise DocumentUnparseable(
                FailureClass.UNPARSEABLE,
                f"GROBID {response.status_code}: {response.text[:160]}",
            )
        return response.text

    def _to_document(self, tei: str, *, page_count: int) -> ParsedDocument:
        try:
            root = ET.fromstring(tei)  # noqa: S314 - our own GROBID, on our own network
        except ET.ParseError as exc:
            raise DocumentUnparseable(
                FailureClass.UNPARSEABLE, f"TEI is not well-formed: {exc}"
            ) from exc

        header = root.find(".//tei:teiHeader", TEI_NS)
        # `is not None`, not truthiness: an Element with no children is
        # falsy, so `if header` would treat a childless header as absent.
        title = (
            _flatten(header.find(".//tei:titleStmt/tei:title", TEI_NS))
            if header is not None
            else ""
        )
        authors = tuple(
            name
            for person in root.findall(".//tei:sourceDesc//tei:author/tei:persName", TEI_NS)
            if (name := _flatten(person))
        )
        abstract = _flatten(root.find(".//tei:profileDesc/tei:abstract", TEI_NS))

        body = root.find(".//tei:text/tei:body", TEI_NS)
        sections = self._sections(body) if body is not None else ()

        if not sections:
            # Structurally valid TEI with no body is still unusable: the chunker
            # takes section *content*, so there is nothing to index (D19).
            raise DocumentUnparseable(
                FailureClass.UNPARSEABLE, "TEI contains no body sections with text"
            )

        blocks = sum(len(s.blocks) for s in sections)
        return ParsedDocument(
            sections=sections,
            title=title or None,
            authors=authors,
            abstract=abstract or None,
            page_count=page_count,
            parser_version=PARSER_VERSION,
            raw_tei=tei,
            stats={
                "sections": len(sections),
                "blocks": blocks,
                "tables": sum(1 for s in sections for b in s.blocks if b.kind is BlockKind.TABLE),
                "chars": sum(len(s.text) for s in sections),
            },
        )

    # -- section assembly --------------------------------------------------

    def _sections(self, body: ET.Element) -> tuple[ParsedSection, ...]:
        sections: list[ParsedSection] = []
        # Maps a heading number ("3") to its path, so "3.1" can find its parent.
        paths_by_number: dict[str, str] = {}
        # Paths already emitted for this document. Uniqueness is enforced here
        # rather than hoped for downstream — see `_unique_path` (D51).
        claimed: set[str] = set()

        for div, tables in self._grouped(body):
            head = div.find("tei:head", TEI_NS)
            raw_title = _flatten(head)
            number = head.get("n", "").rstrip(".") if head is not None else ""

            path = unique_section_path(self._path_for(raw_title, number, paths_by_number), claimed)
            if number:
                paths_by_number[number] = path

            section = self._section(div, tables, raw_title=raw_title, path=path)
            if section is not None:
                sections.append(section)
                # Claimed only once the section survives: a div with no text is
                # never persisted, so reserving its path would push a counter
                # onto the next real section for no reason.
                claimed.add(path)

        return tuple(sections)

    @staticmethod
    def _grouped(body: ET.Element) -> list[tuple[ET.Element, list[ET.Element]]]:
        """Pair each `<div>` with the tables that follow it.

        GROBID emits `<figure>` as a **sibling** of `<div>`, directly under
        `<body>` — never nested inside the section it belongs to. Looking for
        tables inside a div finds exactly zero, and that is a *silent* miss: the
        paper still parses, sections still have their text, and tables simply
        never enter the index. Only real-corpus data surfaced it.

        A float is emitted after the div it follows, so document order is both
        the owning section and the correct position — appending is already
        right. Tables before the first div are dropped: they have no section to
        belong to, and inventing one would misattribute them.
        """
        grouped: list[tuple[ET.Element, list[ET.Element]]] = []
        for element in body:
            tag = element.tag.split("}")[-1]
            if tag == "div":
                grouped.append((element, []))
            elif tag == "figure" and element.get("type") == "table" and grouped:
                grouped[-1][1].append(element)
        return grouped

    @staticmethod
    def _path_for(raw_title: str, number: str, paths_by_number: dict[str, str]) -> str:
        """Breadcrumb path, e.g. `Experiments > Multi-Document QA` (doc 6 §6.3).

        Hierarchy is reconstructed from heading numbering because GROBID's body
        is flat. When a head is unnumbered — about two thirds of them — the path
        is the title alone, which is honest: an unknown parent is better than an
        invented one.

        `UNTITLED_SECTION` is the domain's general name for a section with no
        heading; **this adapter is what makes it common.** GROBID emits a
        head-less `<div>` for orphan prose, and every one of them lands on that
        same name — which, with only ~29% of heads carrying numbering, makes
        these the main source of the path collisions `unique_section_path`
        exists to resolve (D51). The eval harness shares the constant but not
        the incidence: QASPER JSON produced 1 unnamed section in 3,796.
        """
        title = raw_title or UNTITLED_SECTION
        if "." not in number:
            return title
        parent_number = number.rsplit(".", 1)[0]
        parent_path = paths_by_number.get(parent_number)
        return f"{parent_path}{SECTION_PATH_SEPARATOR}{title}" if parent_path else title

    def _section(
        self,
        div: ET.Element,
        tables: list[ET.Element],
        *,
        raw_title: str,
        path: str,
    ) -> ParsedSection | None:
        """Build one section, assembling its text and blocks from one pass.

        Text and offsets are produced together and never re-derived, which is
        what makes `ParsedSection`'s invariant hold by construction: there is no
        second pass that could normalise the text out from under the offsets.
        """
        pieces: list[str] = []
        blocks: list[ParsedBlock] = []
        cursor = 0

        def add(text: str, page: int, kind: BlockKind) -> None:
            nonlocal cursor
            if not text:
                return
            if pieces:
                cursor += len(BLOCK_JOINER)
            pieces.append(text)
            blocks.append(
                ParsedBlock(
                    kind=kind,
                    text=text,
                    page=page,
                    char_start=cursor,
                    char_end=cursor + len(text),
                )
            )
            cursor += len(text)

        for element in div:
            if element.tag.split("}")[-1] == "p":
                for text, page in self._paragraph_runs(element):
                    add(text, page, BlockKind.PARAGRAPH)

        # Tables come from the body, not from inside the div — see _grouped.
        for figure in tables:
            add(
                _table_markdown(figure),
                _first_page(figure.get("coords")) or 1,
                BlockKind.TABLE,
            )

        if not blocks:
            return None

        return ParsedSection(
            kind=normalise_section(raw_title) if raw_title else SectionKind.OTHER,
            path=path,
            blocks=tuple(blocks),
            text=BLOCK_JOINER.join(pieces),
            # The heading as written, not the disambiguated path: `sections`
            # stores both, and a reader view showing "Results (2)" as the
            # heading would report a document that does not exist.
            raw_title=raw_title or UNTITLED_SECTION,
        )

    @staticmethod
    def _paragraph_runs(paragraph: ET.Element) -> list[tuple[str, int]]:
        """Split one TEI paragraph into runs of text sharing a page (D40).

        A paragraph spanning a page break becomes two blocks so each reports the
        page it is actually on. Without this, 6.81% of sentences would report a
        page that is too low.
        """
        sentences = paragraph.findall("tei:s", TEI_NS)
        if not sentences:
            # No sentence segmentation for this paragraph: fall back to the
            # paragraph's own coords, accepting the whole-paragraph page.
            text = _flatten(paragraph)
            page = _first_page(paragraph.get("coords")) or 1
            return [(text, page)] if text else []

        runs: list[tuple[str, int]] = []
        current: list[str] = []
        current_page: int | None = None

        for sentence in sentences:
            text = _flatten(sentence)
            if not text:
                continue
            page = (
                _first_page(sentence.get("coords"))
                or current_page
                or _first_page(paragraph.get("coords"))
                or 1
            )
            if current_page is not None and page != current_page:
                runs.append((" ".join(current), current_page))
                current = []
            current.append(text)
            current_page = page

        if current and current_page is not None:
            runs.append((" ".join(current), current_page))
        return runs
