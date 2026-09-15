"""QASPER rows to `ParsedDocument` — D-RET's corpus (D68).

D68 chose QASPER's own `full_text` over re-parsing the PDFs, because QASPER's
evidence strings were extracted from QASPER's parse: matching them against text
drawn from that same parse is a string comparison, while matching them against
GROBID output is a fuzzy one that fails silently on exactly the content with
unusual typography. A biased judgment set is worse than a smaller honest one.

**This builds the same domain objects the GROBID adapter builds**, deliberately:
`ParsedSection` enforces D42/D45's round trip at construction, `unique_section_path`
is imported rather than reimplemented (D51 must not diverge between the two
corpora), and `normalise_section` assigns the same taxonomy. The corpus differs
from production in its *source*, not in its shape — which is the only way the
D68 PDF-subset comparison measures the parser rather than two unrelated pipelines.

**Two limitations this corpus has by construction**, both measured rather than
assumed, both consequences of the D68 tradeoff rather than defects:

- **No pagination.** QASPER's JSON has no page numbers, so every block is page 1.
  Page-level provenance (FR-I5) is therefore *not* exercised by D-RET and is not
  measurable on it. The PDF subset is the only place it is.
- **No tables or figures.** `full_text` holds prose paragraphs only; QASPER keeps
  floats in a separate `figures_and_tables` field with its own caption format.
  Measured: **9.0% of QASPER's evidence strings are `FLOAT SELECTED:` references**
  to exactly that content, so roughly one evidence string in eleven names
  something this corpus cannot contain. Table-heavy questions are consequently
  under-represented in D-RET, which matters because tables are what naive
  extraction mangles (chunking rule 3) and what users most often ask about.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from scholiast.application.ports.parsing import ParsedDocument
from scholiast.domain.chunking import (
    UNTITLED_SECTION,
    BlockKind,
    ParsedBlock,
    ParsedSection,
    unique_section_path,
)
from scholiast.domain.paper import SectionKind, normalise_section

#: Stamped onto the corpus exactly as a real parser version is (NFR-M5), because
#: a change to how QASPER JSON becomes sections changes every chunk id and every
#: judgment that points at one. Bump on ANY behavioural change here.
# 1.0.0: initial — paragraphs become blocks, abstract becomes its own section.
QASPER_BUILDER_VERSION = "qasper-json/1.0.0"

#: What joins two paragraphs in a section's canonical text. Two newlines, matching
#: the GROBID adapter, so child-splitting sees the same paragraph boundaries.
_PARAGRAPH_SEPARATOR = "\n\n"

#: QASPER's marker for evidence drawn from a table or figure rather than prose.
FLOAT_EVIDENCE_PREFIX = "FLOAT SELECTED"

_WHITESPACE = re.compile(r"\s+")


def normalise_for_match(text: str) -> str:
    """Collapse whitespace for evidence lookup — matching only, never storage.

    Measured over the vendored snapshot: of 2,808 evidence strings, 2,349 match a
    paragraph byte-for-byte and a further 77 match only after this collapse —
    line-wrapping differences between QASPER's annotation tool and its extracted
    text. **Zero** evidence strings match as a mere substring of a paragraph, and
    **zero** are absent from the corpus altogether, so this normalisation resolves
    uniquely and adds no ambiguity: it is exact matching modulo whitespace, not
    fuzzy matching.

    It must never touch the text that is *stored*. Offsets index the canonical
    section text (D42), and `ParsedBlock` asserts `text[start:end] == block.text`
    at construction — normalising before recording offsets is the precise failure
    that invariant exists to catch.
    """
    return _WHITESPACE.sub(" ", text).strip()


@dataclass(frozen=True, slots=True)
class CorpusPaper:
    """One QASPER paper as the pipeline sees it, plus what judgments need.

    `paragraph_index` maps normalised paragraph text to `(section_path, offsets)`
    so an evidence string can be resolved to a span without re-walking the
    document. Built here rather than in the judgment code because it is derived
    from the same pass that assigns the offsets, and deriving it twice is how the
    two drift apart.
    """

    paper_id: str
    title: str
    document: ParsedDocument
    #: normalised paragraph text -> (section path, char_start, char_end)
    paragraph_index: dict[str, tuple[str, int, int]]
    #: Section headings, normalised. QASPER annotators sometimes cite a heading
    #: instead of a paragraph (measured: 129 of 2,808 evidence strings); those are
    #: section-level pointers, not paragraph evidence, and are reported rather than
    #: silently promoted to a grade.
    section_headings: frozenset[str]


def _section_text_and_blocks(paragraphs: list[str]) -> tuple[str, tuple[ParsedBlock, ...]]:
    """Assemble canonical section text and the blocks indexing it.

    Blank paragraphs are dropped rather than carried: `ParsedBlock` requires
    `char_end > char_start`, and a zero-length block is not a thing the chunker
    has any use for. Measured: 328 of 13,594 paragraphs in the snapshot are blank.
    """
    kept = [p for p in paragraphs if p.strip()]
    text = _PARAGRAPH_SEPARATOR.join(kept)
    blocks: list[ParsedBlock] = []
    cursor = 0
    for paragraph in kept:
        start = cursor
        end = start + len(paragraph)
        blocks.append(
            ParsedBlock(
                kind=BlockKind.PARAGRAPH,
                text=paragraph,
                # QASPER JSON carries no pagination; see the module docstring.
                page=1,
                char_start=start,
                char_end=end,
            )
        )
        cursor = end + len(_PARAGRAPH_SEPARATOR)
    return text, tuple(blocks)


def build_paper(row: dict[str, Any]) -> CorpusPaper:
    """One vendored QASPER row into the domain's parsed-document shape."""
    paper_id = str(row["id"])
    sections: list[ParsedSection] = []
    index: dict[str, tuple[str, int, int]] = {}
    headings: set[str] = set()
    claimed: set[str] = set()

    abstract = (row.get("abstract") or "").strip()
    if abstract:
        text, blocks = _section_text_and_blocks([abstract])
        path = unique_section_path("Abstract", claimed)
        claimed.add(path)
        sections.append(
            ParsedSection(
                kind=SectionKind.ABSTRACT,
                path=path,
                blocks=blocks,
                text=text,
                raw_title="Abstract",
            )
        )
        for block in blocks:
            index.setdefault(
                normalise_for_match(block.text), (path, block.char_start, block.char_end)
            )

    full_text = row["full_text"]
    for raw_name, paragraphs in zip(
        full_text["section_name"], full_text["paragraphs"], strict=True
    ):
        raw_title = (raw_name or "").strip()
        if raw_title:
            headings.add(normalise_for_match(raw_title))
        text, blocks = _section_text_and_blocks(list(paragraphs))
        if not blocks:
            # An empty section carries nothing to retrieve. Skipped rather than
            # constructed, because `ParsedSection` would accept it and the chunker
            # would then count a section it produced no chunks for.
            continue
        path = unique_section_path(raw_title or UNTITLED_SECTION, claimed)
        claimed.add(path)
        sections.append(
            ParsedSection(
                kind=normalise_section(raw_title) if raw_title else SectionKind.OTHER,
                path=path,
                blocks=blocks,
                text=text,
                raw_title=raw_title or UNTITLED_SECTION,
            )
        )
        for block in blocks:
            # setdefault, not assignment: a paragraph repeated verbatim in two
            # sections resolves to its first occurrence, deterministically. Picking
            # the last would make the mapping depend on iteration order.
            index.setdefault(
                normalise_for_match(block.text), (path, block.char_start, block.char_end)
            )

    document = ParsedDocument(
        sections=tuple(sections),
        title=str(row.get("title") or ""),
        abstract=abstract or None,
        page_count=0,
        parser_version=QASPER_BUILDER_VERSION,
        stats={"sections": len(sections), "blocks": sum(len(s.blocks) for s in sections)},
    )
    return CorpusPaper(
        paper_id=paper_id,
        title=str(row.get("title") or ""),
        document=document,
        paragraph_index=index,
        section_headings=frozenset(headings),
    )
