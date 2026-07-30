"""Structure-aware chunking with parent/child windows (ML-2, doc 6 §6.3).

The highest-leverage decision in a RAG system, and the one retrieval cannot
recover from. Three rules do the work:

1. **Never cross a section boundary.** A chunk spanning Method into Experiments
   blends two different kinds of claim. More importantly, a chunk that straddles
   Related Work loses the signal that its claims belong to *someone else's* paper
   — the misattribution failure doc 2 §2.3 opens with.

2. **Parent/child (small-to-big).** Children (~300 tokens) are embedded and
   searched, because small units match queries precisely. Parents (~1000 tokens)
   are returned as generation context, because large units answer completely.
   A single chunk size has to compromise between the two; decoupling them does not.

3. **Tables are atomic.** Tables hold the facts users most often ask for
   ("what accuracy on ImageNet?") and are exactly what naive extraction mangles.
   A table split mid-row is worse than useless — it is confidently wrong. When a
   table must be split, the header repeats, because a table fragment without its
   header cannot be read.

Deliberately pure: no tokenizer dependency, no I/O. `token_counter` is injected so
production can pass the real tokenizer while tests and the domain stay
framework-free (see the `import-linter` contract in pyproject.toml).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum, unique

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import (
    Chunk,
    ChunkRole,
    Provenance,
    SectionKind,
)

#: Bump on ANY behavioural change here. Stamped onto every chunk (NFR-M5) so a
#: quality shift can be attributed to chunking rather than to embeddings or the
#: reranker — and so re-indexing can target only affected chunks.
CHUNKER_VERSION = "structure-aware/1.0.0"

#: ~4 characters per token for English prose. Good enough for boundary decisions;
#: production injects the real tokenizer for anything that must be exact.
_CHARS_PER_TOKEN = 4


def approximate_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


@unique
class BlockKind(StrEnum):
    """What the parser found."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
    EQUATION = "equation"


@dataclass(frozen=True, slots=True)
class ParsedBlock:
    """A contiguous run of text with a known kind and source position.

    `char_start`/`char_end` are offsets into the parsed document text, which is
    what lets a citation highlight an exact span (FR-I5).
    """

    kind: BlockKind
    text: str
    page: int
    char_start: int
    char_end: int

    def __post_init__(self) -> None:
        if self.char_end <= self.char_start:
            raise InvariantViolation("block char_end must exceed char_start")

    @property
    def is_atomic(self) -> bool:
        """Blocks that must never be split across chunks."""
        return self.kind in (BlockKind.TABLE, BlockKind.EQUATION)


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One section of a parsed document, with its blocks in reading order."""

    kind: SectionKind
    path: str
    blocks: tuple[ParsedBlock, ...]

    @property
    def is_empty(self) -> bool:
        return not self.blocks


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """Tunable boundaries. Every value here is an ablation axis (doc 9 §9.3)."""

    child_target_tokens: int = 300
    parent_target_tokens: int = 1000
    overlap_ratio: float = 0.15
    #: Below this, a trailing fragment is merged backwards instead of standing
    #: alone — a 20-token chunk carries too little context to retrieve well.
    min_child_tokens: int = 80
    #: Back-matter is indexed but rarely useful as evidence; skipping it keeps the
    #: index smaller and precision higher.
    skip_non_body: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.overlap_ratio < 1.0:
            raise InvariantViolation(f"overlap_ratio must be in [0, 1), got {self.overlap_ratio}")
        if self.child_target_tokens >= self.parent_target_tokens:
            raise InvariantViolation(
                "parent windows must be larger than children — that is the point "
                "of small-to-big retrieval"
            )
        if self.min_child_tokens >= self.child_target_tokens:
            raise InvariantViolation("min_child_tokens must be below the child target")


@dataclass(slots=True)
class ChunkingResult:
    """Chunks plus the counters the ingestion pipeline reports."""

    chunks: list[Chunk] = field(default_factory=list)
    sections_processed: int = 0
    sections_skipped: int = 0
    tables_preserved: int = 0

    @property
    def children(self) -> list[Chunk]:
        return [c for c in self.chunks if c.role is ChunkRole.CHILD]

    @property
    def parents(self) -> list[Chunk]:
        return [c for c in self.chunks if c.role is ChunkRole.PARENT]

    @property
    def embeddable(self) -> list[Chunk]:
        """What actually gets embedded: children and tables, not parent windows.

        Embedding parents too would double the index for no retrieval gain —
        parents exist to be *returned*, not matched.
        """
        return [c for c in self.chunks if c.role is not ChunkRole.PARENT]


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence split, good enough for chunk boundaries.

    Not linguistically correct — "et al." and "Fig. 3" will fool it. That is
    acceptable: the cost of a slightly misplaced boundary is small, and the cost
    of a spaCy dependency in the domain layer is a framework import the
    architecture contract forbids.
    """
    # A block with no sentence-ending punctuation still yields one unit.
    parts = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


class StructureAwareChunker:
    """Turns a parsed document into parent windows and child chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: Callable[[str], int] = approximate_tokens,
    ) -> None:
        self._config = config or ChunkingConfig()
        self._count = token_counter

    @property
    def version(self) -> str:
        return CHUNKER_VERSION

    def chunk(self, paper_id: str, sections: Sequence[ParsedSection]) -> ChunkingResult:
        result = ChunkingResult()
        ordinal = 0

        for section in sections:
            if section.is_empty:
                continue
            if self._config.skip_non_body and not section.kind.is_body:
                result.sections_skipped += 1
                continue

            result.sections_processed += 1

            # Rule 1: parent windows are built *within* a section, so no chunk can
            # span a section boundary.
            for window in self._parent_windows(section):
                parent, children, tables = self._materialise(
                    paper_id, section, window, start_ordinal=ordinal
                )
                result.chunks.append(parent)
                result.chunks.extend(children)
                result.chunks.extend(tables)
                result.tables_preserved += len(tables)
                ordinal += 1 + len(children) + len(tables)

        return result

    # -- windowing ---------------------------------------------------------

    def _parent_windows(self, section: ParsedSection) -> Iterator[list[ParsedBlock]]:
        """Group blocks into parent-sized windows without splitting any block."""
        window: list[ParsedBlock] = []
        tokens = 0

        for block in section.blocks:
            block_tokens = self._count(block.text)

            # An atomic block larger than a whole window still gets its own
            # window rather than being broken up.
            if tokens and tokens + block_tokens > self._config.parent_target_tokens:
                yield window
                window, tokens = [], 0

            window.append(block)
            tokens += block_tokens

        if window:
            yield window

    def _materialise(
        self,
        paper_id: str,
        section: ParsedSection,
        window: list[ParsedBlock],
        *,
        start_ordinal: int,
    ) -> tuple[Chunk, list[Chunk], list[Chunk]]:
        """Build the parent chunk, its children, and any atomic table chunks."""
        prose = [b for b in window if not b.is_atomic]
        atomic = [b for b in window if b.is_atomic]

        parent_id = f"{paper_id}:p{start_ordinal}"
        parent_text = "\n\n".join(b.text for b in window)
        parent = Chunk(
            id=parent_id,
            text=parent_text,
            role=ChunkRole.PARENT,
            ordinal=start_ordinal,
            provenance=self._provenance(paper_id, section, window),
            chunker_version=CHUNKER_VERSION,
            token_count=self._count(parent_text),
        )

        children: list[Chunk] = []
        if prose:
            for i, text in enumerate(self._child_texts(prose)):
                children.append(
                    Chunk(
                        id=f"{parent_id}:c{i}",
                        text=text,
                        role=ChunkRole.CHILD,
                        ordinal=start_ordinal + 1 + i,
                        provenance=self._provenance(paper_id, section, prose),
                        chunker_version=CHUNKER_VERSION,
                        token_count=self._count(text),
                        parent_id=parent_id,
                    )
                )

        # Rule 3: tables are their own chunks, never merged into prose children
        # and never split mid-row.
        tables: list[Chunk] = []
        offset = start_ordinal + 1 + len(children)
        for i, block in enumerate(atomic):
            tables.append(
                Chunk(
                    id=f"{parent_id}:t{i}",
                    text=block.text,
                    role=ChunkRole.TABLE,
                    ordinal=offset + i,
                    provenance=Provenance(
                        paper_id=paper_id,
                        section_path=section.path,
                        section_kind=section.kind,
                        page=block.page,
                        char_start=block.char_start,
                        char_end=block.char_end,
                    ),
                    chunker_version=CHUNKER_VERSION,
                    token_count=self._count(block.text),
                )
            )

        return parent, children, tables

    def _child_texts(self, blocks: Sequence[ParsedBlock]) -> list[str]:
        """Split prose into overlapping child-sized units on sentence boundaries."""
        sentences: list[str] = []
        for block in blocks:
            sentences.extend(_split_sentences(block.text))
        if not sentences:
            return []

        target = self._config.child_target_tokens
        overlap_target = int(target * self._config.overlap_ratio)

        chunks: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._count(sentence)

            if current and current_tokens + sentence_tokens > target:
                chunks.append(" ".join(current))
                # Rule: carry a tail of the previous chunk forward so a claim
                # straddling the boundary survives in at least one chunk.
                current = self._overlap_tail(current, overlap_target)
                current_tokens = sum(self._count(s) for s in current)

            current.append(sentence)
            current_tokens += sentence_tokens

        if current:
            tail = " ".join(current)
            # A short trailing fragment is merged backwards rather than left to
            # stand alone with too little context to retrieve well.
            if chunks and self._count(tail) < self._config.min_child_tokens:
                chunks[-1] = f"{chunks[-1]} {tail}"
            else:
                chunks.append(tail)

        return chunks

    def _overlap_tail(self, sentences: list[str], overlap_tokens: int) -> list[str]:
        if overlap_tokens <= 0:
            return []
        tail: list[str] = []
        total = 0
        for sentence in reversed(sentences):
            sentence_tokens = self._count(sentence)
            if total + sentence_tokens > overlap_tokens and tail:
                break
            tail.insert(0, sentence)
            total += sentence_tokens
        return tail

    @staticmethod
    def _provenance(
        paper_id: str, section: ParsedSection, blocks: Sequence[ParsedBlock]
    ) -> Provenance:
        return Provenance(
            paper_id=paper_id,
            section_path=section.path,
            section_kind=section.kind,
            page=min(b.page for b in blocks),
            char_start=min(b.char_start for b in blocks),
            char_end=max(b.char_end for b in blocks),
        )
