"""NFR-R6's rebuild, against fakes (D101).

> "Relational store is source of truth; vector store is a derived index and must
> be rebuildable from scratch by a documented command."

The claim is only worth anything if the rebuilt index is the **same** index. The
test that matters most here is therefore not "did vectors appear" but *what was
embedded*: ingestion embeds `chunk.embedding_input(title, year)` — ~20 tokens of
paper context prepended (ML-3) — and a rebuild from raw `chunk.text` would
produce a plausible, differently-shaped index and silently re-baseline every
measurement in M2. Nothing would fail; the numbers would just move.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import VectorRecord
from scholiast.application.use_cases.rebuild_index import RebuildIndexUseCase
from scholiast.domain.paper import Chunk, ChunkRole, Paper, PaperVisibility, Provenance, SectionKind

PAPER_ID = "55555555-5555-4555-8555-555555555555"
TITLE = "Structure-Aware Chunking for Scientific Retrieval"
YEAR = 2024


def a_paper(paper_id: str = PAPER_ID) -> Paper:
    return Paper(
        id=paper_id,
        content_hash="0" * 64,
        visibility=PaperVisibility.PUBLIC,
        title=TITLE,
        year=YEAR,
    )


def a_chunk(paper_id: str = PAPER_ID, ordinal: int = 0, role: ChunkRole = ChunkRole.CHILD) -> Chunk:
    return Chunk(
        id=f"{paper_id}:{ordinal}",
        text="We propose a chunker that respects section boundaries.",
        role=role,
        ordinal=ordinal,
        provenance=Provenance(
            paper_id=paper_id,
            section_path="Method",
            section_kind=SectionKind.METHOD,
            page=3,
            char_start=0,
            char_end=54,
        ),
        chunker_version="test-chunker/1.0.0",
        token_count=9,
        # The domain requires it: a child is a window into a parent, and one
        # without a parent could not be expanded for generation (§6.3).
        parent_id=f"{paper_id}:parent" if role is ChunkRole.CHILD else None,
    )


class FakePapers:
    def __init__(self, papers: dict[str, Paper]) -> None:
        self.papers = papers

    async def all_paper_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.papers))

    async def get(self, paper_id: str) -> Paper | None:
        return self.papers.get(paper_id)


class FakeChunks:
    def __init__(self, chunks: dict[str, list[Chunk]]) -> None:
        self.chunks = chunks

    async def list_for_paper(self, paper_id: str) -> list[Chunk]:
        return list(self.chunks.get(paper_id, []))


@dataclass
class FakeEmbedder:
    version: str = "test-embedder/1.0.0"
    dimensions: int = 4
    seen: list[str] = field(default_factory=list)
    roles: list[TextRole] = field(default_factory=list)

    async def embed(
        self, texts: Sequence[str], *, role: TextRole, batch_size: int | None = None
    ) -> list[list[float]]:
        self.seen.extend(texts)
        self.roles.append(role)
        return [[float(len(text))] * self.dimensions for text in texts]


@dataclass
class FakeIndex:
    upserted: list[VectorRecord] = field(default_factory=list)
    orphan_calls: list[tuple[str, ...]] = field(default_factory=list)
    orphans: int = 0

    async def upsert(self, records: Sequence[VectorRecord]) -> int:
        self.upserted.extend(records)
        return len(records)

    async def delete_orphans(self, live_paper_ids: Sequence[str]) -> int:
        self.orphan_calls.append(tuple(live_paper_ids))
        return self.orphans


def build(
    *,
    papers: dict[str, Paper] | None = None,
    chunks: dict[str, list[Chunk]] | None = None,
    orphans: int = 0,
) -> tuple[RebuildIndexUseCase, FakeEmbedder, FakeIndex]:
    embedder = FakeEmbedder()
    index = FakeIndex(orphans=orphans)
    use_case = RebuildIndexUseCase(
        papers=FakePapers(papers if papers is not None else {PAPER_ID: a_paper()}),  # type: ignore[arg-type]
        chunks=FakeChunks(chunks if chunks is not None else {PAPER_ID: [a_chunk()]}),  # type: ignore[arg-type]
        embedder=embedder,  # type: ignore[arg-type]
        index=index,  # type: ignore[arg-type]
    )
    return use_case, embedder, index


class TestTheRebuildReproducesTheIndexItReplaces:
    async def test_the_contextual_header_is_embedded_not_the_bare_chunk_text(self) -> None:
        # The silent-re-baseline test. Embedding `chunk.text` would produce a
        # working index whose vectors differ from every one M2 measured, and
        # nothing anywhere would report it.
        use_case, embedder, _ = build()

        await use_case.execute()

        assert len(embedder.seen) == 1
        assert embedder.seen[0] == a_chunk().embedding_input(TITLE, YEAR)
        assert embedder.seen[0] != a_chunk().text

    async def test_the_title_comes_from_the_paper_row(self) -> None:
        # Safe only because ingestion persists GROBID's title onto the row
        # before chunking (`_parsed_metadata`), so the row now holds the same
        # string the original embed used. If it held the upload filename, every
        # rebuilt vector would carry a different header.
        use_case, embedder, _ = build()

        await use_case.execute()

        assert TITLE in embedder.seen[0]

    async def test_chunks_are_embedded_as_passages(self) -> None:
        # bge prefixes queries and passages differently; the wrong role here
        # would shift every vector in the collection.
        use_case, embedder, _ = build()

        await use_case.execute()

        assert embedder.roles == [TextRole.PASSAGE]

    async def test_parent_windows_are_not_indexed(self) -> None:
        """The defect a dry run against the real database exposed.

        Ingestion indexes `ChunkingResult.embeddable` — children and tables,
        never parent windows, because "parents exist to be *returned*, not
        matched". A rebuild that embedded every chunk row would index 111
        vectors where ingestion wrote 67 on this corpus, and a parent window
        would become a hit competing with the child inside it. Nothing would
        fail; every retrieval number in M2 would simply move.

        This is the same claim as the contextual-header test one level up: the
        rebuilt index must be the index it replaced, not merely a working one.
        """
        use_case, embedder, index = build(
            chunks={
                PAPER_ID: [
                    a_chunk(ordinal=0, role=ChunkRole.CHILD),
                    a_chunk(ordinal=1, role=ChunkRole.PARENT),
                    a_chunk(ordinal=2, role=ChunkRole.TABLE),
                ]
            }
        )

        result = await use_case.execute()

        assert len(embedder.seen) == 2, "a parent window reached the embedder"
        assert {r.chunk_id for r in index.upserted} == {f"{PAPER_ID}:0", f"{PAPER_ID}:2"}
        assert result.chunks_embedded == 2

    async def test_a_paper_of_only_parents_contributes_nothing(self) -> None:
        use_case, embedder, index = build(
            chunks={PAPER_ID: [a_chunk(ordinal=0, role=ChunkRole.PARENT)]}
        )

        result = await use_case.execute()

        assert result.papers == 0
        assert embedder.seen == []
        assert index.upserted == []

    async def test_the_records_carry_the_chunk_identity_and_stamps(self) -> None:
        use_case, embedder, index = build()

        await use_case.execute()

        (record,) = index.upserted
        assert record.chunk_id == f"{PAPER_ID}:0"
        assert record.paper_id == PAPER_ID
        assert record.section_kind is SectionKind.METHOD
        assert record.chunker_version == "test-chunker/1.0.0"
        # NFR-M5: stamped with the embedder that actually ran, so a rebuild
        # performed with a different model is detectable rather than silent.
        assert record.embedder_version == embedder.version


class TestTheRebuildIsHonestAboutWhatItFound:
    async def test_it_reports_what_it_wrote(self) -> None:
        use_case, _, _ = build()

        result = await use_case.execute()

        assert result.papers == 1
        assert result.chunks_embedded == 1
        assert result.vectors_upserted == 1

    async def test_orphans_are_removed_against_the_full_paper_set(self) -> None:
        # The live set passed to the index must be every paper, not only the
        # ones that had chunks — otherwise a paper still being ingested would
        # have its vectors deleted by a rebuild.
        second = "66666666-6666-4666-8666-666666666666"
        use_case, _, index = build(
            papers={PAPER_ID: a_paper(), second: a_paper(second)},
            chunks={PAPER_ID: [a_chunk()]},
            orphans=3,
        )

        result = await use_case.execute()

        assert index.orphan_calls == [(PAPER_ID, second)]
        assert result.orphans_deleted == 3

    async def test_a_paper_with_no_chunks_contributes_nothing(self) -> None:
        # Unparsed, failed, or mid-flight. Zero vectors is the correct rebuilt
        # state for it, and it must not count as a rebuilt paper.
        use_case, embedder, index = build(chunks={})

        result = await use_case.execute()

        assert result.papers == 0
        assert embedder.seen == []
        assert index.upserted == []

    async def test_a_paper_deleted_mid_rebuild_is_skipped(self) -> None:
        # The id listing and the row read are not one transaction. Skipping is
        # correct: the orphan pass at the end removes its vectors anyway.
        use_case, _, index = build(papers={}, chunks={PAPER_ID: [a_chunk()]})

        result = await use_case.execute()

        assert result.papers == 0
        assert index.upserted == []
