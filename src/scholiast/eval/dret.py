"""D-RET: the labelled retrieval set (FR-E1, doc 9 §9.2, RSK-2).

Queries with graded relevance judgments over the chunks the index actually holds.
Everything here is derived from the vendored snapshot by pure functions, so the
whole dataset is a deterministic function of (snapshot, chunker version, builder
version) — which is what FR-E4's "the same command reproduces the same numbers"
requires of its *inputs* before it can require it of its outputs.

**Judgments are on indexed chunks, not on paragraphs.** Retrieval returns what is
in the index — children and table chunks (`ChunkingResult.embeddable`), never
parent windows, which exist to be returned as context rather than matched. A
judgment against a paragraph would be judging something the retriever cannot
return. Evidence is therefore resolved to its span in the section, and every
indexed chunk whose span **overlaps** that paragraph is relevant: a paragraph
routinely spans several children, and a child routinely covers several paragraphs.

**Grades are 2 and 0 here; grade 1 is a human pass that has not happened.**
Doc 9 §9.2 specifies 3-point graded judgments — 2 directly answers, 1 relevant
context, 0 not relevant — and says grade 1 is "assigned by the same annotator
(you) during review". QASPER's evidence annotations supply grade 2 only. This
module does not invent grade 1, and the report says how many judgments exist
rather than implying the scale is fully populated. nDCG computed on a 2/0 set is
valid; it is simply less discriminating than the design intends, and saying so is
cheaper than discovering it during an interview.

**The split is by paper and it is verified, not asserted** (doc 9 §9.2). If M7's
reranker fine-tuning and the eval set share *papers*, the reranker sees eval
content during training and every downstream number is invalid — even though no
query is duplicated. `verify_split` is what makes that checkable; the test suite
calls it, and so should CI.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

from scholiast.domain.chunking import (
    CHUNKER_VERSION,
    ChunkingConfig,
    StructureAwareChunker,
)
from scholiast.domain.errors import InvariantViolation
from scholiast.domain.paper import Chunk
from scholiast.eval.qasper import (
    FLOAT_EVIDENCE_PREFIX,
    QASPER_BUILDER_VERSION,
    CorpusPaper,
    build_paper,
    normalise_for_match,
)

#: Bump on any change to how judgments are derived. Part of the dataset identity
#: alongside the chunker and builder versions (NFR-M5).
DRET_VERSION = "d-ret/1.0.0"

#: Share of papers held out for evaluation. The rest are available to M7's
#: fine-tuning without contaminating it. Assigned by hashing the paper id rather
#: than by sampling, so the split is stable under snapshot growth: adding papers
#: never moves an existing one across the boundary and silently invalidates a
#: published number.
EVAL_SPLIT_FRACTION = 0.30

#: Share of the *eval split* that forms D-RET-CORE, the primary benchmark (D69).
#:
#: A fixed threshold on a hash of the query id — not "the lowest N by hash", and not
#: a shuffle. The property being bought is **stability under corpus growth**, the one
#: D68 established for the paper split and D69 requires here: adding papers must
#: never silently change which queries the primary benchmark contains.
#:
#: **The reason is structural, not empirical.** Under a threshold, membership is a
#: function of the query's own id and nothing else, so no addition to the corpus can
#: move an existing query in or out. Under any lowest-N rule, membership is a
#: function of the whole pool, and stability becomes contingent on the arithmetic
#: cancelling — which it usually does, and is not the same thing.
#:
#: Measured over the real corpus grown by 90 papers (26 of them landing in the eval
#: split), evictions from CORE were: **threshold 0**, proportional lowest-N
#: (`round(0.193 * pool)`) **0**, fixed-size lowest-N (`[:50]`) **5 of 50**. An
#: earlier note here claimed lowest-N evicted "12 of 51"; that number came from
#: comparing a *fixed* cutoff against a growing pool, which is a strawman — the fair
#: proportional comparison evicts nothing on this data. The threshold still wins,
#: because "usually cancels" is a property of this particular growth and "cannot
#: move" is a property of the rule.
#:
#: 0.193 is 50/259 — doc 9 §9.2's target of ~50 over the eval pool as it stands
#: today. It is a constant rather than a target recomputed per run, because
#: recomputing it from the pool size is precisely what makes membership depend on
#: the pool again. The realised count therefore drifts as the corpus grows, which is
#: why doc 8 states a band (40-60) rather than a number. Currently **53 queries over
#: 41 papers**.
CORE_SELECTION_THRESHOLD = 0.193


class Grade(IntEnum):
    """Doc 9 §9.2's 3-point scale. `CONTEXT` is declared but not yet populated."""

    NOT_RELEVANT = 0
    CONTEXT = 1
    DIRECTLY_ANSWERS = 2


class Split(IntEnum):
    TRAIN = 0
    EVAL = 1


@dataclass(frozen=True, slots=True)
class Judgment:
    """One (query, chunk) relevance grade."""

    query_id: str
    chunk_id: str
    grade: Grade


@dataclass(frozen=True, slots=True)
class Query:
    """One question, scoped to the paper it was written about."""

    id: str
    text: str
    paper_id: str
    split: Split
    #: Member of D-RET-CORE, the primary benchmark (D69). Only ever true for
    #: `Split.EVAL` queries — CORE is a subset of the held-out split, never a
    #: second sample drawn across it, or it would carry M7's training papers into
    #: the headline number.
    in_core: bool = False


@dataclass(slots=True)
class EvidenceAccounting:
    """Where every evidence string went. Reported, never rounded away.

    The point of counting these separately is that each has a different meaning
    and a different remedy, and a single "match rate" would hide all of them.
    """

    total: int = 0
    #: Matched a paragraph byte-for-byte.
    exact: int = 0
    #: Matched only after whitespace collapse (line-wrapping differences).
    whitespace_normalised: int = 0
    #: `FLOAT SELECTED:` — a table or figure. Absent from this corpus by
    #: construction (D68); the real-PDF subset is where these become reachable.
    float_reference: int = 0
    #: The annotator cited a section heading rather than a paragraph. A
    #: section-level pointer, deliberately NOT promoted to a chunk-level grade 2.
    section_heading: int = 0
    #: Resolved to a paragraph but that paragraph produced no indexed chunk —
    #: back-matter dropped by `skip_non_body`, for instance.
    no_indexed_chunk: int = 0
    #: Should be zero. Anything here is a real defect in the resolution logic.
    unresolved: int = 0

    @property
    def resolved(self) -> int:
        return self.exact + self.whitespace_normalised

    def as_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "exact": self.exact,
            "whitespace_normalised": self.whitespace_normalised,
            "float_reference": self.float_reference,
            "section_heading": self.section_heading,
            "no_indexed_chunk": self.no_indexed_chunk,
            "unresolved": self.unresolved,
        }


@dataclass(slots=True)
class DRet:
    """The assembled dataset plus the accounting that makes it auditable."""

    queries: list[Query] = field(default_factory=list)
    judgments: list[Judgment] = field(default_factory=list)
    chunks: dict[str, Chunk] = field(default_factory=dict)
    papers: dict[str, CorpusPaper] = field(default_factory=dict)
    evidence: EvidenceAccounting = field(default_factory=EvidenceAccounting)
    #: Answerable questions that resolved to no chunk at all, and are therefore
    #: absent from `queries`. A query with an empty qrel set silently depresses
    #: Recall@k for every configuration equally, which looks like a hard dataset
    #: rather than a broken one.
    dropped_queries: int = 0

    @property
    def versions(self) -> dict[str, str]:
        """The identity of this dataset. Any change here changes the numbers."""
        return {
            "dret": DRET_VERSION,
            "builder": QASPER_BUILDER_VERSION,
            "chunker": CHUNKER_VERSION,
        }

    def qrels(self, query_id: str) -> dict[str, Grade]:
        return {j.chunk_id: j.grade for j in self.judgments if j.query_id == query_id}

    @property
    def core(self) -> list[Query]:
        """D-RET-CORE — the primary benchmark (D69).

        Every headline number and statistical claim comes from here, and doc 9
        §9.4's method is calibrated to its n≈50.
        """
        return [q for q in self.queries if q.in_core]

    @property
    def full(self) -> list[Query]:
        """D-RET-FULL — the whole eval split, the secondary robustness pool (D69).

        A superset of `core`, not a disjoint sample: the robustness question is
        "does the CORE result hold at five times the sample", which needs the same
        queries in it. Reported in its own block, never merged into CORE's.
        """
        return [q for q in self.queries if q.split is Split.EVAL]

    @property
    def populated_grades(self) -> frozenset[Grade]:
        """Which grades the judgments actually carry — the fact, measured.

        Derived rather than declared, so it cannot drift from the data. Doc 9 §9.2
        specifies a 3-point scale; QASPER's evidence spans supply grade 2 only, and
        grade 1 arrives when the CORE hand-review happens.
        """
        return frozenset(j.grade for j in self.judgments)

    @property
    def is_fully_graded(self) -> bool:
        return Grade.CONTEXT in self.populated_grades

    @property
    def grade_scale(self) -> str:
        """One line a report can print, derived from the data so it cannot lie.

        D69 requires every report to state the scale rather than imply a full
        3-point one. Making that a computed string rather than a convention is the
        difference between a rule and a hope: a reporter that prints this is
        correct by construction, and one that forgets is visibly missing it.
        """
        if self.is_fully_graded:
            return "graded 3-point (0/1/2)"
        return "binary 2/0 — grade 1 (CONTEXT) unpopulated, pending CORE hand-review (D69)"


def assign_split(paper_id: str, *, fraction: float = EVAL_SPLIT_FRACTION) -> Split:
    """Deterministic, stable, and by paper — the three properties that matter.

    Hashing rather than sampling means the assignment does not depend on how many
    papers exist or what order they arrived in, so re-vendoring a larger snapshot
    leaves every existing paper where it was. A shuffled sample would quietly
    reassign papers and invalidate every number published before it.
    """
    digest = hashlib.sha256(paper_id.encode("utf-8")).digest()
    # First 8 bytes as a fraction of the space; well below float precision limits.
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return Split.EVAL if position < fraction else Split.TRAIN


def assign_core(query_id: str, *, threshold: float = CORE_SELECTION_THRESHOLD) -> bool:
    """Is this query in D-RET-CORE? A pure function of its id (D69).

    **Reads nothing but the id, deliberately.** Selecting on any property of the
    query — how many chunks it is judged against, how much of its evidence
    resolved, which section the evidence sits in — would bias the primary
    benchmark toward queries the pipeline already handles well, and every headline
    number would then be measured on a set chosen to flatter it. The id carries no
    signal about difficulty, which is exactly why it is the only input.

    The `"core:"` prefix keeps this hash independent of `assign_split`'s. Without
    it both functions would consume the same digest of related strings, and CORE
    membership could correlate with split assignment in ways nobody intended and
    nobody would notice.
    """
    digest = hashlib.sha256(f"core:{query_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return position < threshold


def _indexed_by_section(chunks: Iterable[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.provenance.section_path, []).append(chunk)
    return grouped


def _overlapping(chunks: Sequence[Chunk], start: int, end: int) -> list[Chunk]:
    """Indexed chunks whose span intersects [start, end).

    Half-open on both sides: a chunk ending exactly where the paragraph begins
    shares no character with it and is not evidence for it.
    """
    return [c for c in chunks if c.provenance.char_start < end and c.provenance.char_end > start]


def build_dret(
    rows: Sequence[dict[str, Any]],
    *,
    config: ChunkingConfig | None = None,
    eval_fraction: float = EVAL_SPLIT_FRACTION,
    core_threshold: float = CORE_SELECTION_THRESHOLD,
    paper_factory: Callable[..., CorpusPaper] = build_paper,
    normaliser: Callable[[str], str] = normalise_for_match,
) -> DRet:
    """Assemble D-RET from vendored QASPER rows. Pure, offline, deterministic.

    `paper_factory` and `normaliser` exist for one caller: the D68 PDF-subset
    check (D81), which needs the SAME chunker, evidence location, grading, split
    and CORE rule applied to papers parsed from real PDFs instead of QASPER JSON.
    Injecting them here rather than duplicating this loop is what keeps the two
    builds comparable — a reimplementation that drifted would be indistinguishable
    from the parsing difference the check exists to measure.

    **The defaults are the published dataset and must stay that way.** Every
    number phases 2-6 reported was produced by them.
    """
    chunker = StructureAwareChunker(config)
    dataset = DRet()
    seen_query_ids: set[str] = set()

    for row in rows:
        paper = paper_factory(row, normaliser=normaliser)
        dataset.papers[paper.paper_id] = paper
        split = assign_split(paper.paper_id, fraction=eval_fraction)

        result = chunker.chunk(paper.paper_id, paper.document.sections)
        indexed = result.embeddable
        for chunk in indexed:
            dataset.chunks[chunk.id] = chunk
        by_section = _indexed_by_section(indexed)

        qas = row["qas"]
        for question, question_id, answers in zip(
            qas["question"], qas["question_id"], qas["answers"], strict=True
        ):
            graded: dict[str, Grade] = {}
            answerable = False

            for annotator in answers["answer"]:
                if annotator.get("unanswerable"):
                    continue
                answerable = True
                for evidence in annotator.get("evidence") or []:
                    dataset.evidence.total += 1
                    if evidence.strip().startswith(FLOAT_EVIDENCE_PREFIX):
                        dataset.evidence.float_reference += 1
                        continue
                    key = normaliser(evidence)
                    located = paper.paragraph_index.get(key)
                    if located is None:
                        if key in paper.section_headings:
                            dataset.evidence.section_heading += 1
                        else:
                            dataset.evidence.unresolved += 1
                        continue
                    if evidence in {b.text for b in _blocks_of(paper, located[0])}:
                        dataset.evidence.exact += 1
                    else:
                        dataset.evidence.whitespace_normalised += 1

                    section_path, start, end = located
                    hits = _overlapping(by_section.get(section_path, ()), start, end)
                    if not hits:
                        dataset.evidence.no_indexed_chunk += 1
                        continue
                    for chunk in hits:
                        graded[chunk.id] = Grade.DIRECTLY_ANSWERS

            if not answerable:
                # Unanswerable questions belong to D-UNANS (abstention calibration),
                # not to a retrieval set. Not counted as dropped: they were never
                # candidates.
                continue
            if not graded:
                dataset.dropped_queries += 1
                continue
            if question_id in seen_query_ids:
                raise InvariantViolation(
                    f"duplicate QASPER question_id {question_id!r} — query ids must be "
                    "unique or judgments from two questions merge silently"
                )
            seen_query_ids.add(str(question_id))

            dataset.queries.append(
                Query(
                    id=str(question_id),
                    text=str(question),
                    paper_id=paper.paper_id,
                    split=split,
                    # CORE is carved out of EVAL, never across the whole set: the
                    # conjunction is what keeps M7's training papers out of the
                    # primary benchmark.
                    in_core=split is Split.EVAL
                    and assign_core(str(question_id), threshold=core_threshold),
                )
            )
            dataset.judgments.extend(
                Judgment(query_id=str(question_id), chunk_id=cid, grade=grade)
                for cid, grade in sorted(graded.items())
            )

    return dataset


def _blocks_of(paper: CorpusPaper, section_path: str) -> tuple[Any, ...]:
    for section in paper.document.sections:
        if section.path == section_path:
            return section.blocks
    return ()


#: Doc 8 M2 deliverable 1's band for the primary benchmark. A range rather than a
#: number because `CORE_SELECTION_THRESHOLD` is fixed, so the realised count drifts
#: as the corpus grows — that drift is the price of stable membership (D69).
CORE_SIZE_BAND = (40, 60)


def verify_core(dataset: DRet) -> None:
    """Assert D-RET-CORE is a usable primary benchmark. Raises, never warns.

    Three things can go wrong quietly enough to reach a published number:

    1. **CORE drawing from TRAIN.** It would still look like a valid benchmark and
       would silently include papers M7 fine-tunes on.
    2. **CORE drifting out of doc 8's band.** The threshold is fixed while the
       corpus grows, so the count moves on its own. Outside 40-60, doc 9 §9.4's
       calibration — written around n≈50 — no longer describes it.
    3. **A CORE query with no judgments.** It depresses every configuration
       equally, so it reads as a hard benchmark rather than a broken one.
    """
    core = dataset.core
    strays = [q.id for q in core if q.split is not Split.EVAL]
    if strays:
        raise InvariantViolation(
            f"{len(strays)} CORE queries are not in the eval split, e.g. {strays[:3]} — "
            "CORE is carved out of EVAL so the primary benchmark cannot contain a "
            "paper M7 trained on"
        )
    low, high = CORE_SIZE_BAND
    if not low <= len(core) <= high:
        raise InvariantViolation(
            f"D-RET-CORE holds {len(core)} queries, outside doc 8's {low}-{high} band. "
            f"The threshold is fixed and the corpus grew; re-derive "
            f"CORE_SELECTION_THRESHOLD deliberately, record it, and re-baseline — "
            f"every number published against the old CORE describes a different set"
        )
    judged = {j.query_id for j in dataset.judgments}
    unjudged = [q.id for q in core if q.id not in judged]
    if unjudged:
        raise InvariantViolation(
            f"{len(unjudged)} CORE queries carry no judgment, e.g. {unjudged[:3]}"
        )


def verify_split(dataset: DRet) -> None:
    """Assert the train/eval partition leaks nothing. Raises, never warns.

    Doc 9 §9.2 calls this "the first thing a careful interviewer probes about a
    fine-tuning result", and the failure mode is silent: no query is duplicated,
    so a query-level check passes while the reranker has already read the eval
    papers. The only check that catches it is at the paper level.
    """
    train = {q.paper_id for q in dataset.queries if q.split is Split.TRAIN}
    evaluation = {q.paper_id for q in dataset.queries if q.split is Split.EVAL}
    overlap = train & evaluation
    if overlap:
        raise InvariantViolation(
            f"{len(overlap)} paper(s) appear in both splits, e.g. {sorted(overlap)[:3]} — "
            "the split is by paper precisely so this cannot happen"
        )
    if not train or not evaluation:
        raise InvariantViolation(
            f"a split is empty (train={len(train)} papers, eval={len(evaluation)}); "
            "no comparison is possible"
        )
    # A judgment pointing at a chunk from a paper on the other side of the split
    # would leak through the qrels even with the paper sets disjoint.
    by_query = {q.id: q for q in dataset.queries}
    for judgment in dataset.judgments:
        query = by_query.get(judgment.query_id)
        if query is None:
            raise InvariantViolation(f"judgment for unknown query {judgment.query_id!r}")
        chunk = dataset.chunks.get(judgment.chunk_id)
        if chunk is None:
            raise InvariantViolation(
                f"judgment points at chunk {judgment.chunk_id!r}, which is not in the corpus"
            )
        if chunk.provenance.paper_id != query.paper_id:
            raise InvariantViolation(
                f"query {query.id!r} (paper {query.paper_id}) is judged against a chunk "
                f"from paper {chunk.provenance.paper_id} — cross-paper judgment"
            )
