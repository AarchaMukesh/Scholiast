"""D68's validity check: the same papers, parsed from real PDFs (D81).

D-RET is built from QASPER's `full_text` JSON, which makes evidence matching an
exact string test — that is why D68 chose it. The cost is that the chunker's input
is QASPER's paragraph structure and never GROBID's section bodies, so every
retrieval number this project publishes describes a pipeline that **diverges from
production at the parse step**. D19 exists because those two differ.

This module turns that caveat into a measurement: take a sample of the same
papers, fetch the real PDFs, run them through the production parser and the same
chunker, and report the Recall@k delta.

**Only the parse varies.** The PDF side goes through `build_dret` itself, with a
paper factory swapped in — same chunker, same evidence location, same grading,
same split and CORE rules. A reimplementation of that loop would make drift
between the two builds indistinguishable from a parsing difference, which is the
one confusion this check exists to remove.

**Both sides are matched by one normaliser.** GROBID text differs from QASPER's by
whitespace, hyphenation at line breaks and ligatures, so the PDF side needs a more
permissive matcher than the published `normalise_for_match`; giving it one the
JSON side lacks would let the sides differ in *matching* as well as parsing. The
shared normaliser lives here and is used for this check only — the vendored
dataset keeps its published behaviour, because changing it would move every number
phases 2-6 reported.
"""

from __future__ import annotations

import asyncio
import random
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from scholiast.application.ports.parsing import ParsedDocument
from scholiast.eval.dret import DRet, Split
from scholiast.eval.qasper import FLOAT_EVIDENCE_PREFIX, CorpusPaper

#: arXiv serves QASPER's paper ids directly — they ARE arXiv ids (e.g. 1912.01214).
ARXIV_PDF_URL = "https://arxiv.org/pdf/{paper_id}"

#: One request per paper, spaced. arXiv asks automated clients to identify
#: themselves and to go slowly; ten PDFs is a small ask made politely.
FETCH_DELAY_SECONDS = 3.0
USER_AGENT = "Scholiast/0.1 (retrieval-evaluation validity check; 10 papers, one pass)"

#: D81's sample. Seeded rather than chosen: papers that parse cleanly are exactly
#: the papers a hand-picked sample would drift towards.
SAMPLE_SIZE = 10
SAMPLE_SEED = 20260924

_WHITESPACE = re.compile(r"\s+")
#: A hyphen followed by a line break — PDF line-wrapping, not a real hyphen.
#: `state-of-the-art` keeps its hyphens because there is no whitespace after them.
#: HYPHEN-MINUS, HYPHEN, NON-BREAKING HYPHEN, FIGURE DASH, EN DASH. Built from
#: code points rather than typed literally: these are indistinguishable on sight
#: in source, and a reader cannot verify a character class they cannot see.
#: HYPHEN-MINUS leads, where a leading `-` is literal rather than a range.
_HYPHENS = "".join(chr(code) for code in (0x2D, 0x2010, 0x2011, 0x2012, 0x2013))
_HYPHEN_BREAK = re.compile(rf"(\w)[{_HYPHENS}]\s+(\w)")


def normalise_for_pdf_match(text: str) -> str:
    """Whitespace, hyphenation and ligatures folded away — matching only (D81).

    Applied to BOTH sides of the check so the two builds differ in parsing alone.
    Deliberately more permissive than the published `normalise_for_match`, which
    can afford to be strict because QASPER's evidence came from QASPER's own
    parse; GROBID's text did not, and D68 names these three differences as the
    reason some evidence would otherwise fail to map.

    Never touches stored text. Offsets index the canonical section text (D42), and
    `ParsedBlock` asserts the round trip at construction.
    """
    folded = unicodedata.normalize("NFKC", text)
    dehyphenated = _HYPHEN_BREAK.sub(r"\1\2", folded)
    return _WHITESPACE.sub(" ", dehyphenated).strip().casefold()


def sample_papers(dataset: DRet, *, count: int = SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> list[str]:
    """Pick the papers to re-parse: eval-split, carrying queries, chosen by seed.

    Eval-split because those are the papers the published numbers are measured on.
    Seeded and sorted because the sample is part of the result: a reader has to be
    able to reproduce it, and a hand-picked ten would drift toward papers that
    parse cleanly.
    """
    eligible = sorted({q.paper_id for q in dataset.queries if q.split is Split.EVAL})
    if not eligible:
        raise ValueError("no eval-split papers to sample from")
    rng = random.Random(seed)  # noqa: S311 - reproducibility, not secrecy
    return sorted(rng.sample(eligible, min(count, len(eligible))))


def paper_from_document(
    row: dict[str, Any], document: ParsedDocument, *, normaliser: Any
) -> CorpusPaper:
    """A `CorpusPaper` whose text came from GROBID instead of QASPER's JSON.

    The paragraph index is rebuilt from the parsed blocks exactly as the JSON
    builder does it — same `setdefault` rule, so a paragraph repeated in two
    sections resolves to its first occurrence — because that index is what the
    evidence lookup reads. Questions and answers still come from the QASPER row:
    the check varies where the TEXT comes from, never the labels.
    """
    index: dict[str, tuple[str, int, int]] = {}
    headings: set[str] = set()
    for section in document.sections:
        if section.raw_title:
            headings.add(normaliser(section.raw_title))
        for block in section.blocks:
            index.setdefault(
                normaliser(block.text), (section.path, block.char_start, block.char_end)
            )
    return CorpusPaper(
        paper_id=str(row["id"]),
        title=str(row.get("title") or ""),
        document=document,
        paragraph_index=index,
        section_headings=frozenset(headings),
    )


class ParserLike(Protocol):
    """The parsing surface this check needs — the production GROBID adapter."""

    async def parse(self, pdf: bytes, *, filename: str = "") -> ParsedDocument: ...

    @property
    def version(self) -> str: ...


class ClientLike(Protocol):
    """The HTTP surface this check needs, spelled out so `httpx` satisfies it."""

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        follow_redirects: bool,
        # ASYNC109 prefers `asyncio.timeout` over a timeout parameter. This one is
        # not ours to rename: the protocol exists to describe httpx's own `get`.
        timeout: float,  # noqa: ASYNC109
    ) -> Any: ...


def cached_pdf(paper_id: str, cache_dir: Path) -> bytes | None:
    """The cached bytes for a paper, or None. Synchronous, and deliberately so.

    The filesystem work lives in these two helpers rather than in the coroutine
    below: blocking calls inside an async function are a real smell even when, as
    here, nothing else is in flight — and keeping them out means the rule can stay
    on for the code where it matters.
    """
    destination = cache_dir / f"{paper_id}.pdf"
    if destination.exists() and destination.stat().st_size > 0:
        return destination.read_bytes()
    return None


def store_pdf(paper_id: str, cache_dir: Path, content: bytes) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{paper_id}.pdf").write_bytes(content)


async def fetch_pdf(paper_id: str, *, cache_dir: Path, client: ClientLike) -> bytes:
    """Fetch one PDF, or reuse the cached copy.

    Cached because a re-run should not re-ask arXiv for bytes it already has, and
    because the check is worth repeating; the cache lives under `var/` and is
    gitignored, so the repository never carries third-party PDFs.
    """
    cached = cached_pdf(paper_id, cache_dir)
    if cached is not None:
        return cached
    response = await client.get(
        ARXIV_PDF_URL.format(paper_id=paper_id),
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
        timeout=60.0,
    )
    response.raise_for_status()
    content: bytes = response.content
    if not content.startswith(b"%PDF"):
        # arXiv serves an HTML holding page for withdrawn or unavailable papers;
        # handing that to GROBID produces an empty parse that would read as a
        # catastrophic chunking regression rather than as the fetch failure it is.
        raise ValueError(
            f"{paper_id}: arXiv returned {len(content)} bytes that are not a PDF — "
            "probably a holding page rather than the paper"
        )
    store_pdf(paper_id, cache_dir, content)
    return content


async def fetch_and_parse(
    paper_ids: Sequence[str],
    *,
    cache_dir: Path,
    client: ClientLike,
    parser: ParserLike,
    delay_seconds: float = FETCH_DELAY_SECONDS,
) -> dict[str, ParsedDocument]:
    """Fetch each paper and parse it through the production path. Sequential.

    Sequential and spaced on purpose: ten PDFs is a small favour to ask of arXiv
    and a rude one to ask ten times at once. The delay is skipped for papers
    already cached, so re-runs cost nothing.
    """
    documents: dict[str, ParsedDocument] = {}
    for position, paper_id in enumerate(paper_ids):
        already_have = cached_pdf(paper_id, cache_dir) is not None
        if position and not already_have:
            await asyncio.sleep(delay_seconds)
        pdf = await fetch_pdf(paper_id, cache_dir=cache_dir, client=client)
        documents[paper_id] = await parser.parse(pdf, filename=f"{paper_id}.pdf")
    return documents


#: How much of an evidence string must appear before "the text is partly there" is
#: a better description than "the text is missing". 60 characters is roughly a
#: sentence opening — long enough that a coincidental match is implausible.
HEAD_CHARS = 60


@dataclass(frozen=True, slots=True)
class MappingDiagnosis:
    """WHY evidence failed to map, which is the whole value of a failure count.

    A bare "84% unresolved" is uninterpretable: it could mean the parse lost the
    text, or that the same text is split into different paragraphs, or that the
    two sources render the same sentence differently. Those have different
    meanings for the corpus and different remedies, so they are counted apart.
    """

    whole_block: int = 0
    boundaries_differ: int = 0
    tail_differs: int = 0
    absent: int = 0
    float_reference: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "resolved_whole_block": self.whole_block,
            "unresolved_boundaries_differ": self.boundaries_differ,
            "unresolved_tail_differs": self.tail_differs,
            "unresolved_absent_from_parse": self.absent,
            "float_reference": self.float_reference,
        }


def diagnose_mapping(
    rows: Sequence[dict[str, Any]],
    papers: Mapping[str, CorpusPaper],
    *,
    normaliser: Any = normalise_for_pdf_match,
    head_chars: int = HEAD_CHARS,
) -> MappingDiagnosis:
    """Classify every evidence string by how it failed to reach a chunk.

    - **whole_block** — the lookup the builder actually performs succeeded.
    - **boundaries_differ** — the text is in the parse, but not as one paragraph.
      A boundary-insensitive matcher would recover these; the corpus has them.
    - **tail_differs** — the opening is there and the rest is not. This is the
      bucket that says the two sources *render the same sentence differently*
      rather than that one of them lost it: QASPER's text comes from arXiv LaTeX
      and carries raw citation keys like `lang2011unsupervised`, while GROBID
      reads the typeset PDF, where the same span reads "(Lang et al., 2011)".
      No normaliser bridges that, because it is a content difference.
    - **absent** — not in the parse at all: a dropped section, a float, back
      matter the parser declined to emit.
    """
    counts = dict.fromkeys(
        ("whole_block", "boundaries_differ", "tail_differs", "absent", "float_reference"), 0
    )
    for row in rows:
        paper = papers.get(str(row["id"]))
        if paper is None:
            continue
        haystack = normaliser(" ".join(section.text for section in paper.document.sections))
        for answers in row["qas"]["answers"]:
            for annotator in answers["answer"]:
                if annotator.get("unanswerable"):
                    continue
                for evidence in annotator.get("evidence") or []:
                    if evidence.strip().startswith(FLOAT_EVIDENCE_PREFIX):
                        counts["float_reference"] += 1
                        continue
                    key = normaliser(evidence)
                    if key in paper.paragraph_index:
                        counts["whole_block"] += 1
                    elif key and key in haystack:
                        counts["boundaries_differ"] += 1
                    elif key[:head_chars] and key[:head_chars] in haystack:
                        counts["tail_differs"] += 1
                    else:
                        counts["absent"] += 1
    return MappingDiagnosis(**counts)


def judged_query_ids(dataset: DRet) -> set[str]:
    """Queries this build actually carries judgments for."""
    return {judgment.query_id for judgment in dataset.judgments}


@dataclass(frozen=True, slots=True)
class Comparable:
    """The queries both builds can be scored on, and what each side lost.

    D81's basis: a query whose evidence mapped on one side and not the other
    measures the MATCHER, not retrieval, so the headline delta is computed over
    the intersection — and the asymmetry is reported beside it rather than
    absorbed into the number.
    """

    shared: tuple[str, ...]
    json_only: tuple[str, ...]
    pdf_only: tuple[str, ...]

    @property
    def report(self) -> dict[str, int]:
        return {
            "scored_on_both": len(self.shared),
            "judged_only_on_json": len(self.json_only),
            "judged_only_on_pdf": len(self.pdf_only),
        }


def comparable_queries(json_side: DRet, pdf_side: DRet) -> Comparable:
    """Split the query sets into what can be compared and what cannot."""
    from_json = judged_query_ids(json_side)
    from_pdf = judged_query_ids(pdf_side)
    return Comparable(
        shared=tuple(sorted(from_json & from_pdf)),
        json_only=tuple(sorted(from_json - from_pdf)),
        pdf_only=tuple(sorted(from_pdf - from_json)),
    )


def queries_by_id(dataset: DRet, ids: Sequence[str]) -> list[Any]:
    """The dataset's own query objects for `ids`, in the given order.

    Each side is scored against ITS OWN judgments — the chunk ids differ between
    builds — so the query objects must come from the matching dataset. Order is
    fixed so the paired bootstrap lines the two runs up query for query.
    """
    catalogue = {query.id: query for query in dataset.queries}
    missing = [qid for qid in ids if qid not in catalogue]
    if missing:
        raise ValueError(f"queries absent from this build: {missing[:3]}")
    return [catalogue[qid] for qid in ids]
