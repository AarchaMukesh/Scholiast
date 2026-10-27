"""Which questions are answerable, which are not, and what is asked where (D177).

M5 phase 1 fits the abstention threshold tau on labelled questions, and D177 fixed
how they are labelled before anything was scored. This module is that labelling,
and nothing else: it scores no signal and calls no model.

**The labels are QASPER's, never ours.** A question is answerable when every
annotator answered it, unanswerable when every annotator marked it unanswerable,
and left out when they disagree. Inferring a label would fit tau to a judgment
this project made, which is the circularity a calibration exists to avoid.

**A second unanswerable category, because nine cannot carry a curve.** QASPER
leaves only 9 unanimously unanswerable questions on D-RET's eval split. Doc 9's
other kind of unanswerable question - one about a paper the workspace does not
hold - is built here: an answerable question asked in a scope that omits its own
paper. Such a question is about its own, absent paper, so another paper's answer
would be about the wrong work (D177's clarification); it is excluded only if it
is general knowledge or names a subject a scope paper covers. Each candidate is
hand-checked against that rule before any signal is scored, and this module only
draws the candidates and writes the sheet the check is done on.

**Every draw is a hash, keyed so it cannot collide with another.** The same three
properties D168's `scope_for` has: independent of corpus order and size, keyed by
question rather than by paper, and reproducible from the ids alone. The keys are
D177's own, so no draw here can coincide with the gate's.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from scholiast.eval.dret import DRet, Split

#: The sets' version (NFR-M5): a changed rule or a changed label is a new version.
SET_VERSION = "d-ans-unans/1.0.0"
#: D177's hash keys. Fixed: a different key draws different candidates and scopes,
#: and the hand-check would then have labelled questions nobody asks.
CANDIDATE_KEY = "d177-candidate"
WITHHELD_KEY = "d177-withheld"
#: The withheld scope's size: D168's deciding size, so a withheld question is asked
#: in a workspace as large as the answerable ones are.
WITHHELD_SCOPE_SIZE = 10
#: How many answerable questions per split become withheld-paper candidates.
CANDIDATES: Mapping[Split, int] = {Split.TRAIN: 60, Split.EVAL: 40}


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    #: Annotators disagree: excluded, counted, never used.
    DISAGREE = "disagree"


def answerability(annotations: Sequence[Mapping[str, Any]]) -> Answerability:
    """QASPER's verdict on one question, from its annotators' `unanswerable` flags."""
    if not annotations:
        raise ValueError("a question with no annotations has no label")
    flags = [bool(annotation["unanswerable"]) for annotation in annotations]
    if all(flags):
        return Answerability.UNANSWERABLE
    if not any(flags):
        return Answerability.ANSWERABLE
    return Answerability.DISAGREE


@dataclass(frozen=True, slots=True)
class LabelledQuestion:
    question_id: str
    paper_id: str
    split: Split
    text: str
    label: Answerability


@dataclass(slots=True)
class LabelledSet:
    """Every labelled question, and what was left out and why."""

    questions: list[LabelledQuestion] = field(default_factory=list)
    #: Questions on papers with no D-RET query: outside D-RET's split, excluded (D177).
    outside_dret: int = 0

    def counts(self) -> dict[str, dict[str, int]]:
        """Split name -> label -> count, the table D177 pre-registered."""
        table: dict[str, dict[str, int]] = {
            split.name.lower(): {label.value: 0 for label in Answerability} for split in Split
        }
        for question in self.questions:
            table[question.split.name.lower()][question.label.value] += 1
        return table

    def of(self, split: Split, label: Answerability) -> list[LabelledQuestion]:
        return [q for q in self.questions if q.split is split and q.label is label]


def label_questions(rows: Sequence[Mapping[str, Any]], dataset: DRet) -> LabelledSet:
    """Label every question in the snapshot, each on its paper's D-RET split.

    The split is read from D-RET's own queries rather than recomputed, so a question
    can never land on a different side of the train/eval line than its paper's
    retrieval queries did. A paper with no D-RET query has no split to read, and its
    questions are excluded, as D177 fixed.
    """
    split_of = {query.paper_id: query.split for query in dataset.queries}
    labelled = LabelledSet()
    for row in rows:
        qas = row["qas"]
        for text, question_id, answers in zip(
            qas["question"], qas["question_id"], qas["answers"], strict=True
        ):
            split = split_of.get(row["id"])
            if split is None:
                labelled.outside_dret += 1
                continue
            labelled.questions.append(
                LabelledQuestion(
                    question_id=question_id,
                    paper_id=row["id"],
                    split=split,
                    text=text,
                    label=answerability(answers["answer"]),
                )
            )
    return labelled


def _digest(*parts: str) -> bytes:
    return hashlib.sha256(":".join(parts).encode("utf-8")).digest()


def withheld_candidates(labelled: LabelledSet) -> list[LabelledQuestion]:
    """D177's candidates: per split, the first answerable questions by hash.

    Ordered by `sha256("d177-candidate:{question_id}")`, which reads nothing but
    the id - the same reason `assign_core` does: selecting on anything about the
    question would choose candidates by how hard they are.
    """
    chosen: list[LabelledQuestion] = []
    for split, count in CANDIDATES.items():
        pool = labelled.of(split, Answerability.ANSWERABLE)
        if len(pool) < count:
            raise ValueError(f"{split.name} has {len(pool)} answerable questions, not {count}")
        chosen.extend(sorted(pool, key=lambda q: _digest(CANDIDATE_KEY, q.question_id))[:count])
    return chosen


def withheld_scope(
    question_id: str, own_paper: str, papers: Iterable[str], *, size: int = WITHHELD_SCOPE_SIZE
) -> tuple[str, ...]:
    """The papers a withheld-paper question is asked against: `size` others, never its own.

    Ranked by `sha256("d177-withheld:{question_id}:{paper}")` and returned sorted,
    as `scope_for` does. Its own paper is required to be in the pool and is then
    left out: a scope drawn from a corpus that never held the paper would look
    withheld while proving nothing.
    """
    pool = set(papers)
    if own_paper not in pool:
        raise ValueError(f"question {question_id!r} belongs to {own_paper!r}, not in the corpus")
    others = sorted(pool - {own_paper})
    if not 1 <= size <= len(others):
        raise ValueError(f"a withheld scope of {size} papers cannot be drawn from {len(others)}")
    ranked = sorted(others, key=lambda paper: _digest(WITHHELD_KEY, question_id, paper))
    return tuple(sorted(ranked[:size]))


@dataclass(frozen=True, slots=True)
class ScopePaper:
    paper_id: str
    title: str
    abstract: str


@dataclass(frozen=True, slots=True)
class SheetEntry:
    """One candidate as the hand-check sees it: the question and the scope, not its paper."""

    question_id: str
    split: Split
    question: str
    scope: tuple[ScopePaper, ...]


def candidate_sheet(rows: Sequence[Mapping[str, Any]], dataset: DRet) -> list[SheetEntry]:
    """Every withheld-paper candidate with its scope's titles and abstracts.

    The candidate's own paper is not shown: the check asks whether any paper in
    the scope could answer the question, and the own paper's text would only
    invite comparing against the answer it gives.
    """
    by_id = {row["id"]: row for row in rows}
    papers = sorted(dataset.papers)
    entries: list[SheetEntry] = []
    for candidate in withheld_candidates(label_questions(rows, dataset)):
        scope = withheld_scope(candidate.question_id, candidate.paper_id, papers)
        entries.append(
            SheetEntry(
                question_id=candidate.question_id,
                split=candidate.split,
                question=candidate.text,
                scope=tuple(
                    ScopePaper(
                        paper_id=paper,
                        title=by_id[paper]["title"],
                        abstract=by_id[paper]["abstract"],
                    )
                    for paper in scope
                ),
            )
        )
    return entries


def sheet_json(entries: Sequence[SheetEntry]) -> str:
    """The sheet as JSON, with a sha256 over its entries that a label file must quote.

    The hash binds labels to the exact questions and scopes they judged: a label
    file quoting another sheet's hash is refused when the sets are built, rather
    than silently applied to candidates nobody looked at.
    """
    body = [
        {
            "question_id": entry.question_id,
            "split": entry.split.name.lower(),
            "question": entry.question,
            "scope": [
                {"paper_id": p.paper_id, "title": p.title, "abstract": p.abstract}
                for p in entry.scope
            ],
        }
        for entry in entries
    ]
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()
    return json.dumps(
        {"set_version": SET_VERSION, "sheet_sha256": digest, "candidates": body},
        indent=2,
        sort_keys=True,
    )


#: The verdicts a label may carry (D177's clarification). `exclude (ii)` names the
#: scope paper in its note.
VERDICTS: tuple[str, ...] = ("keep", "exclude (i)", "exclude (ii)")

#: How a candidate is judged, as D177's clarification fixed it before any labelling.
#: The sheet prints these lines verbatim, so the rule a labeller reads is the rule
#: the decisions log records.
INSTRUCTIONS: tuple[str, ...] = (
    "Each question is about its own paper, which is NOT in the scope and not shown -",
    'explicitly ("they", "their", "the authors") or implicitly ("the dataset", "the new',
    """approach"). Asked in this scope it is unanswerable: another paper's data or results""",
    "would be an answer about the wrong work. So the verdict is `keep`, unless:",
    "",
    "- `exclude (i)`: it is not about any particular work at all - a general-knowledge",
    '  question ("What is BERT?", "What does F1 measure?").',
    "- `exclude (ii)`: it names its subject (a model, dataset, system or method) and a",
    "  paper below introduces or reports on that named subject, so it could be answered",
    "  about the same thing it asks about. Name that paper in the note.",
    "",
    "Judge from the titles and abstracts. When unsure whether (ii) holds, exclude: that",
    "keeps the unanswerable class clean. Do not exclude a question for being generic -",
    "a generic question about an absent paper is exactly what this category is for.",
)


def sheet_markdown(entries: Sequence[SheetEntry], *, sheet_sha256: str) -> str:
    """The sheet as a person reads it: one candidate per section, a verdict to fill in."""
    lines = [
        "# D177 withheld-paper candidates",
        "",
        f"Sheet sha256 `{sheet_sha256}` - {len(entries)} candidates, set {SET_VERSION}.",
        "",
        *INSTRUCTIONS,
        "",
    ]
    for number, entry in enumerate(entries, start=1):
        lines += [
            f"## {number}. `{entry.question_id}` ({entry.split.name.lower()})",
            "",
            f"**Question:** {entry.question}",
            "",
            f"**Verdict:** {' / '.join(VERDICTS)}  -  **Note:**",
            "",
        ]
        for paper in entry.scope:
            abstract = " ".join(paper.abstract.split())
            lines.append(f"- **{paper.title}** (`{paper.paper_id}`): {abstract}")
        lines.append("")
    return "\n".join(lines)


# --- step 2: the owner's verdicts, and the versioned sets ---------------------------

#: Where the sets live once built: plain JSON beside its manifest, like the dedup
#: fixture, so a change to one label is a readable diff.
ROOT = Path(__file__).resolve().parents[3]
SETS = ROOT / "data" / "eval" / "d-ans-unans.json"
SETS_MANIFEST = ROOT / "data" / "eval" / "d-ans-unans.manifest.json"
#: The verdict file's columns, as `eval --withheld-sheet`'s template wrote them.
VERDICT_COLUMNS: tuple[str, ...] = (
    "number",
    "question_id",
    "split",
    "question",
    "verdict",
    "note",
    "sheet_sha256",
)
_ARXIV_ID = re.compile(r"\d{4}\.\d{4,5}")


class Category(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE_QASPER = "unanswerable-qasper"
    UNANSWERABLE_WITHHELD = "unanswerable-withheld"


class VerdictFileInvalid(ValueError):
    """The verdict file does not match the sheet it claims to judge. Every problem is listed."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"{len(problems)} problem(s): " + "; ".join(problems[:10]))


@dataclass(frozen=True, slots=True)
class Verdict:
    question_id: str
    verdict: str
    note: str


def read_verdicts(text: str, sheet: Mapping[str, Any]) -> list[Verdict]:
    """The labeller's verdicts, refused unless every row matches the sheet exactly.

    Nothing is normalised: a verdict that is not exactly one of `VERDICTS`, a row out
    of the sheet's order, a question whose text was edited, or a hash from another
    sheet is a problem to report, not something to repair - a repaired label is one
    nobody made. `exclude (ii)` must name a paper from that candidate's own scope
    and no other (D177's clarification). All problems are collected before raising,
    so one pass shows everything that needs the labeller's attention.
    """
    candidates = sheet["candidates"]
    digest = sheet["sheet_sha256"]
    recomputed = hashlib.sha256(json.dumps(candidates, sort_keys=True).encode("utf-8")).hexdigest()
    if recomputed != digest:
        raise VerdictFileInvalid(["the sheet's stated sha256 does not match its own candidates"])
    reader = csv.reader(io.StringIO(text.removeprefix("\ufeff"), newline=""))
    header = next(reader, None)
    problems: list[str] = []
    if header is None or tuple(header) != VERDICT_COLUMNS:
        problems.append(f"header is {header}, expected {list(VERDICT_COLUMNS)}")
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) != len(candidates):
        problems.append(f"{len(rows)} rows, the sheet has {len(candidates)} candidates")
    verdicts: list[Verdict] = []
    for number, (row, candidate) in enumerate(zip(rows, candidates, strict=False), start=1):
        where = f"row {number}"
        if len(row) != len(VERDICT_COLUMNS):
            problems.append(f"{where}: {len(row)} columns")
            continue
        fields = dict(zip(VERDICT_COLUMNS, row, strict=True))
        expected = {
            "number": str(number),
            "question_id": candidate["question_id"],
            "split": candidate["split"],
            "question": candidate["question"],
            "sheet_sha256": digest,
        }
        for column, value in expected.items():
            if fields[column] != value:
                problems.append(f"{where}: {column} {fields[column]!r} does not match the sheet")
        verdict, note = fields["verdict"], fields["note"]
        if verdict not in VERDICTS:
            problems.append(f"{where}: verdict {verdict!r} is not exactly one of {list(VERDICTS)}")
        if verdict == "exclude (ii)":
            scope = {paper["paper_id"] for paper in candidate["scope"]}
            named = set(_ARXIV_ID.findall(note))
            if not named & scope:
                problems.append(f"{where}: exclude (ii) names no paper from its scope")
            if named - scope:
                problems.append(
                    f"{where}: exclude (ii) names {sorted(named - scope)}, outside its scope"
                )
        verdicts.append(Verdict(candidate["question_id"], verdict, note))
    if problems:
        raise VerdictFileInvalid(problems)
    return verdicts


def build_sets(
    rows: Sequence[Mapping[str, Any]],
    dataset: DRet,
    verdicts: Sequence[Verdict],
    *,
    sheet_sha256: str,
) -> dict[str, Any]:
    """D177's sets: every labelled question, by category, with what was left out.

    A withheld question appears twice by design: once as `answerable` (asked beside
    its own paper) and once as `unanswerable-withheld` (asked in the hashed scope
    that omits it). Each entry is keyed by question and category, and a withheld
    entry carries its scope, so scoring needs nothing but this file and the corpus.
    """
    labelled = label_questions(rows, dataset)
    candidates = withheld_candidates(labelled)
    by_id = {verdict.question_id: verdict for verdict in verdicts}
    if [c.question_id for c in candidates] != [v.question_id for v in verdicts]:
        raise ValueError("the verdicts are not for this snapshot's withheld candidates")
    papers = sorted(dataset.papers)
    entries: list[dict[str, Any]] = []
    for question in labelled.questions:
        if question.label is Answerability.DISAGREE:
            continue
        category = (
            Category.ANSWERABLE
            if question.label is Answerability.ANSWERABLE
            else Category.UNANSWERABLE_QASPER
        )
        entries.append(_entry(question, category))
    excluded: list[dict[str, Any]] = []
    for candidate in candidates:
        verdict = by_id[candidate.question_id]
        scope = list(withheld_scope(candidate.question_id, candidate.paper_id, papers))
        if verdict.verdict == "keep":
            entry = _entry(candidate, Category.UNANSWERABLE_WITHHELD)
            entry |= {"withheld_paper": candidate.paper_id, "scope": scope}
            entries.append(entry)
        else:
            excluded.append(
                {
                    "question_id": candidate.question_id,
                    "verdict": verdict.verdict,
                    "note": verdict.note,
                    "scope": scope,
                }
            )
    entries.sort(key=lambda e: (e["split"], e["category"], e["question_id"]))
    counts: dict[str, int] = {}
    for entry in entries:
        key = f"{entry['split']}/{entry['category']}"
        counts[key] = counts.get(key, 0) + 1
    return {
        "set_version": SET_VERSION,
        "decision": "D177",
        "sheet_sha256": sheet_sha256,
        "counts": dict(sorted(counts.items())),
        "left_out": {
            "disagree": sum(q.label is Answerability.DISAGREE for q in labelled.questions),
            "outside_dret": labelled.outside_dret,
            "withheld_excluded": len(excluded),
        },
        "withheld_excluded": excluded,
        "questions": entries,
    }


def _entry(question: LabelledQuestion, category: Category) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "paper_id": question.paper_id,
        "split": question.split.name.lower(),
        "category": category.value,
        "question": question.text,
    }


def write_sets(
    sets: Mapping[str, Any],
    *,
    labeller: str,
    labelled_on: str,
    verdicts_sha256: str,
    path: Path | None = None,
    manifest: Path | None = None,
) -> dict[str, Any]:
    """Write the sets and their manifest; return the manifest.

    The paths default to `SETS` and `SETS_MANIFEST`, read at call time so a test
    can point them elsewhere.
    """
    path = SETS if path is None else path
    manifest = SETS_MANIFEST if manifest is None else manifest
    body = json.dumps(sets, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    data = body.encode("utf-8")
    path.write_bytes(data)
    record = {
        "dataset": "scholiast-m5-answerability",
        "set_version": sets["set_version"],
        "decision": "D177",
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "counts": sets["counts"],
        "left_out": sets["left_out"],
        "derived_from": "data/eval/qasper-validation.json.gz, labels from its annotators",
        "licence": "CC BY 4.0 (derived from QASPER)",
        "sheet_sha256": sets["sheet_sha256"],
        "verdicts_sha256": verdicts_sha256,
        "labeller": labeller,
        "labelled_on": labelled_on,
    }
    manifest.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return record


def load_sets(path: Path | None = None, manifest: Path | None = None) -> dict[str, Any]:
    """The sets, refused unless they are byte-for-byte the file the manifest records."""
    path = SETS if path is None else path
    manifest = SETS_MANIFEST if manifest is None else manifest
    data = path.read_bytes()
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"{path.name} does not match its manifest's sha256; it was edited")
    sets: dict[str, Any] = json.loads(data.decode("utf-8"))
    if sets["set_version"] != SET_VERSION:
        raise ValueError(f"{path.name} is {sets['set_version']}, this code reads {SET_VERSION}")
    return sets
