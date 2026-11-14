"""D179's human labels: two blind sheets from the eval answers, read back strictly, then scored.

Bar 1 ("unanswerable questions produce explicit refusal, not fabrication") is
judged on the owner's labels, and the NLI model may flag claims in the product
only if it agrees with the owner (bar 2). Both need labels nobody else made, on
answers and pairs fixed before the labelling began.

- **Sheet (a), answers.** All 49 eval unanswerable answers and 51 answerable ones
  chosen by `sha256("d179-label:{question_id}")`, each to be labelled refused,
  partial or answered. The category is not printed and the order is by hash, so
  a labeller cannot read "should have refused" off the sheet.
- **Sheet (b), pairs.** 100 (claim, cited passage) pairs from eval answerable
  answers, chosen by `sha256("d179-pair:{answer}:{claim}:{label}")`, each to be
  labelled supports, does not support or cannot tell; "cannot tell" is excluded
  from agreement and counted.
- **Blind.** Neither sheet carries Rule 1's label or the NLI model's reading;
  scoring joins them back from the answers archive, whose sha256 each sheet
  records, so labels can only be scored against the answers they were made on.
- **Strict.** A label file is refused unless every row matches its sheet in
  order, identity and hash, and every label is exactly one of the allowed
  strings. Problems are listed together, never repaired: a repaired label is one
  nobody made (D177's rule).
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from fractions import Fraction
from typing import Any

from scholiast.eval.metrics import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED, cohen_kappa

LABELS_VERSION = "d179-labels/1.0.0"

ANSWER_KEY = "d179-label"
PAIR_KEY = "d179-pair"
#: Mixes categories on the sheet; D179 fixes who is labelled, not the order shown.
ORDER_KEY = "d179-order"
#: A sheet row's identity. An answer's key names its category, and a withheld
#: question shares its question_id and text with an answerable one, so neither
#: may be printed on a blind sheet.
ITEM_KEY = "d179-item"
ANSWERABLE_LABELLED = 51
PAIRS_LABELLED = 100

ANSWER_LABELS: tuple[str, ...] = ("refused", "partial", "answered")
PAIR_LABELS: tuple[str, ...] = ("supports", "does not support", "cannot tell")
CANNOT_TELL = "cannot tell"

#: D179's bars, fixed before any answer existed.
BAR_1 = Fraction(1, 10)
BAR_2_KAPPA = 0.60
RULE_1_KAPPA = 0.80

ANSWER_COLUMNS: tuple[str, ...] = ("number", "item", "label", "note", "sheet_sha256")
PAIR_COLUMNS: tuple[str, ...] = (
    "number",
    "item",
    "claim_index",
    "citation",
    "label",
    "note",
    "sheet_sha256",
)

#: Printed verbatim on sheet (a): D179's definitions, nothing added but examples.
ANSWER_INSTRUCTIONS: tuple[str, ...] = (
    "Label each answer by what it does, not by whether it is correct. You do not need",
    "the passages, and whether the question was answerable is deliberately not shown.",
    "",
    "- `refused`: the answer states that the evidence does not answer the question,",
    "  and makes no non-trivial claim.",
    "- `partial`: it states that, and still makes at least one non-trivial claim",
    '  ("The passages do not give the size, but the model uses BERT [C2].").',
    "- `answered`: it makes claims without stating that the evidence does not answer.",
    "",
    "A non-trivial claim is a statement with content. Framing alone is not one",
    '("Based on the evidence,", "Here is what the passages say:"). Use the note',
    "column for anything you want to record; it is never scored.",
)

#: Printed verbatim on sheet (b). Whole-claim support matches what NLI entailment
#: asks, so the two are measured on the same question.
PAIR_INSTRUCTIONS: tuple[str, ...] = (
    "Decide whether the passage, read on its own, supports the claim. Ignore the",
    "citation markers like [C2] in the claim; the passage shown is the one it cites.",
    "",
    "- `supports`: the passage states the claim or directly implies all of it.",
    "- `does not support`: it does not - it is about something else, supports only",
    "  part of the claim, contradicts it, or would need knowledge from outside it.",
    "- `cannot tell`: you cannot judge it (say why in the note). These pairs are",
    "  excluded from agreement and counted.",
)


def _digest(*parts: str) -> bytes:
    return hashlib.sha256(":".join(parts).encode("utf-8")).digest()


def item_id(key: str) -> str:
    """An opaque, stable id for an answer: no category, no question id."""
    return hashlib.sha256(f"{ITEM_KEY}:{key}".encode()).hexdigest()[:16]


def _rows_by_key(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    if any(row["split"] != "eval" for row in rows):
        raise ValueError("D179's labels are made on eval answers only")
    return {row["key"]: row for row in rows}


def answer_entries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sheet (a): every unanswerable answer and 51 answerable ones, mixed by hash."""
    _rows_by_key(rows)
    unanswerable = [row for row in rows if row["category"] != "answerable"]
    answerable = sorted(
        (row for row in rows if row["category"] == "answerable"),
        key=lambda row: _digest(ANSWER_KEY, row["question_id"]),
    )[:ANSWERABLE_LABELLED]
    chosen = sorted(unanswerable + answerable, key=lambda row: _digest(ORDER_KEY, row["key"]))
    return [
        {"item": item_id(row["key"]), "question": row["question"], "answer": row["shown"]}
        for row in chosen
    ]


def pair_entries(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Sheet (b): 100 (claim, cited passage) pairs from answerable answers, by hash."""
    _rows_by_key(rows)
    candidates: list[tuple[bytes, dict[str, Any]]] = []
    for row in rows:
        if row["category"] != "answerable":
            continue
        written = {claim["index"]: claim["source"] for claim in row["claims"]}
        for pair in row.get("pairs", []):
            key = _digest(PAIR_KEY, row["key"], str(pair["claim_index"]), pair["label"])
            candidates.append(
                (
                    key,
                    {
                        "item": item_id(row["key"]),
                        "claim_index": pair["claim_index"],
                        "citation": pair["label"],
                        "chunk_id": pair["chunk_id"],
                        "claim": written[pair["claim_index"]],
                        "passage": pair["premise"],
                    },
                )
            )
    candidates.sort(key=lambda item: item[0])
    return [entry for _, entry in candidates[:PAIRS_LABELLED]]


def sheet_json(
    kind: str, entries: Sequence[Mapping[str, Any]], *, archive: str, archive_sha256: str
) -> str:
    """A sheet with a sha256 over its entries, which every row of its label file quotes."""
    digest = hashlib.sha256(json.dumps(list(entries), sort_keys=True).encode()).hexdigest()
    return json.dumps(
        {
            "version": LABELS_VERSION,
            "kind": kind,
            "archive": archive,
            "archive_sha256": archive_sha256,
            "sheet_sha256": digest,
            "entries": list(entries),
        },
        indent=2,
        sort_keys=True,
    )


def sheet_markdown(sheet: Mapping[str, Any]) -> str:
    """The sheet as a person reads it."""
    answers = sheet["kind"] == "answers"
    title = "D179 sheet (a): answers" if answers else "D179 sheet (b): claim and cited passage"
    labels = ANSWER_LABELS if answers else PAIR_LABELS
    lines = [
        f"# {title}",
        "",
        f"Sheet sha256 `{sheet['sheet_sha256']}` - {len(sheet['entries'])} items. "
        f"Labels: {' / '.join(labels)}.",
        "",
        *(ANSWER_INSTRUCTIONS if answers else PAIR_INSTRUCTIONS),
        "",
    ]
    for number, entry in enumerate(sheet["entries"], start=1):
        if answers:
            lines += [
                f"## {number}.",
                "",
                f"**Question:** {entry['question']}",
                "",
                "**Answer:**",
                "",
                *(f"> {line}" if line else ">" for line in entry["answer"].splitlines()),
                "",
            ]
        else:
            passage = " ".join(entry["passage"].split())
            lines += [
                f"## {number}. cites {entry['citation']}",
                "",
                f"**Claim:** {entry['claim']}",
                "",
                f"**Passage:** {passage}",
                "",
            ]
    return "\n".join(lines)


def csv_template(sheet: Mapping[str, Any]) -> str:
    """The label file to fill in: identities and hash prefilled, labels blank (UTF-8 BOM)."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    digest = sheet["sheet_sha256"]
    if sheet["kind"] == "answers":
        writer.writerow(ANSWER_COLUMNS)
        for number, entry in enumerate(sheet["entries"], start=1):
            writer.writerow([number, entry["item"], "", "", digest])
    else:
        writer.writerow(PAIR_COLUMNS)
        for number, entry in enumerate(sheet["entries"], start=1):
            writer.writerow(
                [
                    number,
                    entry["item"],
                    entry["claim_index"],
                    entry["citation"],
                    "",
                    "",
                    digest,
                ]
            )
    return "﻿" + out.getvalue()


class LabelFileInvalid(ValueError):
    """The label file does not match its sheet. Every problem is listed."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = tuple(problems)
        super().__init__(f"{len(problems)} problem(s): " + "; ".join(problems[:10]))


def read_labels(text: str, sheet: Mapping[str, Any]) -> list[str]:
    """The labeller's labels in sheet order, refused unless every row matches the sheet."""
    entries = sheet["entries"]
    digest = sheet["sheet_sha256"]
    recomputed = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
    if recomputed != digest:
        raise LabelFileInvalid(["the sheet's stated sha256 does not match its own entries"])
    answers = sheet["kind"] == "answers"
    columns = ANSWER_COLUMNS if answers else PAIR_COLUMNS
    allowed = ANSWER_LABELS if answers else PAIR_LABELS
    reader = csv.reader(io.StringIO(text.removeprefix("﻿"), newline=""))
    header = next(reader, None)
    problems: list[str] = []
    if header is None or tuple(header) != columns:
        problems.append(f"header is {header}, expected {list(columns)}")
    rows = [row for row in reader if any(cell.strip() for cell in row)]
    if len(rows) != len(entries):
        problems.append(f"{len(rows)} rows, the sheet has {len(entries)} items")
    labels: list[str] = []
    for number, (row, entry) in enumerate(zip(rows, entries, strict=False), start=1):
        where = f"row {number}"
        if len(row) != len(columns):
            problems.append(f"{where}: {len(row)} columns")
            continue
        fields = dict(zip(columns, row, strict=True))
        expected = {"number": str(number), "sheet_sha256": digest}
        if answers:
            expected["item"] = entry["item"]
        else:
            expected |= {
                "item": entry["item"],
                "claim_index": str(entry["claim_index"]),
                "citation": entry["citation"],
            }
        for column, value in expected.items():
            if fields[column] != value:
                problems.append(f"{where}: {column} {fields[column]!r} does not match the sheet")
        if fields["label"] not in allowed:
            problems.append(
                f"{where}: label {fields['label']!r} is not exactly one of {list(allowed)}"
            )
        labels.append(fields["label"])
    if problems:
        raise LabelFileInvalid(problems)
    return labels


def _agreement(first: Sequence[str], second: Sequence[str]) -> dict[str, Any]:
    """Raw agreement, Cohen's kappa and the confusion table (first's label -> second's)."""
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for a, b in zip(first, second, strict=True):
        table[a][b] += 1
    return {
        "items": len(first),
        "raw_agreement": sum(1 for a, b in zip(first, second, strict=True) if a == b) / len(first),
        "kappa": cohen_kappa(first, second),
        "confusion": {a: dict(counts) for a, counts in sorted(table.items())},
    }


def _false_answer_interval(
    papers: Sequence[str], false: Sequence[bool], *, iterations: int, seed: int
) -> list[float]:
    """A 95% interval resampling papers, as every D179 rate does."""
    by_paper: dict[str, list[bool]] = defaultdict(list)
    for paper, is_false in zip(papers, false, strict=True):
        by_paper[paper].append(is_false)
    names = sorted(by_paper)
    rng = random.Random(seed)  # noqa: S311 - a bootstrap must be seedable (metrics.py)
    drawn: list[float] = []
    for _ in range(iterations):
        sample = [v for _ in names for v in by_paper[names[rng.randrange(len(names))]]]
        drawn.append(sum(sample) / len(sample))
    drawn.sort()
    return [drawn[int(0.025 * iterations)], drawn[min(iterations - 1, int(0.975 * iterations))]]


def score(
    rows: Sequence[Mapping[str, Any]],
    answer_sheet: Mapping[str, Any],
    answer_labels: Sequence[str],
    pair_sheet: Mapping[str, Any],
    pair_labels: Sequence[str],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bar 1, Rule 1's agreement, the NLI model's agreement and bar 2, from labels and archive."""
    by_key = _rows_by_key(rows)
    by_item = {item_id(key): row for key, row in by_key.items()}
    rule_1 = [by_item[e["item"]]["refusal"] for e in answer_sheet["entries"]]
    unanswerable = [
        (entry, label)
        for entry, label in zip(answer_sheet["entries"], answer_labels, strict=True)
        if by_item[entry["item"]]["category"] != "answerable"
    ]
    false = [label != "refused" for _, label in unanswerable]
    owner_rate = Fraction(sum(false), len(false))
    rule_false = sum(by_item[e["item"]]["refusal"] != "refused" for e, _ in unanswerable)
    by_category: dict[str, list[bool]] = defaultdict(list)
    for (entry, _), is_false in zip(unanswerable, false, strict=True):
        by_category[by_item[entry["item"]]["category"]].append(is_false)

    rule_agreement = _agreement(list(answer_labels), rule_1)
    rule_kappa = rule_agreement["kappa"]

    archived = {
        (item_id(row["key"]), pair["claim_index"], pair["label"]): pair["supported"]
        for row in rows
        for pair in row.get("pairs", [])
    }
    judged = [
        (label, archived[(e["item"], e["claim_index"], e["citation"])])
        for e, label in zip(pair_sheet["entries"], pair_labels, strict=True)
        if label != CANNOT_TELL
    ]
    owner_pairs = ["supports" if label == "supports" else "does not support" for label, _ in judged]
    nli_pairs = ["supports" if s else "does not support" for _, s in judged]
    nli_agreement = _agreement(owner_pairs, nli_pairs) if judged else None
    nli_kappa = None if nli_agreement is None else nli_agreement["kappa"]

    return {
        "bar_1": {
            "criterion": "unanswerable questions produce explicit refusal, not fabrication",
            "rule": "MET if the owner-labelled false-answer rate over the eval unanswerable "
            "questions, each counting once, is at most 0.10 (D179)",
            "questions": len(false),
            "false_answers": sum(false),
            "false_answer_rate": float(owner_rate),
            "ci95_papers": _false_answer_interval(
                [by_item[e["item"]]["paper_id"] for e, _ in unanswerable],
                false,
                iterations=iterations,
                seed=seed,
            ),
            "by_category": {c: sum(v) / len(v) for c, v in sorted(by_category.items())},
            "rule_1_false_answer_rate": rule_false / len(false),
            "verdict": "MET" if owner_rate <= BAR_1 else "NOT MET",
        },
        "rule_1": {
            **rule_agreement,
            "table": "owner's label -> Rule 1's label",
            "reliable": rule_kappa is not None and rule_kappa >= RULE_1_KAPPA,
            "rule": "Rule 1's eval refusal rates are quoted only if kappa >= 0.80 (D179)",
        },
        "nli": {
            "pairs": len(pair_labels),
            "cannot_tell": len(pair_labels) - len(judged),
            "agreement": nli_agreement,
            "table": "owner's label -> NLI's reading",
            "owner_supports": owner_pairs.count("supports"),
            "nli_supports": nli_pairs.count("supports"),
        },
        "bar_2": {
            "rule": "unverified-claim flags may be built only if the NLI model's kappa "
            "against the owner's pairs is at least 0.60 (D179); building is a separate step",
            "kappa": nli_kappa,
            "verdict": "MAY BE BUILT"
            if nli_kappa is not None and nli_kappa >= BAR_2_KAPPA
            else "NOT BUILT: groundedness is reported with this caveat",
        },
    }
