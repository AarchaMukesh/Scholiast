"""D179's measurements over the eval answers: refusal, groundedness, citation support and coverage.

Everything here reads answers that already exist, so it can be rerun from an
archive without Qdrant or Ollama; only `read_pairs` needs the NLI model, and it
takes the port, not the adapter.

- **Which answers carry claims.** An answer Rule 1 labels answered or partial
  makes non-trivial claims; a refused one makes none. The claim metrics run over
  answered and partial answerable answers, and, reported but not judged, over the
  false answers to unanswerable questions (D179).
- **One pair per valid citation.** A non-trivial claim is paired with each
  distinct label it carries that resolves to a passage the model was shown. The
  premise is that chunk's D-RET text, which `EvidenceBlock` rendered under its
  heading; the hypothesis is Rule 2's NLI text for the claim (D179's
  clarification). A fabricated label makes no pair: it is invalid, and an
  uncited claim counts as unsupported.
- **Every metric is a ratio of per-answer counts.** That is what lets one
  bootstrap resample papers for all of them, a withheld question moving with its
  own absent paper, as phase 1 did (D177). A resample that leaves a rate with no
  denominator is counted, not imputed.

Does not establish: support is read one passage at a time, so a claim that needs
two passages together reads as unsupported; a sentence is not an atomic claim
(D179). Bar 1 is judged on the owner's labels, not on Rule 1's numbers here.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from scholiast.application.ports.ml import Entailment, EntailmentChecker
from scholiast.domain.claims import ClaimKind
from scholiast.domain.retrieval import CitationStatus
from scholiast.eval.abstention import UNANSWERABLE, Instance
from scholiast.eval.answerability import Category
from scholiast.eval.answers import Answer, Reading
from scholiast.eval.dret import DRet
from scholiast.eval.metrics import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED
from scholiast.eval.refusal import Refusal

GROUNDING_VERSION = "grounding/1.0.0"

#: D179 regenerates 20 eval answers and reports how many changed; it does not say
#: which 20, so they are chosen like every other sample in M5, by hash.
REGEN_KEY = "d179-regen"
REGENERATED = 20

PairKey = tuple[str, int, str]


def carries_claims(reading: Reading) -> bool:
    """Answered or partial: an answer that makes non-trivial claims (D179)."""
    return reading.refusal is not Refusal.REFUSED


@dataclass(frozen=True, slots=True)
class Pair:
    """One (cited passage, claim) the NLI model reads."""

    answer_key: str
    claim_index: int
    label: str
    chunk_id: str
    premise: str
    hypothesis: str

    @property
    def key(self) -> PairKey:
        return (self.answer_key, self.claim_index, self.label)


def nli_pairs(answer: Answer, reading: Reading, dataset: DRet) -> list[Pair]:
    """Every non-trivial claim against each valid passage it cites; none for a refusal."""
    if not carries_claims(reading):
        return []
    shown = {passage.label: passage.chunk_id for passage in answer.passages}
    pairs: list[Pair] = []
    for claim in reading.claims:
        if claim.kind is not ClaimKind.CLAIM or not claim.text.strip():
            continue
        for label in claim.labels:
            chunk_id = shown.get(label)
            if chunk_id is not None:
                pairs.append(
                    Pair(
                        answer.key,
                        claim.index,
                        label,
                        chunk_id,
                        dataset.chunks[chunk_id].text,
                        claim.text,
                    )
                )
    return pairs


async def read_pairs(
    pairs: Sequence[Pair], checker: EntailmentChecker
) -> dict[PairKey, Entailment]:
    """The NLI model's reading of every pair, keyed by (answer, claim, label)."""
    readings = await checker.entails([p.premise for p in pairs], [p.hypothesis for p in pairs])
    return {pair.key: reading for pair, reading in zip(pairs, readings, strict=True)}


@dataclass(frozen=True, slots=True)
class Counts:
    """One answer's contribution to every D179 rate."""

    paper_id: str
    category: Category
    evidence_shown: bool | None
    refusal: Refusal
    claims: int
    covered_claims: int
    supported_claims: int
    pairs: int
    supported_pairs: int
    truncated_pairs: int
    labels_emitted: int
    labels_resolving: int


def count(answer: Answer, reading: Reading, nli: Mapping[PairKey, Entailment]) -> Counts:
    """What one answer adds to each numerator and denominator."""
    shown = {passage.label for passage in answer.passages}
    claims = covered = supported = pairs = supported_pairs = truncated = 0
    if carries_claims(reading):
        for claim in reading.claims:
            if claim.kind is not ClaimKind.CLAIM:
                continue
            claims += 1
            valid = [label for label in claim.labels if label in shown]
            readings = [
                nli[(answer.key, claim.index, label)]
                for label in valid
                if (answer.key, claim.index, label) in nli
            ]
            covered += bool(valid)
            supported += any(r.supported for r in readings)
            pairs += len(readings)
            supported_pairs += sum(r.supported for r in readings)
            truncated += sum(r.premise_truncated for r in readings)
    return Counts(
        paper_id=answer.paper_id,
        category=answer.category,
        evidence_shown=answer.evidence_shown,
        refusal=reading.refusal,
        claims=claims,
        covered_claims=covered,
        supported_claims=supported,
        pairs=pairs,
        supported_pairs=supported_pairs,
        truncated_pairs=truncated,
        labels_emitted=len(answer.citations),
        labels_resolving=sum(
            1 for _, _, status in answer.citations if status != CitationStatus.FABRICATED
        ),
    )


def _ratio(numerator: int, denominator: int) -> Fraction | None:
    return None if denominator == 0 else Fraction(numerator, denominator)


def _claim_rates(prefix: str, group: Sequence[Counts]) -> dict[str, Fraction | None]:
    """D179's claim metrics over answers that make claims."""
    rows = [c for c in group if c.refusal is not Refusal.REFUSED]
    claims = sum(c.claims for c in rows)
    return {
        f"{prefix}/groundedness": _ratio(sum(c.supported_claims for c in rows), claims),
        f"{prefix}/citation_support_precision": _ratio(
            sum(c.supported_pairs for c in rows), sum(c.pairs for c in rows)
        ),
        f"{prefix}/citation_coverage": _ratio(sum(c.covered_claims for c in rows), claims),
        f"{prefix}/citation_validity": _ratio(
            sum(c.labels_resolving for c in rows), sum(c.labels_emitted for c in rows)
        ),
        f"{prefix}/hallucination": _ratio(
            sum(1 for c in rows if c.supported_claims < c.claims), len(rows)
        ),
    }


def rates(counts: Sequence[Counts]) -> dict[str, Fraction | None]:
    """Every D179 rate, from counts alone."""
    answerable = [c for c in counts if c.category is Category.ANSWERABLE]
    unanswerable = [c for c in counts if c.category in UNANSWERABLE]

    def answered(rows: Sequence[Counts]) -> int:
        return sum(1 for c in rows if c.refusal is not Refusal.REFUSED)

    out: dict[str, Fraction | None] = {
        "false_answer": _ratio(answered(unanswerable), len(unanswerable)),
    }
    for category in UNANSWERABLE:
        rows = [c for c in unanswerable if c.category is category]
        out[f"false_answer/{category.value}"] = _ratio(answered(rows), len(rows))
    for name, label in (("refused", Refusal.REFUSED), ("partial", Refusal.PARTIAL)):
        out[name] = _ratio(sum(1 for c in answerable if c.refusal is label), len(answerable))
    for shown, tag in ((True, "shown"), (False, "not_shown"), (None, "no_judged_evidence")):
        rows = [c for c in answerable if c.evidence_shown is shown]
        out[f"refused/evidence_{tag}"] = _ratio(
            sum(1 for c in rows if c.refusal is Refusal.REFUSED), len(rows)
        )
    out |= _claim_rates("answerable", answerable)
    out |= _claim_rates("unanswerable_false", unanswerable)
    return out


def _percentiles(values: Sequence[float]) -> list[float] | None:
    """2.5th and 97.5th, by `bootstrap_interval`'s own convention."""
    if not values:
        return None
    n = len(values)
    return [values[int(0.025 * n)], values[min(n - 1, int(0.975 * n))]]


def paper_intervals(
    counts: Sequence[Counts],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Every rate with a 95% interval that resamples PAPERS (D179, as D177 did)."""
    by_paper: dict[str, list[Counts]] = defaultdict(list)
    for row in counts:
        by_paper[row.paper_id].append(row)
    papers = sorted(by_paper)
    point = rates(counts)
    drawn: dict[str, list[float]] = {name: [] for name in point}
    undefined: dict[str, int] = dict.fromkeys(point, 0)
    rng = random.Random(seed)  # noqa: S311 - a bootstrap must be seedable (metrics.py)
    for _ in range(iterations):
        sample = [row for _ in papers for row in by_paper[papers[rng.randrange(len(papers))]]]
        for name, value in rates(sample).items():
            if value is None:
                undefined[name] += 1
            else:
                drawn[name].append(float(value))
    out: dict[str, Any] = {
        name: {
            "point": None if point[name] is None else float(point[name]),  # type: ignore[arg-type]
            "ci95": _percentiles(sorted(drawn[name])),
            "undefined_resamples": undefined[name],
        }
        for name in point
    }
    out["papers"] = {"resampled": len(papers), "iterations": iterations, "seed": seed}
    return out


def totals(counts: Sequence[Counts]) -> dict[str, int]:
    """The raw counts behind the rates, so every rate can be checked by hand."""
    return {
        "answers": len(counts),
        "claims": sum(c.claims for c in counts),
        "pairs": sum(c.pairs for c in counts),
        "truncated_pairs": sum(c.truncated_pairs for c in counts),
        "labels_emitted": sum(c.labels_emitted for c in counts),
    }


def regeneration_sample(todo: Sequence[Instance]) -> list[Instance]:
    """The 20 eval questions asked a second time, by hash, in their original order."""
    ranked = sorted(
        todo, key=lambda i: hashlib.sha256(f"{REGEN_KEY}:{i.question_id}".encode()).digest()
    )
    chosen = {i.key for i in ranked[:REGENERATED]}
    return [i for i in todo if i.key in chosen]


def compare_regenerated(
    first: Sequence[tuple[Answer, Reading]], again: Sequence[tuple[Answer, Reading]]
) -> dict[str, Any]:
    """How many of the regenerated answers changed text, and how many changed Rule 1's label.

    Reported, not gated: production samples, and this says by how much (D179).
    """
    pairs = list(zip(first, again, strict=True))
    for (a, _), (b, _) in pairs:
        if a.key != b.key:
            raise ValueError(f"regenerated {b.key} against {a.key}")
    return {
        "regenerated": len(pairs),
        "text_changed": sum(1 for (a, _), (b, _) in pairs if a.text != b.text),
        "refusal_label_changed": sum(1 for (_, x), (_, y) in pairs if x.refusal != y.refusal),
        "changes": [
            {"key": a.key, "first": x.refusal.value, "again": y.refusal.value}
            for (a, x), (_, y) in pairs
            if x.refusal != y.refusal
        ],
    }
