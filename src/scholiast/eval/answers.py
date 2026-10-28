"""D179's answers: production's own answer to each of D177's questions, generated offline.

M5 phase 2 measures whether the model refuses when it should (FR-Q4, now that
D178 adopted no tau) and whether what it says is backed by what it cites (ML-12,
FR-Q11). Both are read off answers, so this module makes the answers and nothing
else; the rules that read them are frozen separately, on train answers, before a
single eval answer exists (D179).

- **What is asked where.** Each question once: answerable and QASPER-unanswerable
  questions in draw 0 of D168's 10-paper scope, withheld questions in the
  10-paper scope the owner checked by hand. Phase 1's `instances` builds both and
  re-checks every withheld scope, so this module filters it rather than redrawing.
- **The answer path is production's.** Hybrid RRF to `HYBRID_DEPTH` within the
  scope, `EvidenceSet.assemble` to `DEFAULT_EVIDENCE_LIMIT` passages, the prompt
  `build_answer_prompt` renders with each passage's paper title, section path
  and other-work marker, and one generator, called as the chain calls it. Every
  constant is imported from the use case, so a change there moves this with it;
  a test drives the real `AnswerQuestionUseCase` and this path on the same
  stand-ins and requires byte-identical messages.
- **Train is for development, eval for measurement.** `dev_sample` picks the 167
  train answers the rules are built on. Eval answers are refused until the rules
  are frozen (`require_frozen_rules`), which is D179's order made mechanical.

An answer is written to a progress file as soon as it exists, so a standby costs
the answer in flight and nothing else (D160).
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scholiast.application.ports.llm import LLMProvider
from scholiast.application.ports.ml import TextRole
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.prompts import EvidenceBlock, build_answer_prompt
from scholiast.application.use_cases.answer_question import (
    DEFAULT_EVIDENCE_LIMIT,
    HYBRID_DEPTH,
    NO_EVIDENCE_ANSWER,
)
from scholiast.domain import claims as claims_module
from scholiast.domain.claims import CLAIMS_VERSION, Claim, split_claims
from scholiast.domain.lexical import query_weights, tokenize
from scholiast.domain.retrieval import (
    CitationStatus,
    EvidenceSet,
    audit_citations,
    reciprocal_rank_fusion,
    strip_labels,
)
from scholiast.eval import refusal as refusal_module
from scholiast.eval.abstention import UNANSWERABLE, Instance, instances
from scholiast.eval.answerability import Category
from scholiast.eval.dret import DRet
from scholiast.eval.gate import DECIDING_SIZE
from scholiast.eval.harness import EmbedderLike, LexicalLike
from scholiast.eval.refusal import REFUSAL_VERSION, Refusal, label_answer

#: The answer record's version stamp (NFR-M5). Rules read these rows, so a change
#: to what a row holds is a change to what they read.
ANSWERS_VERSION = "answers/1.0.0"

#: The one draw each answerable and QASPER-unanswerable question is asked in.
ANSWER_DRAW = 0

#: Train answers the rules are built on: every unanswerable train question, and
#: this many answerable ones, first by `sha256("d179-dev:{question_id}")`.
DEV_KEY = "d179-dev"
DEV_ANSWERABLE = 60

#: Each rule's source file, by version: the bytes D179's freeze covers.
RULE_SOURCES: Mapping[str, Path] = {
    CLAIMS_VERSION: Path(claims_module.__file__),
    REFUSAL_VERSION: Path(refusal_module.__file__),
}

#: The rules' source hashes, frozen on the 167 train answers and recorded in D179's
#: clarification before any eval answer is generated. A rule whose source no longer
#: hashes to its entry refuses eval answers.
FROZEN_RULE_HASHES: Mapping[str, str] = {
    CLAIMS_VERSION: "22e83044da599a13ba5db123dd9ebbabdafbe8af642e3a794c081074074df68c",
    REFUSAL_VERSION: "0d93a40a50689eb7899f0c5dbd593de44e348f53f756545e9e9b52b199f17c98",
}


class RulesNotFrozen(RuntimeError):
    """Eval answers were asked for before the rules that read them were frozen."""


class EmptyGeneration(RuntimeError):
    """The generator finished without text. Production would fail over; nothing is kept."""


def rule_hash(path: Path) -> str:
    """sha256 of a rule's source with LF line endings, so every checkout hashes it alike."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def require_frozen_rules(
    frozen: Mapping[str, str] | None = None, sources: Mapping[str, Path] | None = None
) -> None:
    """Refuse eval answers unless both rules are frozen and unchanged since.

    A refusal detector or claim splitter adjusted after reading eval answers would
    be fitted to the numbers it reports, so the order is enforced, not trusted: a
    rule edited after its freeze no longer matches its hash, and the run stops.
    """
    frozen = FROZEN_RULE_HASHES if frozen is None else frozen
    sources = RULE_SOURCES if sources is None else sources
    if not frozen:
        raise RulesNotFrozen(
            "eval answers are generated only after Rule 1 and Rule 2 are frozen by "
            "D179's clarification; generate train answers with --dev"
        )
    for version, path in sources.items():
        if version not in frozen:
            raise RulesNotFrozen(f"{version} has no frozen hash; D179 freezes both rules")
        if rule_hash(path) != frozen[version]:
            raise RulesNotFrozen(
                f"{version} changed after it was frozen ({path.name} no longer hashes to "
                f"{frozen[version][:12]}); a rule changed after the freeze needs a new one"
            )


# --- what is asked ----------------------------------------------------------------


def answer_instances(sets: Mapping[str, Any], dataset: DRet) -> list[Instance]:
    """Every question once, in the scope D179 asks it in, in the sets' order."""
    return [
        i for i in instances(sets, dataset) if i.size == DECIDING_SIZE and i.draw == ANSWER_DRAW
    ]


def dev_sample(asked: Sequence[Instance]) -> list[Instance]:
    """The train answers the rules are built on, in the sets' order."""
    train = [i for i in asked if i.split == "train"]
    ranked = sorted(
        (i for i in train if i.category is Category.ANSWERABLE),
        key=lambda i: hashlib.sha256(f"{DEV_KEY}:{i.question_id}".encode()).digest(),
    )
    chosen = {i.question_id for i in ranked[:DEV_ANSWERABLE]}
    return [i for i in train if i.category in UNANSWERABLE or i.question_id in chosen]


def eval_set(asked: Sequence[Instance]) -> list[Instance]:
    return [i for i in asked if i.split == "eval"]


def judged_chunks(dataset: DRet) -> dict[str, frozenset[str]]:
    """Each question's QASPER evidence chunks, which say whether it was shown."""
    judged: dict[str, set[str]] = {}
    for judgment in dataset.judgments:
        if judgment.grade > 0:
            judged.setdefault(judgment.query_id, set()).add(judgment.chunk_id)
    return {qid: frozenset(chunks) for qid, chunks in judged.items()}


# --- one answer -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Passage:
    """One passage the model was shown, under the label it was shown with."""

    label: str
    chunk_id: str
    paper_id: str
    score: float
    #: One of the question's QASPER evidence chunks.
    judged: bool


@dataclass(frozen=True, slots=True)
class Answer:
    """One generated answer and everything a rule or a metric reads off it."""

    key: str
    question_id: str
    paper_id: str
    split: str
    category: Category
    question: str
    scope: tuple[str, ...]
    passages: tuple[Passage, ...]
    #: sha256 of the system and user messages, NUL-separated: which prompt, exactly.
    messages_sha256: str | None
    prompt_chars: int
    #: The model's raw text, labels and all.
    text: str
    #: What the user is shown: fabricated labels stripped, as production stores it.
    shown: str
    #: (label, chunk id, status) per label occurrence, from layer 2's audit.
    citations: tuple[tuple[str, str | None, str], ...]
    provider: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_ms: int
    generation_ms: int
    #: D60: retrieval found nothing, so no generation call was made.
    structural_refusal: bool
    #: How many QASPER evidence chunks the corpus holds for this question.
    judged: int

    @property
    def evidence_shown(self) -> bool | None:
        """Whether any of the question's evidence was among the passages; None if it has none."""
        if self.judged == 0:
            return None
        return any(p.judged for p in self.passages)

    @property
    def fabricated(self) -> int:
        return sum(1 for _, _, status in self.citations if status == CitationStatus.FABRICATED)

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "question_id": self.question_id,
            "paper_id": self.paper_id,
            "split": self.split,
            "category": self.category.value,
            "question": self.question,
            "scope": list(self.scope),
            "passages": [
                {
                    "label": p.label,
                    "chunk_id": p.chunk_id,
                    "paper_id": p.paper_id,
                    "score": p.score,
                    "judged": p.judged,
                }
                for p in self.passages
            ],
            "messages_sha256": self.messages_sha256,
            "prompt_chars": self.prompt_chars,
            "text": self.text,
            "shown": self.shown,
            "citations": [list(c) for c in self.citations],
            "provider": self.provider,
            "model": self.model,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retrieval_ms": self.retrieval_ms,
            "generation_ms": self.generation_ms,
            "structural_refusal": self.structural_refusal,
            "judged": self.judged,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> Answer:
        return cls(
            key=data["key"],
            question_id=data["question_id"],
            paper_id=data["paper_id"],
            split=data["split"],
            category=Category(data["category"]),
            question=data["question"],
            scope=tuple(data["scope"]),
            passages=tuple(Passage(**p) for p in data["passages"]),
            messages_sha256=data["messages_sha256"],
            prompt_chars=data["prompt_chars"],
            text=data["text"],
            shown=data["shown"],
            citations=tuple((c[0], c[1], c[2]) for c in data["citations"]),
            provider=data["provider"],
            model=data["model"],
            prompt_tokens=data["prompt_tokens"],
            completion_tokens=data["completion_tokens"],
            retrieval_ms=data["retrieval_ms"],
            generation_ms=data["generation_ms"],
            structural_refusal=data["structural_refusal"],
            judged=data["judged"],
        )


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)


async def answer_one(
    instance: Instance,
    *,
    dataset: DRet,
    index: VectorIndex,
    lexical: LexicalLike,
    embedder: EmbedderLike,
    generator: LLMProvider,
    judged: Mapping[str, frozenset[str]],
) -> Answer:
    """Production's answer to one question in one scope.

    Retrieval and assembly are `AnswerQuestionUseCase._retrieve` and `_prepare`
    in hybrid mode, on the scope where production reads a workspace's paper ids;
    the passages are rendered from the corpus as production renders them from its
    repositories. The generator is called once, as the chain calls its first
    provider: there is no second provider to fail over to, so a generator that
    is down raises, and the run stops and resumes rather than keeping a
    degraded answer that production would have served differently.
    """
    started = time.perf_counter()
    vector = (await embedder.embed([instance.question], role=TextRole.QUERY))[0]
    dense = await index.search(vector, paper_ids=instance.scope, limit=HYBRID_DEPTH)
    sparse = await lexical.search_lexical(
        query_weights(tokenize(instance.question)), paper_ids=instance.scope, limit=HYBRID_DEPTH
    )
    fused = reciprocal_rank_fusion([dense, sparse], limit=HYBRID_DEPTH)
    retrieval_ms = _ms(started)

    evidence = EvidenceSet.assemble(fused, limit=DEFAULT_EVIDENCE_LIMIT)
    evidence_of = judged.get(instance.question_id, frozenset())
    passages: list[Passage] = []
    blocks: list[EvidenceBlock] = []
    for hit in evidence.chunks:
        chunk = dataset.chunks[hit.chunk_id]
        label = evidence.labels[hit.chunk_id]
        passages.append(
            Passage(label, hit.chunk_id, hit.paper_id, hit.score, hit.chunk_id in evidence_of)
        )
        blocks.append(
            EvidenceBlock(
                label=label,
                paper_title=dataset.papers[chunk.provenance.paper_id].title,
                section_path=chunk.provenance.section_path,
                text=chunk.text,
                describes_other_work=chunk.provenance.section_kind.describes_other_work,
            )
        )

    generation_started = time.perf_counter()
    text, messages_sha256, prompt_chars = "", None, 0
    provider = model = None
    prompt_tokens = completion_tokens = None
    if evidence.is_empty:
        text = NO_EVIDENCE_ANSWER
    else:
        system, user = build_answer_prompt(instance.question, blocks)
        messages_sha256 = hashlib.sha256(f"{system}\0{user}".encode()).hexdigest()
        prompt_chars = len(system) + len(user)
        async for piece in generator.stream(system=system, user=user):
            text += piece.text
            if piece.usage is not None and piece.usage.reported:
                prompt_tokens = piece.usage.prompt_tokens
                completion_tokens = piece.usage.completion_tokens
        if not text.strip():
            raise EmptyGeneration(f"{instance.key}: {generator.name} returned no text")
        provider, model = generator.name, generator.model

    audit = audit_citations(text, evidence)
    return Answer(
        key=instance.key,
        question_id=instance.question_id,
        paper_id=instance.paper_id,
        split=instance.split,
        category=instance.category,
        question=instance.question,
        scope=instance.scope,
        passages=tuple(passages),
        messages_sha256=messages_sha256,
        prompt_chars=prompt_chars,
        text=text,
        shown=strip_labels(text.strip(), audit.fabricated_labels),
        citations=tuple((c.label, c.chunk_id, c.status.value) for c in audit.citations),
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        retrieval_ms=retrieval_ms,
        generation_ms=_ms(generation_started),
        structural_refusal=evidence.is_empty,
        judged=len(evidence_of),
    )


async def answer_all(
    todo: Sequence[Instance],
    *,
    done: Mapping[str, Answer],
    sink: Callable[[Answer], None],
    after_first: Callable[[], Awaitable[None]] | None = None,
    **adapters: Any,
) -> list[Answer]:
    """Every instance answered, reusing `done` and handing each new answer to `sink` at once.

    `after_first` runs once, after the first answer this run generates: the
    model is loaded only then, so that is the first moment its placement can be
    read (D23), and a model on the CPU should stop the run at one answer, not 167.
    """
    rows: list[Answer] = []
    checked = after_first is None
    for instance in todo:
        row = done.get(instance.key)
        if row is None:
            row = await answer_one(instance, **adapters)
            sink(row)
            if not checked and after_first is not None and not row.structural_refusal:
                await after_first()
                checked = True
        rows.append(row)
    return rows


def read_answers(path: Path) -> dict[str, Answer]:
    """Answers a previous, interrupted run wrote. Only a torn LAST line is forgiven."""
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: dict[str, Answer] = {}
    for number, line in enumerate(lines, start=1):
        try:
            row = Answer.from_json(json.loads(line))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            if number == len(lines):
                break
            raise
        rows[row.key] = row
    return rows


# --- what the run reports ---------------------------------------------------------


def summarise(rows: Sequence[Answer], *, context_length: int | None) -> dict[str, Any]:
    """Counts a reader needs before trusting the answers, none of them a D179 metric.

    `at_context_limit` counts answers whose prompt filled the generator's context
    window: Ollama keeps what fits and drops the rest, and what it drops first is
    the start of the prompt, where the grounding rules are. Any such answer is a
    finding about production, not a number to report around.
    """
    reported = [r.prompt_tokens for r in rows if r.prompt_tokens is not None]
    shown = Counter(
        "no judged evidence" if r.evidence_shown is None else str(r.evidence_shown) for r in rows
    )
    return {
        "answers": len(rows),
        "by_category": dict(Counter(r.category.value for r in rows)),
        "structural_refusals": sum(1 for r in rows if r.structural_refusal),
        "evidence_shown": dict(shown),
        "fabricated_labels": sum(r.fabricated for r in rows),
        "answers_with_fabricated_labels": sum(1 for r in rows if r.fabricated),
        "prompt_tokens_reported": len(reported),
        "prompt_tokens_max": max(reported, default=None),
        "prompt_chars_max": max((r.prompt_chars for r in rows), default=0),
        "context_length": context_length,
        "at_context_limit": (
            None
            if context_length is None
            else sum(1 for tokens in reported if tokens >= context_length)
        ),
    }


# --- what the rules read ----------------------------------------------------------


def headings(answer: Answer, dataset: DRet) -> tuple[str, ...]:
    """Each shown passage's `title · section path`, as `EvidenceBlock` rendered it.

    The model sometimes copies these back, and Rule 2 removes them before it
    reads a sentence; they are rebuilt from the corpus, as the prompt built them.
    """
    out: list[str] = []
    for passage in answer.passages:
        provenance = dataset.chunks[passage.chunk_id].provenance
        out.append(f"{dataset.papers[provenance.paper_id].title} · {provenance.section_path}")
    return tuple(out)


@dataclass(frozen=True, slots=True)
class Reading:
    """What Rule 2 and Rule 1 make of one answer."""

    claims: tuple[Claim, ...]
    refusal: Refusal

    def to_json(self) -> dict[str, Any]:
        return {
            "refusal": self.refusal.value,
            "claims": [
                {
                    "index": c.index,
                    "source": c.source,
                    "text": c.text,
                    "labels": list(c.labels),
                    "kind": c.kind.value,
                    "refuses": c.refuses,
                }
                for c in self.claims
            ],
        }


def read_answer(answer: Answer, dataset: DRet) -> Reading:
    """Rule 2 on the model's raw text, then Rule 1 on its sentences (D179)."""
    claims = split_claims(answer.text, headings=headings(answer, dataset))
    return Reading(claims, label_answer(claims, structural=answer.structural_refusal))


def summarise_readings(rows: Sequence[Answer], readings: Sequence[Reading]) -> dict[str, Any]:
    """Rule 1's labels by category, and how many sentences Rule 2 found of each kind.

    Counts, not D179's metrics: those are rates with intervals, and on eval answers
    they wait for the NLI model and the owner's labels.
    """
    labels: dict[str, Counter[str]] = {}
    for row, reading in zip(rows, readings, strict=True):
        labels.setdefault(row.category.value, Counter())[reading.refusal.value] += 1
    return {
        "rules": {version: rule_hash(path) for version, path in RULE_SOURCES.items()},
        "refusal_by_category": {category: dict(counts) for category, counts in labels.items()},
        "sentences_by_kind": dict(Counter(c.kind.value for r in readings for c in r.claims)),
    }
