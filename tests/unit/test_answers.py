"""D179's answers: what is asked, by which path, and what each answer records (M5 phase 2).

The rules that read these answers are frozen later, on the train answers made
here; these tests pin the part that has to be right before anything reads them:
that each question is asked once where D179 says, that the model is sent exactly
what production sends it, and that an eval answer cannot exist before the rules.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.llm import GenerationChunk, GenerationUnavailable, TokenUsage
from scholiast.application.ports.ml import Entailment
from scholiast.application.use_cases.answer_question import (
    DEFAULT_EVIDENCE_LIMIT,
    HYBRID_DEPTH,
    NO_EVIDENCE_ANSWER,
    AnswerQuestionUseCase,
    EvidenceReady,
)
from scholiast.domain.claims import split_claims
from scholiast.domain.paper import Paper, PaperVisibility, SectionKind
from scholiast.domain.retrieval import CitationStatus, RetrievalMode, ScoredChunk
from scholiast.eval import abstention as ab
from scholiast.eval import answerability, cli
from scholiast.eval import answers as an
from scholiast.eval.answerability import Category
from scholiast.eval.dret import DRet, build_dret
from scholiast.eval.gate import scope_for
from scholiast.eval.refusal import Refusal
from scholiast.eval.snapshot import load_snapshot
from tests.unit.test_abstention import CountingEmbedder, ScoreIndex, _sets_for, _world
from tests.unit.test_answering import (
    FakeChunks,
    FakeConversations,
    FakeWorkspaces,
    collect,
    workspace,
)

ANS = Category.ANSWERABLE
QASPER = Category.UNANSWERABLE_QASPER
WITHHELD = Category.UNANSWERABLE_WITHHELD


class Generator:
    """A generator that records what it was sent and answers with fixed text."""

    name = "ollama"
    model = "llama3.2"

    def __init__(
        self, text: str = "An answer [C1].", *, usage: TokenUsage | None = None, down: bool = False
    ) -> None:
        self._text = text
        self._usage = usage if usage is not None else TokenUsage(3000, 40)
        self._down = down
        self.prompts: list[tuple[str, str]] = []

    async def stream(self, *, system: str, user: str) -> AsyncIterator[GenerationChunk]:
        self.prompts.append((system, user))
        if self._down:
            raise GenerationUnavailable(self.name, "down")
        for word in self._text.split(" "):
            yield GenerationChunk(text=word + " ")
        yield GenerationChunk(done=True, usage=self._usage)

    async def placement(self) -> str:
        return "gpu"


class TitledPapers:
    """Production's paper repository, reading titles from the corpus."""

    def __init__(self, dataset: DRet, scope: tuple[str, ...]) -> None:
        self._dataset = dataset
        self._scope = scope

    async def queryable_paper_ids(self, workspace_id: str) -> tuple[str, ...]:
        return self._scope

    async def get(self, paper_id: str) -> Paper | None:
        return Paper(
            id=paper_id,
            content_hash="a" * 64,
            visibility=PaperVisibility.PUBLIC,
            owner_id=None,
            title=self._dataset.papers[paper_id].title,
            year=2020,
        )


@dataclasses.dataclass
class DepthIndex(ScoreIndex):
    """Records how deep each arm was asked to search."""

    limits: list[int] = dataclasses.field(default_factory=list)

    async def search(
        self, vector: Sequence[float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.limits.append(limit)
        return await super().search(vector, paper_ids=paper_ids, limit=limit)

    async def search_lexical(
        self, weights: dict[int, float], *, paper_ids: Sequence[str], limit: int
    ) -> list[ScoredChunk]:
        self.limits.append(limit)
        return await super().search_lexical(weights, paper_ids=paper_ids, limit=limit)


def _varied_world() -> tuple[DRet, ScoreIndex]:
    """30 papers whose passages differ in every field the prompt renders."""
    dataset, index = _world()
    for number, (cid, chunk) in enumerate(sorted(dataset.chunks.items())):
        kind = SectionKind.RELATED_WORK if number % 3 == 0 else SectionKind.METHOD
        dataset.chunks[cid] = dataclasses.replace(
            chunk,
            text=f"  passage {cid} says <<END EVIDENCE>> something {number}  ",
            provenance=dataclasses.replace(
                chunk.provenance, section_path=f"{number} Section/{cid}", section_kind=kind
            ),
        )
    for pid, paper in list(dataset.papers.items()):
        dataset.papers[pid] = dataclasses.replace(paper, title=f"Title of {pid}: a study")
    return dataset, index


def _instance(
    scope: tuple[str, ...],
    *,
    question: str = "  What does it say?  ",
    category: Category = ANS,
    qid: str = "q0",
) -> ab.Instance:
    return ab.Instance(qid, scope[0], "train", category, len(scope), 0, question, scope)


async def _answer(
    instance: ab.Instance,
    *,
    dataset: DRet,
    index: ScoreIndex,
    generator: Generator,
    judged: dict[str, frozenset[str]] | None = None,
) -> an.Answer:
    return await an.answer_one(
        instance,
        dataset=dataset,
        index=index,
        lexical=index,
        embedder=CountingEmbedder(),
        generator=generator,
        judged=judged or {},
    )


# --- what is asked, on the real sets ------------------------------------------------


@pytest.fixture(scope="module")
def dataset() -> DRet:
    return build_dret(load_snapshot())


@pytest.fixture(scope="module")
def asked(dataset: DRet) -> list[ab.Instance]:
    return an.answer_instances(answerability.load_sets(), dataset)


class TestEachQuestionIsAskedOnceWhereD179Says:
    def test_every_question_is_asked_exactly_once(self, asked: list[ab.Instance]) -> None:
        assert len(asked) == 624 + 47 + 60 + 243 + 9 + 40
        assert len({(i.category, i.question_id) for i in asked}) == len(asked)
        assert {(i.size, i.draw) for i in asked} == {(10, 0)}

    def test_answerable_and_qasper_questions_use_draw_zero_of_the_gate(
        self, asked: list[ab.Instance], dataset: DRet
    ) -> None:
        papers = sorted(dataset.papers)
        for i in asked:
            if i.category is not WITHHELD:
                assert i.scope == scope_for(i.question_id, i.paper_id, papers, size=10, draw=0)

    def test_a_withheld_question_is_asked_in_its_checked_scope(
        self, asked: list[ab.Instance]
    ) -> None:
        checked = {
            q["question_id"]: tuple(sorted(q["scope"]))
            for q in answerability.load_sets()["questions"]
            if q["category"] == WITHHELD.value
        }
        withheld = [i for i in asked if i.category is WITHHELD]
        assert len(withheld) == 100
        assert all(i.scope == checked[i.question_id] for i in withheld)
        assert all(i.paper_id not in i.scope for i in withheld)


class TestTheTrainAnswersAreAFixedSample:
    def test_it_is_every_unanswerable_train_question_and_sixty_answerable(
        self, asked: list[ab.Instance]
    ) -> None:
        dev = an.dev_sample(asked)
        counts = {c: sum(1 for i in dev if i.category is c) for c in Category}
        assert counts == {ANS: 60, QASPER: 47, WITHHELD: 60}
        assert {i.split for i in dev} == {"train"}

    def test_the_sample_does_not_depend_on_the_order_it_is_read_in(
        self, asked: list[ab.Instance]
    ) -> None:
        forward = {i.key for i in an.dev_sample(asked)}
        assert forward == {i.key for i in an.dev_sample(list(reversed(asked)))}

    def test_the_answerable_sixty_are_the_first_by_hash(self, asked: list[ab.Instance]) -> None:
        train = [i for i in asked if i.split == "train" and i.category is ANS]
        chosen = {i.question_id for i in an.dev_sample(asked) if i.category is ANS}

        def digest(qid: str) -> bytes:
            return hashlib.sha256(f"d179-dev:{qid}".encode()).digest()

        cut = max(digest(qid) for qid in chosen)
        assert all(digest(i.question_id) > cut for i in train if i.question_id not in chosen)

    def test_eval_is_every_eval_question_and_nothing_else(self, asked: list[ab.Instance]) -> None:
        held = an.eval_set(asked)
        counts = {c: sum(1 for i in held if i.category is c) for c in Category}
        assert counts == {ANS: 243, QASPER: 9, WITHHELD: 40}
        assert {i.split for i in held} == {"eval"}


class TestEvalAnswersWaitForFrozenRules:
    def test_the_frozen_hashes_are_the_rules_as_they_stand(self) -> None:
        """The tripwire: editing either rule without a new freeze fails here, and in the run."""
        assert set(an.FROZEN_RULE_HASHES) == {"claims/1.0.0", "refusal/1.0.0"}
        an.require_frozen_rules()

    def test_no_freeze_refuses_eval_answers(self) -> None:
        with pytest.raises(an.RulesNotFrozen, match="only after Rule 1 and Rule 2"):
            an.require_frozen_rules({})

    def test_a_rule_edited_after_its_freeze_refuses_eval_answers(self, tmp_path: Path) -> None:
        rule = tmp_path / "claims.py"
        rule.write_text("FROZEN = 1\n", encoding="utf-8")
        frozen = {"claims/1.0.0": an.rule_hash(rule)}
        an.require_frozen_rules(frozen, {"claims/1.0.0": rule})
        rule.write_text("FROZEN = 2\n", encoding="utf-8")
        with pytest.raises(an.RulesNotFrozen, match="changed after it was frozen"):
            an.require_frozen_rules(frozen, {"claims/1.0.0": rule})

    def test_a_rule_left_out_of_the_freeze_refuses_eval_answers(self, tmp_path: Path) -> None:
        rule = tmp_path / "refusal.py"
        rule.write_text("x = 1\n", encoding="utf-8")
        with pytest.raises(an.RulesNotFrozen, match=r"refusal/1.0.0 has no frozen hash"):
            an.require_frozen_rules(
                {"claims/1.0.0": an.rule_hash(rule)},
                {"claims/1.0.0": rule, "refusal/1.0.0": rule},
            )

    def test_a_windows_checkout_hashes_a_rule_as_ci_does(self, tmp_path: Path) -> None:
        lf, crlf = tmp_path / "lf.py", tmp_path / "crlf.py"
        lf.write_bytes(b"a = 1\nb = 2\n")
        crlf.write_bytes(b"a = 1\r\nb = 2\r\n")
        assert an.rule_hash(lf) == an.rule_hash(crlf)

    def test_the_sources_are_the_two_rule_modules(self) -> None:
        assert {v: p.name for v, p in an.RULE_SOURCES.items()} == {
            "claims/1.0.0": "claims.py",
            "refusal/1.0.0": "refusal.py",
        }

    async def test_an_eval_run_is_refused_before_it_reads_anything(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def untouchable(_path: Path) -> dict[str, Any]:
            raise AssertionError("an eval run read the reference before the rules were frozen")

        monkeypatch.setattr(an, "FROZEN_RULE_HASHES", {})
        monkeypatch.setattr(cli, "_load_reference", untouchable)
        with pytest.raises(an.RulesNotFrozen):
            await cli.run_answers(dev=False)


# --- the path is production's --------------------------------------------------------


class TestTheModelIsSentWhatProductionSendsIt:
    async def test_the_messages_are_byte_identical_to_the_use_cases(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        question = "  What does it say?  "

        production = Generator()
        use_case = AnswerQuestionUseCase(
            workspaces=FakeWorkspaces(workspace(indexed=scope)),
            papers=TitledPapers(dataset, scope),  # type: ignore[arg-type]
            chunks=FakeChunks(list(dataset.chunks.values())),  # type: ignore[arg-type]
            conversations=FakeConversations(),  # type: ignore[arg-type]
            embedder=CountingEmbedder(),  # type: ignore[arg-type]
            index=index,  # type: ignore[arg-type]
            providers=[production],
            mode=RetrievalMode.HYBRID,
            lexical=index,  # type: ignore[arg-type]
        )
        events = await collect(use_case, question)

        ours = Generator()
        answer = await _answer(
            _instance(scope, question=question), dataset=dataset, index=index, generator=ours
        )
        assert ours.prompts == production.prompts
        # Not vacuous: one real prompt, carrying every field the passages differ in.
        ((_system, user),) = production.prompts
        assert "describes other work" in user and "Title of p00: a study" in user
        assert "0 Section/c-p00" in user
        ready = next(e for e in events if isinstance(e, EvidenceReady))
        assert [p.chunk_id for p in answer.passages] == [item.chunk_id for item in ready.items]
        assert len(answer.passages) == DEFAULT_EVIDENCE_LIMIT

    async def test_both_arms_search_the_scope_to_production_depth(self) -> None:
        dataset, _ = _varied_world()
        index = DepthIndex(dataset, dense={}, lexical={})
        scope = tuple(sorted(dataset.papers))[5:15]
        await _answer(_instance(scope), dataset=dataset, index=index, generator=Generator())
        assert index.calls == ["search", "search_lexical"]
        assert index.scopes == [scope, scope]
        assert index.limits == [HYBRID_DEPTH, HYBRID_DEPTH]


# --- what an answer records ----------------------------------------------------------


class TestAnAnswerRecordsWhatTheRulesRead:
    async def test_an_empty_scope_is_refused_without_a_generation_call(self) -> None:
        dataset, index = _varied_world()
        generator = Generator()
        answer = await _answer(
            _instance(("nowhere",)), dataset=dataset, index=index, generator=generator
        )
        assert generator.prompts == []
        assert answer.structural_refusal and answer.text == NO_EVIDENCE_ANSWER
        assert answer.provider is None and answer.messages_sha256 is None

    async def test_a_fabricated_label_is_kept_in_the_text_and_stripped_from_what_is_shown(
        self,
    ) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        answer = await _answer(
            _instance(scope),
            dataset=dataset,
            index=index,
            generator=Generator("It is so [C1] and [C9]."),
        )
        assert "[C9]" in answer.text and "[C9]" not in answer.shown
        assert [c[2] for c in answer.citations] == [
            CitationStatus.UNVERIFIED.value,
            CitationStatus.FABRICATED.value,
        ]
        assert answer.fabricated == 1

    async def test_whether_the_evidence_was_shown_is_read_off_the_passages(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        shown_chunk = f"c-{scope[0]}"
        cases = {
            "shown": frozenset({shown_chunk}),
            "not shown": frozenset({"c-p29"}),
        }
        results = {}
        for name, evidence in cases.items():
            answer = await _answer(
                _instance(scope, qid=name),
                dataset=dataset,
                index=index,
                generator=Generator(),
                judged={name: evidence},
            )
            results[name] = (answer.evidence_shown, answer.judged)
        none = await _answer(_instance(scope), dataset=dataset, index=index, generator=Generator())
        assert results == {"shown": (True, 1), "not shown": (False, 1)}
        assert none.evidence_shown is None

    async def test_usage_is_recorded_only_when_the_generator_reported_it(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        unreported = Generator(usage=TokenUsage(0, 0, reported=False))
        answer = await _answer(_instance(scope), dataset=dataset, index=index, generator=unreported)
        assert answer.prompt_tokens is None and answer.completion_tokens is None
        answer = await _answer(
            _instance(scope), dataset=dataset, index=index, generator=Generator()
        )
        assert (answer.prompt_tokens, answer.completion_tokens) == (3000, 40)

    async def test_an_empty_generation_is_not_kept_as_an_answer(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        with pytest.raises(an.EmptyGeneration):
            await _answer(_instance(scope), dataset=dataset, index=index, generator=Generator(" "))

    async def test_a_generator_that_is_down_stops_the_run_rather_than_failing_over(
        self,
    ) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        with pytest.raises(GenerationUnavailable):
            await _answer(
                _instance(scope), dataset=dataset, index=index, generator=Generator(down=True)
            )

    async def test_an_answer_survives_its_own_progress_line(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        answer = await _answer(
            _instance(scope, category=WITHHELD),
            dataset=dataset,
            index=index,
            generator=Generator("x [C2] [C7]."),
            judged={"q0": frozenset({f"c-{scope[1]}"})},
        )
        assert an.Answer.from_json(json.loads(json.dumps(answer.to_json()))) == answer


# --- the run's bookkeeping -----------------------------------------------------------


class TestTheRulesReadEachAnswer:
    async def test_the_headings_are_the_lines_the_prompt_showed(self) -> None:
        dataset, index = _varied_world()
        generator = Generator()
        answer = await _answer(
            _instance(tuple(sorted(dataset.papers))[:10]),
            dataset=dataset,
            index=index,
            generator=generator,
        )
        (_system, user), *_ = generator.prompts
        shown = an.headings(answer, dataset)
        assert len(shown) == len(answer.passages) > 1
        for passage, heading in zip(answer.passages, shown, strict=True):
            assert f"[{passage.label}] {heading}" in user

    async def test_an_echoed_heading_is_read_as_the_label_of_the_text_below(self) -> None:
        dataset, index = _varied_world()
        scope = tuple(sorted(dataset.papers))[:10]
        first = await _answer(_instance(scope), dataset=dataset, index=index, generator=Generator())
        heading = an.headings(first, dataset)[0]
        echoed = await _answer(
            _instance(scope),
            dataset=dataset,
            index=index,
            generator=Generator(f"[C1] {heading}\nThe passage says something."),
        )
        reading = an.read_answer(echoed, dataset)
        assert [(c.text, c.labels) for c in reading.claims] == [
            ("The passage says something.", ("C1",))
        ]
        assert reading.refusal is Refusal.ANSWERED

    async def test_a_structural_refusal_reads_as_refused(self) -> None:
        dataset, index = _varied_world()
        answer = await _answer(
            _instance(("nowhere",)), dataset=dataset, index=index, generator=Generator()
        )
        assert an.read_answer(answer, dataset).refusal is Refusal.REFUSED
        # D60's flag decides, not its wording: the same row with other text is still refused.
        reworded = dataclasses.replace(answer, text="The model is a CNN.")
        assert an.read_answer(reworded, dataset).refusal is Refusal.REFUSED

    def test_a_reading_serialises_every_field_a_metric_needs(self) -> None:
        claims = split_claims("There is no mention of it. But [C2] says X holds.")
        record = an.Reading(claims, Refusal.PARTIAL).to_json()
        assert record["refusal"] == "partial"
        assert record["claims"][1] == {
            "index": 1,
            "source": "But [C2] says X holds.",
            "text": "But X holds.",
            "labels": ["C2"],
            "kind": "claim",
            "refuses": False,
        }


class TestAnInterruptedRunResumes:
    async def test_a_torn_last_line_is_forgiven(self, tmp_path: Path) -> None:
        dataset, index = _varied_world()
        answer = await _answer(
            _instance(tuple(sorted(dataset.papers))[:10]),
            dataset=dataset,
            index=index,
            generator=Generator(),
        )
        path = tmp_path / "progress.jsonl"
        path.write_text(json.dumps(answer.to_json()) + '\n{"key": "ans', encoding="utf-8")
        assert an.read_answers(path) == {answer.key: answer}

    def test_a_torn_line_in_the_middle_is_not(self, tmp_path: Path) -> None:
        path = tmp_path / "progress.jsonl"
        path.write_text('{"key": "ans\n{"key": "x"}\n', encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            an.read_answers(path)

    def test_no_file_means_nothing_done(self, tmp_path: Path) -> None:
        assert an.read_answers(tmp_path / "absent.jsonl") == {}


class TestThePlacementIsReadAfterTheFirstAnswer:
    async def _run(
        self,
        todo: Sequence[ab.Instance],
        done: dict[str, an.Answer],
        check: Any,
        generator: Generator | None = None,
    ) -> list[an.Answer]:
        dataset, index = _varied_world()
        written: list[an.Answer] = []
        await an.answer_all(
            todo,
            done=done,
            sink=written.append,
            after_first=check,
            dataset=dataset,
            index=index,
            lexical=index,
            embedder=CountingEmbedder(),
            generator=generator or Generator(),
            judged={},
        )
        return written

    async def test_it_runs_once_after_the_first_generated_answer(self) -> None:
        papers = tuple(f"p{i:02d}" for i in range(10))
        todo = [_instance(papers, qid=f"q{n}") for n in range(3)]
        calls: list[int] = []

        async def check() -> None:
            calls.append(len(calls))

        written = await self._run(todo, {}, check)
        assert calls == [0] and len(written) == 3

    async def test_a_failed_check_stops_the_run_at_one_answer(self) -> None:
        papers = tuple(f"p{i:02d}" for i in range(10))
        todo = [_instance(papers, qid=f"q{n}") for n in range(3)]

        async def on_cpu() -> None:
            raise RuntimeError("cpu")

        dataset, index = _varied_world()
        written: list[an.Answer] = []
        with pytest.raises(RuntimeError):
            await an.answer_all(
                todo,
                done={},
                sink=written.append,
                after_first=on_cpu,
                dataset=dataset,
                index=index,
                lexical=index,
                embedder=CountingEmbedder(),
                generator=Generator(),
                judged={},
            )
        assert len(written) == 1

    async def test_a_structural_refusal_does_not_count_as_the_model_loading(self) -> None:
        papers = tuple(f"p{i:02d}" for i in range(10))
        todo = [_instance(("nowhere",), qid="empty"), _instance(papers, qid="real")]
        generator = Generator()
        generated_before_check: list[int] = []

        async def check() -> None:
            generated_before_check.append(len(generator.prompts))

        await self._run(todo, {}, check, generator)
        # Read once, and only after the model was actually called.
        assert generated_before_check == [1]


class TestTheSummary:
    def _row(self, tokens: int | None) -> an.Answer:
        return an.Answer(
            key=f"k{tokens}",
            question_id="q",
            paper_id="p",
            split="train",
            category=ANS,
            question="?",
            scope=("p",),
            passages=(),
            messages_sha256="m",
            prompt_chars=10,
            text="t",
            shown="t",
            citations=(),
            provider="ollama",
            model="llama3.2",
            prompt_tokens=tokens,
            completion_tokens=1,
            retrieval_ms=1,
            generation_ms=1,
            structural_refusal=False,
            judged=0,
        )

    def test_a_prompt_that_fills_the_window_is_counted(self) -> None:
        rows = [self._row(4095), self._row(4096), self._row(None)]
        summary = an.summarise(rows, context_length=4096)
        assert summary["at_context_limit"] == 1
        assert summary["prompt_tokens_reported"] == 2

    def test_an_unknown_window_is_not_reported_as_no_truncation(self) -> None:
        assert an.summarise([self._row(9000)], context_length=None)["at_context_limit"] is None


# --- the run, end to end on stand-ins ----------------------------------------------


@pytest.fixture
def run_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Any]:
    dataset, index = _varied_world()
    embedder = CountingEmbedder()
    generator = Generator()
    sets_path = tmp_path / "sets.json"
    sets_path.write_text("{}", encoding="utf-8")
    archive = {
        "blocks": {
            cli.GATE_DENSE_BLOCK: {"collection": "eval", "embedder_version": embedder.version},
            cli.GATE_HYBRID_BLOCK: {
                "collection": "eval_hybrid",
                "embedder_version": embedder.version,
            },
        }
    }
    state = {
        "ollama_version": "0.40.11",
        "digest": "a80c4f17acd5",
        "placement": "gpu",
        "context_length": 4096,
    }
    states: list[dict[str, Any]] = []

    def build(_dataset: DRet, _spec: Any, *, sparse: bool = False) -> tuple[Any, Any, str]:
        return embedder, index, "eval_hybrid" if sparse else "eval"

    async def ollama(_generator: Any) -> dict[str, Any]:
        states.append(dict(state))
        return dict(state)

    monkeypatch.setattr(cli, "_load_reference", lambda _path: archive)
    monkeypatch.setattr(cli, "load_sets", lambda: _sets_for(dataset))
    monkeypatch.setattr(answerability, "SETS", sets_path)
    monkeypatch.setattr(cli, "_build_dependencies", build)
    monkeypatch.setattr(cli, "_build_generator", lambda: generator)
    monkeypatch.setattr(cli, "_ollama_state", ollama)
    monkeypatch.setattr(cli, "load_snapshot", lambda: [])
    monkeypatch.setattr(cli, "build_dret", lambda _rows: dataset)
    monkeypatch.setattr(cli, "verify_split", lambda _d: None)
    monkeypatch.setattr(cli, "verify_core", lambda _d: None)
    # The real checker would load the NLI model in a unit test; CI has no torch.
    checker = Checker()
    monkeypatch.setattr(cli, "_build_nli", lambda: checker)
    return {
        "index": index,
        "dir": tmp_path,
        "generator": generator,
        "state": state,
        "states": states,
        "checker": checker,
        "dataset": dataset,
    }


class Checker:
    """An NLI checker that records every pair and supports all of them."""

    version = "stand-in-nli/1"

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    async def entails(self, premises: Sequence[str], claims: Sequence[str]) -> list[Entailment]:
        self.pairs.extend(zip(premises, claims, strict=True))
        return [Entailment(0.8, 0.1, 0.1, False) for _ in premises]


class TestTheRun:
    async def test_it_answers_the_train_sample_and_reads_the_store_only(
        self, run_world: dict[str, Any]
    ) -> None:
        report = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        index: ScoreIndex = run_world["index"]
        assert "ensure_ready" not in index.calls
        assert index.calls.count("existing_count") == 1
        # 4 answerable + 1 QASPER + 1 withheld train questions in the stand-in sets.
        rows = report.blocks["answers"]["rows"]
        assert len(rows) == 6 and {r["split"] for r in rows} == {"train"}
        assert len(run_world["generator"].prompts) == 6
        assert report.config["split"] == "train-dev"
        assert report.blocks["generator"]["context_length"] == 4096
        json.loads(report.to_json())

    async def test_a_rerun_resumes_from_its_progress_file(self, run_world: dict[str, Any]) -> None:
        first = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        run_world["generator"].prompts.clear()
        second = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert second.config["resumed_rows"] == 6
        assert run_world["generator"].prompts == []
        assert second.blocks["answers"] == first.blocks["answers"]

    async def test_another_model_digest_does_not_resume_the_old_answers(
        self, run_world: dict[str, Any]
    ) -> None:
        await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        run_world["state"]["digest"] = "b" * 12
        again = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert again.config["resumed_rows"] == 0

    async def test_train_and_eval_answers_never_share_a_progress_file(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        monkeypatch.setattr(cli, "require_frozen_rules", lambda: None)
        held = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        assert held.config["progress_file"] != dev.config["progress_file"]
        assert {r["split"] for r in held.blocks["answers"]["rows"]} == {"eval"}
        assert held.config["split"] == "eval"

    async def test_an_eval_run_measures_and_archives_every_pair_it_read(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # D179: every pair's premise, hypothesis and probabilities are archived, so
        # each rate can be recomputed without Qdrant, Ollama or the model.
        monkeypatch.setattr(cli, "require_frozen_rules", lambda: None)
        report = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        checker: Checker = run_world["checker"]
        rows = report.blocks["answers"]["rows"]
        archived = [pair for row in rows for pair in row["pairs"]]
        assert archived and len(archived) == len(checker.pairs)
        dataset: DRet = run_world["dataset"]
        for pair in archived:
            assert pair["premise"] == dataset.chunks[pair["chunk_id"]].text
            assert (pair["premise"], pair["hypothesis"]) in checker.pairs
            assert pair["supported"] is True
        grounding = report.blocks["grounding"]
        assert grounding["nli"] == "stand-in-nli/1"
        assert grounding["totals"]["pairs"] == len(archived)
        assert grounding["rates"]["answerable/groundedness"]["point"] == 1.0
        assert grounding["rates"]["papers"]["seed"] == 20260915
        json.loads(report.to_json())

    async def test_regenerated_answers_are_reported_but_never_stored(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(cli, "require_frozen_rules", lambda: None)
        report = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        answered = len(report.blocks["answers"]["rows"])
        regen = report.blocks["regeneration"]
        assert regen["regenerated"] == min(20, answered)
        # The stand-in answers identically every time.
        assert (regen["text_changed"], regen["refusal_label_changed"]) == (0, 0)
        assert len(run_world["generator"].prompts) == answered + regen["regenerated"]
        stored = an.read_answers(Path(report.config["progress_file"]))
        assert len(stored) == answered

    async def test_a_resumed_run_still_regenerates_instead_of_reusing(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A resumed run finds every answer stored. Handing those to the
        # regeneration would compare each answer with itself and report zero
        # variation that was never measured.
        monkeypatch.setattr(cli, "require_frozen_rules", lambda: None)
        await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        before = len(run_world["generator"].prompts)
        resumed = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        assert resumed.config["resumed_rows"] == len(resumed.blocks["answers"]["rows"])
        regenerated = resumed.blocks["regeneration"]["regenerated"]
        assert regenerated > 0
        assert len(run_world["generator"].prompts) - before == regenerated

    async def test_a_dev_run_reads_no_pairs_and_regenerates_nothing(
        self, run_world: dict[str, Any]
    ) -> None:
        report = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert run_world["checker"].pairs == []
        assert "grounding" not in report.blocks and "regeneration" not in report.blocks
        assert all("pairs" not in row for row in report.blocks["answers"]["rows"])

    async def test_a_model_off_the_gpu_stops_the_run_after_one_answer(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["state"]["placement"] = "cpu"
        with pytest.raises(RuntimeError, match="not the GPU"):
            await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert len(run_world["generator"].prompts) == 1

    async def test_a_model_that_is_not_pulled_is_refused_before_any_answer(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["state"]["digest"] = None
        with pytest.raises(ValueError, match="not pulled"):
            await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert run_world["generator"].prompts == []

    async def test_a_model_that_changes_during_the_run_is_refused(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = 0

        async def drifting(_generator: Any) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {**run_world["state"], "digest": "first" if calls == 1 else "second"}

        monkeypatch.setattr(cli, "_ollama_state", drifting)
        with pytest.raises(ValueError, match="digest changed"):
            await cli.run_answers(dev=True, progress_dir=run_world["dir"])

    async def test_every_row_carries_rule_1s_label_and_rule_2s_sentences(
        self, run_world: dict[str, Any]
    ) -> None:
        report = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        rows = report.blocks["answers"]["rows"]
        assert {r["refusal"] for r in rows} == {"answered"}
        assert [c["labels"] for c in rows[0]["claims"]] == [["C1"]]
        rules = report.blocks["rules"]
        assert set(rules["rules"]) == {"claims/1.0.0", "refusal/1.0.0"}
        assert sum(n for by in rules["refusal_by_category"].values() for n in by.values()) == 6
        assert "rule 1" in cli._render_answers(report)

    async def test_eval_answers_belong_to_the_rule_versions_they_were_made_under(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dev = await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        monkeypatch.setattr(cli, "require_frozen_rules", lambda: None)
        held = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        assert "claims_version" not in dev.config
        assert (held.config["claims_version"], held.config["refusal_version"]) == (
            "claims/1.0.0",
            "refusal/1.0.0",
        )
        first = held.config["progress_file"]
        monkeypatch.setattr(cli, "REFUSAL_VERSION", "refusal/1.0.1")
        again = await cli.run_answers(dev=False, progress_dir=run_world["dir"])
        assert again.config["progress_file"] != first and again.config["resumed_rows"] == 0

    async def test_a_partial_collection_is_refused_without_being_built(
        self, run_world: dict[str, Any]
    ) -> None:
        run_world["index"].points = 7
        with pytest.raises(ValueError, match="holds 7 of 30 chunks"):
            await cli.run_answers(dev=True, progress_dir=run_world["dir"])
        assert "ensure_ready" not in run_world["index"].calls


class TestTheCommand:
    def test_dev_without_answers_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--dev"])

    def test_dense_beside_answers_is_refused(self) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--answers", "--dense"])

    def test_plain_answers_is_refused_until_the_rules_are_frozen(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(an, "FROZEN_RULE_HASHES", {})
        with pytest.raises(an.RulesNotFrozen):
            cli.main(["--answers", "--no-archive"])

    def test_train_answers_archive_as_their_own_kind(
        self, run_world: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reports = run_world["dir"] / "reports"
        original = cli.run_answers

        async def here(*, dev: bool) -> cli.EvalReport:
            return await original(dev=dev, progress_dir=run_world["dir"])

        monkeypatch.setattr(cli, "run_answers", here)
        assert cli.main(["--answers", "--dev", "--report-dir", str(reports)]) == 0
        (archived,) = reports.glob("answers-dev-*.json")
        assert json.loads(archived.read_text(encoding="utf-8"))["config"]["split"] == "train-dev"
