"""D177's labels, candidates and withheld scopes, against the vendored snapshot.

The counts are the ones D177 pre-registered from a read-only pass over
`qasper-validation.json.gz`; a change to the labelling rule, the split or the
snapshot that moved them would move every number fitted on these sets, so it
fails here first.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scholiast.eval import answerability as sets
from scholiast.eval import cli
from scholiast.eval.answerability import Answerability, LabelledSet
from scholiast.eval.dret import DRet, Split, build_dret
from scholiast.eval.gate import scope_for
from scholiast.eval.snapshot import load_snapshot


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return load_snapshot()


@pytest.fixture(scope="module")
def dataset(rows: list[dict[str, Any]]) -> DRet:
    return build_dret(rows)


@pytest.fixture(scope="module")
def labelled(rows: list[dict[str, Any]], dataset: DRet) -> LabelledSet:
    return sets.label_questions(rows, dataset)


def annotation(*, unanswerable: bool) -> dict[str, Any]:
    return {"unanswerable": unanswerable}


class TestTheLabelsAreQaspersOwn:
    def test_every_annotator_answering_is_answerable(self) -> None:
        anns = [annotation(unanswerable=False)] * 2
        assert sets.answerability(anns) is Answerability.ANSWERABLE

    def test_every_annotator_declining_is_unanswerable(self) -> None:
        anns = [annotation(unanswerable=True)] * 3
        assert sets.answerability(anns) is Answerability.UNANSWERABLE

    def test_one_dissent_is_a_disagreement_not_a_majority(self) -> None:
        # A majority vote would turn 2-against-1 into a label; D177 excludes it.
        anns = [annotation(unanswerable=True)] * 2 + [annotation(unanswerable=False)]
        assert sets.answerability(anns) is Answerability.DISAGREE

    def test_a_question_with_no_annotations_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no annotations"):
            sets.answerability([])


class TestTheSetsMatchWhatD177PreRegistered:
    def test_the_counts_on_each_split(self, labelled: LabelledSet) -> None:
        assert labelled.counts() == {
            "train": {"answerable": 624, "unanswerable": 47, "disagree": 51},
            "eval": {"answerable": 243, "unanswerable": 9, "disagree": 24},
        }

    def test_questions_on_papers_outside_d_ret_are_excluded_and_counted(
        self, labelled: LabelledSet
    ) -> None:
        assert labelled.outside_dret == 7

    def test_every_question_sits_on_its_papers_retrieval_split(
        self, labelled: LabelledSet, dataset: DRet
    ) -> None:
        # A question on the other side of the line from its paper's retrieval
        # queries would let tau be fitted on a paper it is then reported on.
        split_of = {q.paper_id: q.split for q in dataset.queries}
        assert all(q.split is split_of[q.paper_id] for q in labelled.questions)


class TestTheWithheldCandidates:
    def test_sixty_train_and_forty_eval_all_answerable(self, labelled: LabelledSet) -> None:
        chosen = sets.withheld_candidates(labelled)
        assert sum(q.split is Split.TRAIN for q in chosen) == 60
        assert sum(q.split is Split.EVAL for q in chosen) == 40
        assert all(q.label is Answerability.ANSWERABLE for q in chosen)

    def test_the_draw_reads_only_the_id(self, labelled: LabelledSet) -> None:
        # Reordering the questions draws the same candidates: selection by hash,
        # not by position, so nothing about the corpus listing can steer it.
        shuffled = LabelledSet(questions=list(reversed(labelled.questions)))
        first = [q.question_id for q in sets.withheld_candidates(labelled)]
        assert [q.question_id for q in sets.withheld_candidates(shuffled)] == first

    def test_the_draw_is_d177s_hash(self, labelled: LabelledSet) -> None:
        # Pinned from `sha256("d177-candidate:{question_id}")` when D177 was
        # registered. Any other key or order still draws 60 and 40 answerable
        # questions, and only these ids tell the difference.
        chosen = sets.withheld_candidates(labelled)
        train = [q.question_id for q in chosen if q.split is Split.TRAIN][:2]
        eval_ = [q.question_id for q in chosen if q.split is Split.EVAL][:2]
        assert train == [
            "af5730d82535464cedfa707a03415ac2e7a21295",
            "c9bc6f53b941863e801280343afa14248521ce43",
        ]
        assert eval_ == [
            "f03112b868b658c954db62fc64430bebbaa7d9e0",
            "e831041d50f3922265330fcbee5a980d0e2586dd",
        ]

    def test_a_split_with_too_few_answerable_questions_is_refused(self) -> None:
        with pytest.raises(ValueError, match="answerable questions"):
            sets.withheld_candidates(LabelledSet())


class TestTheWithheldScope:
    PAPERS = tuple(f"p{n:02d}" for n in range(30))

    def test_it_never_holds_the_questions_own_paper(self) -> None:
        scope = sets.withheld_scope("q1", "p07", self.PAPERS)
        assert "p07" not in scope and len(scope) == sets.WITHHELD_SCOPE_SIZE

    def test_it_is_independent_of_corpus_order(self) -> None:
        forward = sets.withheld_scope("q1", "p07", self.PAPERS)
        assert sets.withheld_scope("q1", "p07", list(reversed(self.PAPERS))) == forward

    def test_two_questions_on_one_paper_get_different_scopes(self) -> None:
        assert sets.withheld_scope("q1", "p07", self.PAPERS) != sets.withheld_scope(
            "q2", "p07", self.PAPERS
        )

    def test_it_is_not_the_gates_scope_with_the_paper_removed(self) -> None:
        # D177's own key: a withheld scope that was the gate's draw minus one
        # paper would ask the question beside the same distractors it was
        # answered beside, and the two categories would not be independent.
        gate = set(scope_for("q1", "p07", self.PAPERS, size=11, draw=0)) - {"p07"}
        assert set(sets.withheld_scope("q1", "p07", self.PAPERS)) != gate

    def test_the_scope_is_d177s_hash(self, dataset: DRet) -> None:
        # The first train candidate's scope, pinned from
        # `sha256("d177-withheld:{question_id}:{paper}")` over the real corpus.
        scope = sets.withheld_scope(
            "af5730d82535464cedfa707a03415ac2e7a21295", "1910.14599", sorted(dataset.papers)
        )
        assert scope == (
            "1610.07809", "1703.07476", "1706.09673", "1901.05287", "1909.02304",
            "1909.10012", "1910.05603", "1912.01852", "2001.05865", "2002.01320",
        )  # fmt: skip

    def test_a_paper_missing_from_the_corpus_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the corpus"):
            sets.withheld_scope("q1", "absent", self.PAPERS)


class TestTheSheet:
    def test_each_candidate_shows_its_scope_and_never_its_own_paper(
        self, rows: list[dict[str, Any]], dataset: DRet, labelled: LabelledSet
    ) -> None:
        entries = sets.candidate_sheet(rows, dataset)
        own = {q.question_id: q.paper_id for q in sets.withheld_candidates(labelled)}
        assert len(entries) == 100
        for entry in entries:
            assert len(entry.scope) == sets.WITHHELD_SCOPE_SIZE
            assert own[entry.question_id] not in {p.paper_id for p in entry.scope}
            assert all(p.title and p.abstract for p in entry.scope)

    def test_its_hash_moves_when_a_candidate_does(
        self, rows: list[dict[str, Any]], dataset: DRet
    ) -> None:
        # The hash binds labels to the questions judged; one that ignored an
        # edit would let labels attach to a candidate nobody looked at.
        entries = sets.candidate_sheet(rows, dataset)
        digest = json.loads(sets.sheet_json(entries))["sheet_sha256"]
        assert json.loads(sets.sheet_json(entries))["sheet_sha256"] == digest
        assert json.loads(sets.sheet_json(entries[1:]))["sheet_sha256"] != digest

    def test_the_command_writes_both_forms_and_archives_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        sheet_dir, report_dir = tmp_path / "sheet", tmp_path / "reports"
        argv = ["--withheld-sheet", "--sheet-dir", str(sheet_dir), "--report-dir", str(report_dir)]
        assert cli.main(argv) == 0
        assert sorted(p.name for p in sheet_dir.iterdir()) == [
            "d177-withheld-candidates.json",
            "d177-withheld-candidates.md",
        ]
        assert not report_dir.exists()
        written = json.loads((sheet_dir / "d177-withheld-candidates.json").read_text("utf-8"))
        assert written["sheet_sha256"] in capsys.readouterr().out
        assert written["set_version"] == sets.SET_VERSION

    def test_it_prints_d177s_clarified_rule_and_verdicts(
        self, rows: list[dict[str, Any]], dataset: DRet
    ) -> None:
        # The labeller reads the rule from the sheet; a sheet still carrying the
        # literal reading D177 rejected would label a different category.
        entries = sets.candidate_sheet(rows, dataset)
        sheet = sets.sheet_markdown(entries, sheet_sha256="x")
        assert "\n".join(sets.INSTRUCTIONS) in sheet
        assert "Do not exclude a question for being generic" in sheet
        assert sheet.count("keep / exclude (i) / exclude (ii)") == len(entries)

    def test_dense_beside_the_sheet_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--dense", "--withheld-sheet", "--sheet-dir", str(tmp_path)])
