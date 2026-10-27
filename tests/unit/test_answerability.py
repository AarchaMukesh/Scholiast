"""D177's labels, candidates and withheld scopes, against the vendored snapshot.

The counts are the ones D177 pre-registered from a read-only pass over
`qasper-validation.json.gz`; a change to the labelling rule, the split or the
snapshot that moved them would move every number fitted on these sets, so it
fails here first.
"""

from __future__ import annotations

import csv
import io
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


# --- step 2: verdicts and the versioned sets --------------------------------------


@pytest.fixture(scope="module")
def sheet(rows: list[dict[str, Any]], dataset: DRet) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(sets.sheet_json(sets.candidate_sheet(rows, dataset)))
    return result


def verdict_csv(sheet: dict[str, Any], overrides: dict[int, dict[str, str]] | None = None) -> str:
    """A completed verdict file: every candidate `keep`, then any row overridden."""
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\r\n")
    writer.writerow(sets.VERDICT_COLUMNS)
    for number, candidate in enumerate(sheet["candidates"], start=1):
        row = {
            "number": str(number),
            "question_id": candidate["question_id"],
            "split": candidate["split"],
            "question": candidate["question"],
            "verdict": "keep",
            "note": "",
            "sheet_sha256": sheet["sheet_sha256"],
        }
        row |= (overrides or {}).get(number, {})
        writer.writerow([row[column] for column in sets.VERDICT_COLUMNS])
    return "\ufeff" + out.getvalue()


def problems_of(text: str, sheet: dict[str, Any]) -> tuple[str, ...]:
    with pytest.raises(sets.VerdictFileInvalid) as refused:
        sets.read_verdicts(text, sheet)
    return refused.value.problems


class TestTheVerdictsMustMatchTheSheetExactly:
    def test_a_complete_file_is_read_with_its_bom(self, sheet: dict[str, Any]) -> None:
        verdicts = sets.read_verdicts(verdict_csv(sheet), sheet)
        assert len(verdicts) == 100 and {v.verdict for v in verdicts} == {"keep"}

    def test_a_verdict_spelled_differently_is_refused_not_repaired(
        self, sheet: dict[str, Any]
    ) -> None:
        # " Keep" means keep to a person and nothing to a rule; repairing it would
        # be a label nobody made.
        found = problems_of(verdict_csv(sheet, {5: {"verdict": "Keep"}}), sheet)
        assert any("row 5: verdict 'Keep'" in p for p in found)

    def test_an_exclusion_naming_a_paper_outside_its_scope_is_refused(
        self, sheet: dict[str, Any]
    ) -> None:
        # What the owner's first pass did on four rows: the note named the
        # question's own paper, seen in another candidate's scope.
        own_elsewhere = sheet["candidates"][1]["scope"][0]["paper_id"]
        mine = {p["paper_id"] for p in sheet["candidates"][0]["scope"]}
        assert own_elsewhere not in mine
        override = {1: {"verdict": "exclude (ii)", "note": own_elsewhere}}
        found = problems_of(verdict_csv(sheet, override), sheet)
        assert any("names no paper from its scope" in p for p in found)
        assert any("outside its scope" in p for p in found)

    def test_an_exclusion_naming_its_own_scope_paper_is_accepted(
        self, sheet: dict[str, Any]
    ) -> None:
        mine = sheet["candidates"][0]["scope"][3]["paper_id"]
        override = {1: {"verdict": "exclude (ii)", "note": f"covered by {mine}"}}
        verdicts = sets.read_verdicts(verdict_csv(sheet, override), sheet)
        assert verdicts[0].verdict == "exclude (ii)" and verdicts[0].note == f"covered by {mine}"

    @pytest.mark.parametrize(
        ("override", "column"),
        [
            ({"question": "an edited question"}, "question"),
            ({"sheet_sha256": "0" * 64}, "sheet_sha256"),
            ({"split": "train"}, "split"),
            ({"question_id": "someone-else"}, "question_id"),
        ],
    )
    def test_a_row_that_differs_from_the_sheet_is_refused(
        self, sheet: dict[str, Any], override: dict[str, str], column: str
    ) -> None:
        row = next(i for i, c in enumerate(sheet["candidates"], 1) if c["split"] == "eval")
        found = problems_of(verdict_csv(sheet, {row: override}), sheet)
        assert any(f"row {row}: {column}" in p for p in found)

    def test_a_missing_row_is_refused(self, sheet: dict[str, Any]) -> None:
        lines = verdict_csv(sheet).split("\r\n")
        short = "\r\n".join([*lines[:-2], ""])
        assert any("99 rows" in p for p in problems_of(short, sheet))

    def test_every_problem_is_reported_at_once(self, sheet: dict[str, Any]) -> None:
        override = {2: {"verdict": "skip"}, 7: {"verdict": "KEEP"}}
        assert len(problems_of(verdict_csv(sheet, override), sheet)) == 2

    def test_a_tampered_sheet_is_refused_before_any_row_is_read(
        self, sheet: dict[str, Any]
    ) -> None:
        tampered = sheet | {"candidates": sheet["candidates"][1:]}
        assert problems_of(verdict_csv(sheet), tampered) == (
            "the sheet's stated sha256 does not match its own candidates",
        )


@pytest.fixture(scope="module")
def built(rows: list[dict[str, Any]], dataset: DRet, sheet: dict[str, Any]) -> dict[str, Any]:
    verdicts = sets.read_verdicts(verdict_csv(sheet), sheet)
    return sets.build_sets(rows, dataset, verdicts, sheet_sha256=sheet["sheet_sha256"])


class TestTheVersionedSets:
    def test_the_counts_are_d177s_with_every_candidate_kept(self, built: dict[str, Any]) -> None:
        assert built["counts"] == {
            "eval/answerable": 243,
            "eval/unanswerable-qasper": 9,
            "eval/unanswerable-withheld": 40,
            "train/answerable": 624,
            "train/unanswerable-qasper": 47,
            "train/unanswerable-withheld": 60,
        }
        assert built["left_out"] == {"disagree": 75, "outside_dret": 7, "withheld_excluded": 0}
        assert built["set_version"] == "d-ans-unans/1.0.0"

    def test_a_withheld_entry_carries_its_scope_without_its_own_paper(
        self, built: dict[str, Any]
    ) -> None:
        withheld = [q for q in built["questions"] if q["category"] == "unanswerable-withheld"]
        for entry in withheld:
            assert entry["withheld_paper"] == entry["paper_id"]
            assert len(entry["scope"]) == 10 and entry["paper_id"] not in entry["scope"]

    def test_an_exclusion_leaves_the_set_and_is_recorded(
        self, rows: list[dict[str, Any]], dataset: DRet, sheet: dict[str, Any]
    ) -> None:
        mine = sheet["candidates"][0]["scope"][0]["paper_id"]
        override = {1: {"verdict": "exclude (ii)", "note": mine}}
        verdicts = sets.read_verdicts(verdict_csv(sheet, override), sheet)
        built = sets.build_sets(rows, dataset, verdicts, sheet_sha256=sheet["sheet_sha256"])
        assert built["counts"]["train/unanswerable-withheld"] == 59
        (excluded,) = built["withheld_excluded"]
        assert excluded["question_id"] == sheet["candidates"][0]["question_id"]
        assert excluded["note"] == mine

    def test_verdicts_for_other_candidates_are_refused(
        self, rows: list[dict[str, Any]], dataset: DRet, sheet: dict[str, Any]
    ) -> None:
        verdicts = sets.read_verdicts(verdict_csv(sheet), sheet)
        with pytest.raises(ValueError, match="not for this snapshot"):
            sets.build_sets(rows, dataset, verdicts[1:], sheet_sha256=sheet["sheet_sha256"])

    def test_the_written_file_loads_and_an_edited_one_is_refused(
        self, built: dict[str, Any], tmp_path: Path
    ) -> None:
        path, manifest = tmp_path / "s.json", tmp_path / "s.manifest.json"
        record = sets.write_sets(
            built,
            labeller="owner",
            labelled_on="2026-10-27",
            verdicts_sha256="v",
            path=path,
            manifest=manifest,
        )
        assert record["labeller"] == "owner" and record["counts"] == built["counts"]
        assert sets.load_sets(path, manifest) == built
        path.write_bytes(path.read_bytes().replace(b'"keep"', b'"kept"', 1) + b" ")
        with pytest.raises(ValueError, match="was edited"):
            sets.load_sets(path, manifest)

    def test_the_command_builds_from_a_valid_file_and_refuses_an_invalid_one(
        self, sheet: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sheet_dir = tmp_path / "labelling"
        sheet_dir.mkdir()
        (sheet_dir / "d177-withheld-candidates.json").write_text(json.dumps(sheet), "utf-8")
        monkeypatch.setattr(sets, "SETS", tmp_path / "sets.json")
        monkeypatch.setattr(sets, "SETS_MANIFEST", tmp_path / "sets.manifest.json")
        good = tmp_path / "good.csv"
        good.write_text(verdict_csv(sheet), encoding="utf-8")
        argv = ["--build-sets", "--sheet-dir", str(sheet_dir), "--labeller", "owner"]
        argv += ["--labelled-on", "2026-10-27", "--verdicts"]
        assert cli.main([*argv, str(good)]) == 0
        assert sets.load_sets()["counts"]["eval/unanswerable-withheld"] == 40
        bad = tmp_path / "bad.csv"
        bad.write_text(verdict_csv(sheet, {3: {"verdict": "maybe"}}), encoding="utf-8")
        (tmp_path / "sets.json").unlink()
        with pytest.raises(sets.VerdictFileInvalid):
            cli.main([*argv, str(bad)])
        assert not (tmp_path / "sets.json").exists()

    def test_the_command_needs_its_labeller_and_date(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            cli.main(["--build-sets", "--verdicts", str(tmp_path / "x.csv")])
