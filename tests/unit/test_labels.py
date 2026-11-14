"""D179's human labels: blind sheets, strict reading, the two bars (M5 phase 2, step B3).

Labels decide bar 1 and gate bar 2, so the failures that matter are the ones that
would let a number stand on labels nobody made, or on labels made while seeing
the answer key: a sheet that leaks Rule 1 or the NLI reading, a label file that
is "repaired" into validity, labels scored against answers they never read, a
bar applied on the wrong side of its boundary, a label file overwritten.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from scholiast.eval import cli
from scholiast.eval import labels as lb
from scholiast.eval.metrics import cohen_kappa


def row(
    n: int,
    *,
    category: str = "answerable",
    refusal: str = "answered",
    paper: str | None = None,
    pairs: Sequence[tuple[int, str, bool]] = (),
    split: str = "eval",
) -> dict[str, Any]:
    return {
        "key": f"{category}:q{n}:10:0",
        "question_id": f"q{n}",
        "paper_id": paper or f"p{n}",
        "split": split,
        "category": category,
        "question": f"question {n}?",
        "shown": f"answer {n} [C1].",
        "refusal": refusal,
        "claims": [
            {"index": i, "source": f"claim {n}.{i} [{label}]", "kind": "claim"}
            for i, label, _ in pairs
        ],
        "pairs": [
            {
                "claim_index": i,
                "label": label,
                "chunk_id": f"k{n}.{i}",
                "premise": f"passage {n}.{i}",
                "hypothesis": f"claim {n}.{i}",
                "entailment": 0.9 if supported else 0.1,
                "supported": supported,
            }
            for i, label, supported in pairs
        ],
    }


def world() -> list[dict[str, Any]]:
    """49 unanswerable and 120 answerable eval answers, with pairs on the answerable."""
    rows = [
        row(i, category="unanswerable-withheld" if i % 5 else "unanswerable-qasper")
        for i in range(49)
    ]
    rows += [row(100 + i, pairs=[(0, "C1", i % 2 == 0), (1, "C2", i % 3 == 0)]) for i in range(120)]
    return rows


def sheets(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    a = json.loads(
        lb.sheet_json("answers", lb.answer_entries(rows), archive="x", archive_sha256="h")
    )
    b = json.loads(lb.sheet_json("pairs", lb.pair_entries(rows), archive="x", archive_sha256="h"))
    return a, b


def filled(sheet: dict[str, Any], labels: Sequence[str]) -> str:
    """The CSV template with its label column filled in, as the owner would."""
    lines = lb.csv_template(sheet).removeprefix("﻿").splitlines()
    out = [lines[0]]
    for line, label in zip(lines[1:], labels, strict=True):
        cells = line.split(",")
        cells[-3] = label
        out.append(",".join(cells))
    return "﻿" + "\n".join(out) + "\n"


class TestKappa:
    def test_perfect_agreement_is_one(self) -> None:
        assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)

    def test_a_skewed_label_set_cannot_hide_behind_raw_agreement(self) -> None:
        # 90% raw agreement, but both say "x" nearly always: kappa is near zero.
        first = ["x"] * 9 + ["y"]
        second = ["x"] * 10
        assert cohen_kappa(first, second) == pytest.approx(0.0)

    def test_one_shared_label_everywhere_is_undefined_not_perfect(self) -> None:
        assert cohen_kappa(["x", "x"], ["x", "x"]) is None

    def test_a_textbook_value(self) -> None:
        # po = 0.7, pe = 0.5*0.6 + 0.5*0.4 = 0.5, kappa = 0.4.
        first = ["y"] * 5 + ["n"] * 5
        second = ["y", "y", "y", "y", "n", "y", "y", "n", "n", "n"]
        assert cohen_kappa(first, second) == pytest.approx(0.4)

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError):
            cohen_kappa(["a"], ["a", "b"])


class TestTheSheetsAreBlind:
    def test_sheet_a_holds_every_unanswerable_answer_and_51_answerable(self) -> None:
        rows = world()
        entries = lb.answer_entries(rows)
        by_key = {lb.item_id(r["key"]): r for r in rows}
        categories = [by_key[e["item"]]["category"] for e in entries]
        assert len(entries) == 100
        assert sum(c != "answerable" for c in categories) == 49
        assert sum(c == "answerable" for c in categories) == lb.ANSWERABLE_LABELLED

    def test_neither_sheet_carries_the_answer_key(self) -> None:
        # D179: blind to both rules and to the NLI output; and the category is
        # not printed, so "should have refused" cannot be read off the sheet.
        a, b = sheets(world())
        text = json.dumps(a) + json.dumps(b) + lb.sheet_markdown(a) + lb.sheet_markdown(b)
        for leak in ("refusal", "answered", "supported", "entailment", "unanswerable", "category"):
            assert leak not in json.dumps(a["entries"]) + json.dumps(b["entries"])
        assert "supported" not in text and "entailment" not in text

    def test_no_sheet_row_names_its_category_or_question_id(self) -> None:
        # An answer's key starts with its category, and a withheld question shares
        # its question_id with an answerable one: either would unblind the sheet.
        rows = world()
        a, b = sheets(rows)
        printed = (
            json.dumps(a["entries"])
            + json.dumps(b["entries"])
            + lb.csv_template(a)
            + lb.csv_template(b)
        )
        for r in rows:
            assert r["key"] not in printed and f'"{r["question_id"]}"' not in printed
        assert "unanswerable" not in printed and "answerable:" not in printed

    def test_sheet_a_mixes_categories_rather_than_grouping_them(self) -> None:
        rows = world()
        by_key = {lb.item_id(r["key"]): r for r in rows}
        order = [by_key[e["item"]]["category"] == "answerable" for e in lb.answer_entries(rows)]
        assert order != sorted(order) and order != sorted(order, reverse=True)

    def test_sheet_b_draws_100_pairs_from_answerable_answers_only(self) -> None:
        rows = world()
        rows[0]["pairs"] = [{"claim_index": 0, "label": "C1", "chunk_id": "z", "premise": "p"}]
        entries = lb.pair_entries(rows)
        assert len(entries) == lb.PAIRS_LABELLED
        by_item = {lb.item_id(r["key"]): r for r in rows}
        assert all(by_item[e["item"]]["category"] == "answerable" for e in entries)

    def test_sheet_b_shows_the_claim_as_written_and_the_cited_passage(self) -> None:
        entry = lb.pair_entries(world())[0]
        n, i = entry["chunk_id"][1:].split(".")
        assert entry["claim"] == f"claim {n}.{i} [{entry['citation']}]"
        assert entry["passage"] == f"passage {n}.{i}"

    def test_the_draws_do_not_depend_on_the_archives_order(self) -> None:
        rows = world()
        assert sheets(rows) == sheets(list(reversed(rows)))

    def test_train_answers_are_refused(self) -> None:
        with pytest.raises(ValueError, match="eval answers only"):
            lb.answer_entries([row(1, split="train")])


class TestLabelFilesAreReadStrictly:
    def test_a_filled_template_reads_back_in_sheet_order(self) -> None:
        a, _ = sheets(world())
        labels = ["refused", "partial", "answered", "answered"] * 25
        assert lb.read_labels(filled(a, labels), a) == labels

    def test_the_template_is_prefilled_except_the_labels(self) -> None:
        a, _ = sheets(world())
        template = lb.csv_template(a)
        assert template.startswith("﻿")
        first = template.splitlines()[1].split(",")
        assert first == ["1", a["entries"][0]["item"], "", "", a["sheet_sha256"]]

    def test_a_blank_or_misspelt_label_is_refused_not_repaired(self) -> None:
        a, _ = sheets(world())
        labels = ["answered"] * 100
        labels[3], labels[7] = "", "Refused"
        with pytest.raises(lb.LabelFileInvalid) as raised:
            lb.read_labels(filled(a, labels), a)
        assert len(raised.value.problems) == 2  # both listed in one pass

    def test_rows_out_of_order_are_refused(self) -> None:
        a, _ = sheets(world())
        lines = filled(a, ["answered"] * 100).splitlines()
        lines[1], lines[2] = lines[2], lines[1]
        with pytest.raises(lb.LabelFileInvalid, match="does not match the sheet"):
            lb.read_labels("\n".join(lines), a)

    def test_labels_quoting_another_sheets_hash_are_refused(self) -> None:
        a, _ = sheets(world())
        text = filled(a, ["answered"] * 100).replace(a["sheet_sha256"], "0" * 64)
        with pytest.raises(lb.LabelFileInvalid, match="sheet_sha256"):
            lb.read_labels(text, a)

    def test_a_sheet_edited_after_hashing_is_refused(self) -> None:
        a, _ = sheets(world())
        text = filled(a, ["answered"] * 100)
        a["entries"][0]["question"] = "edited"
        with pytest.raises(lb.LabelFileInvalid, match="does not match its own entries"):
            lb.read_labels(text, a)

    def test_a_missing_row_is_refused(self) -> None:
        a, _ = sheets(world())
        lines = filled(a, ["answered"] * 100).splitlines()
        with pytest.raises(lb.LabelFileInvalid, match="99 rows"):
            lb.read_labels("\n".join(lines[:-1]), a)

    def test_pair_labels_check_their_pair_identity(self) -> None:
        _, b = sheets(world())
        text = filled(b, ["supports"] * 100)
        assert lb.read_labels(text, b) == ["supports"] * 100
        tampered = text.replace(f",{b['entries'][0]['citation']},", ",C9,", 1)
        with pytest.raises(lb.LabelFileInvalid, match="citation"):
            lb.read_labels(tampered, b)


def scored(
    rows: list[dict[str, Any]],
    answer_labels: dict[str, str] | None = None,
    pair_label: Any = None,
) -> dict[str, Any]:
    """Score with labels chosen per key (default: the owner agrees with Rule 1)."""
    a, b = sheets(rows)
    by_key = {lb.item_id(r["key"]): r for r in rows}
    a_labels = [
        (answer_labels or {}).get(e["item"], by_key[e["item"]]["refusal"]) for e in a["entries"]
    ]
    p_labels = [pair_label(e) if pair_label else _nli_says(by_key, e) for e in b["entries"]]
    return lb.score(rows, a, a_labels, b, p_labels, iterations=200)


def _nli_says(by_key: dict[str, Any], entry: dict[str, Any]) -> str:
    for pair in by_key[entry["item"]]["pairs"]:
        if pair["claim_index"] == entry["claim_index"] and pair["label"] == entry["citation"]:
            return "supports" if pair["supported"] else "does not support"
    raise AssertionError("pair not found")


class TestBarOne:
    def _unanswerable_keys(self, rows: list[dict[str, Any]]) -> list[str]:
        return [lb.item_id(r["key"]) for r in rows if r["category"] != "answerable"]

    def test_it_is_judged_on_the_owners_labels_not_rule_1s(self) -> None:
        rows = world()  # Rule 1 says every unanswerable answer was answered
        refusals = dict.fromkeys(self._unanswerable_keys(rows), "refused")
        got = scored(rows, refusals)["bar_1"]
        assert got["false_answer_rate"] == 0.0 and got["verdict"] == "MET"
        assert got["rule_1_false_answer_rate"] == 1.0

    def test_a_partial_answer_is_a_false_answer(self) -> None:
        rows = world()
        labels = dict.fromkeys(self._unanswerable_keys(rows), "partial")
        got = scored(rows, labels)["bar_1"]
        assert got["false_answer_rate"] == 1.0 and got["verdict"] == "NOT MET"

    def test_the_boundary_is_at_most_one_in_ten(self) -> None:
        # 4 of 49 is 0.082 (met); 5 of 49 is 0.102 (not met).
        rows = world()
        keys = self._unanswerable_keys(rows)
        for false, verdict in ((4, "MET"), (5, "NOT MET")):
            labels = {k: ("answered" if i < false else "refused") for i, k in enumerate(keys)}
            got = scored(rows, labels)["bar_1"]
            assert got["false_answers"] == false and got["verdict"] == verdict

    def test_exactly_one_in_ten_is_met(self) -> None:
        # "At most 0.10": with 49 questions 0.10 cannot occur, so 10 pin the edge.
        rows = [row(i, category="unanswerable-withheld") for i in range(10)]
        rows += [row(100 + i, pairs=[(0, "C1", True)]) for i in range(60)]
        keys = [lb.item_id(r["key"]) for r in rows if r["category"] != "answerable"]
        labels = {k: ("answered" if i == 0 else "refused") for i, k in enumerate(keys)}
        got = scored(rows, labels)["bar_1"]
        assert got["false_answer_rate"] == pytest.approx(0.1)
        assert got["verdict"] == "MET"

    def test_its_interval_resamples_papers(self) -> None:
        got = scored(world())["bar_1"]
        assert got["ci95_papers"] == [1.0, 1.0]  # every unanswerable answer answered


class TestAgreement:
    def test_rule_1_agreeing_everywhere_is_reliable(self) -> None:
        rows = world()
        rows[0]["refusal"] = "refused"
        rows[1]["refusal"] = "partial"
        got = scored(rows)["rule_1"]
        assert got["raw_agreement"] == 1.0 and got["reliable"]

    def test_rule_1_below_0_80_is_flagged_unreliable(self) -> None:
        rows = world()
        for i, r in enumerate(rows[:49]):
            r["refusal"] = ("refused", "partial", "answered")[i % 3]
        labels = {lb.item_id(r["key"]): "answered" for r in rows[:49]}
        got = scored(rows, labels)["rule_1"]
        assert got["kappa"] is not None and got["kappa"] < lb.RULE_1_KAPPA
        assert not got["reliable"]

    def test_cannot_tell_is_excluded_from_agreement_and_counted(self) -> None:
        rows = world()
        by_key = {lb.item_id(r["key"]): r for r in rows}
        calls = {"n": 0}

        def label(entry: dict[str, Any]) -> str:
            calls["n"] += 1
            return "cannot tell" if calls["n"] <= 10 else _nli_says(by_key, entry)

        got = scored(rows, pair_label=label)["nli"]
        assert got["cannot_tell"] == 10
        assert got["agreement"]["items"] == 90
        assert got["agreement"]["raw_agreement"] == 1.0

    def test_the_nli_side_is_read_from_the_archive(self) -> None:
        # The owner says every pair is supported; the archive says about half.
        got = scored(world(), pair_label=lambda _e: "supports")["nli"]
        assert got["owner_supports"] == 100
        assert 0 < got["nli_supports"] < 100


class TestBarTwo:
    def test_full_agreement_permits_the_flags(self) -> None:
        got = scored(world())["bar_2"]
        assert got["kappa"] == pytest.approx(1.0) and got["verdict"] == "MAY BE BUILT"

    def test_high_raw_agreement_with_no_real_agreement_keeps_them_unbuilt(self) -> None:
        # The NLI model supports 80% of pairs and the owner says "supports" to all:
        # raw agreement is about 0.8, kappa is 0. Bar 2 is on kappa (D179).
        rows = [row(i, category="unanswerable-qasper") for i in range(49)]
        rows += [row(100 + i, pairs=[(0, "C1", i % 5 != 0)]) for i in range(120)]
        got = scored(rows, pair_label=lambda _e: "supports")
        assert got["nli"]["agreement"]["raw_agreement"] >= 0.6
        assert got["bar_2"]["kappa"] == pytest.approx(0.0)
        assert got["bar_2"]["verdict"].startswith("NOT BUILT")

    def test_disagreement_keeps_them_unbuilt(self) -> None:
        rows = world()
        by_key = {lb.item_id(r["key"]): r for r in rows}

        def opposite(entry: dict[str, Any]) -> str:
            return "does not support" if _nli_says(by_key, entry) == "supports" else "supports"

        got = scored(rows, pair_label=opposite)["bar_2"]
        assert got["kappa"] < lb.BAR_2_KAPPA
        assert got["verdict"].startswith("NOT BUILT")


class TestTheCommandLine:
    def _archive(self, tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
        path = tmp_path / "answers.json"
        path.write_text(
            json.dumps({"config": {"split": "eval"}, "blocks": {"answers": {"rows": rows}}}),
            encoding="utf-8",
        )
        return path

    def test_sheets_are_written_and_an_identical_rerun_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        archive = self._archive(tmp_path, world())
        first = cli.write_label_sheets(archive, tmp_path / "labelling")
        again = cli.write_label_sheets(archive, tmp_path / "labelling")
        assert set(first["answers"]["files"].values()) == {"written"}
        assert set(again["answers"]["files"].values()) == {"unchanged"}

    def test_a_label_file_with_labels_in_it_is_never_overwritten(self, tmp_path: Path) -> None:
        archive = self._archive(tmp_path, world())
        out = tmp_path / "labelling"
        cli.write_label_sheets(archive, out)
        labels = out / "d179-answer-labels.csv"
        labels.write_text(labels.read_text(encoding="utf-8") + "my work\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="may hold labels"):
            cli.write_label_sheets(archive, out)
        assert labels.read_text(encoding="utf-8").endswith("my work\n")

    def test_labels_made_on_other_answers_are_refused(self, tmp_path: Path) -> None:
        rows = world()
        archive = self._archive(tmp_path, rows)
        out = tmp_path / "labelling"
        cli.write_label_sheets(archive, out)
        rows[0]["shown"] = "a different answer"
        archive.write_text(
            json.dumps({"config": {"split": "eval"}, "blocks": {"answers": {"rows": rows}}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="labels score only the answers they read"):
            cli.score_labels(archive, out)

    def test_a_dev_archive_is_refused(self, tmp_path: Path) -> None:
        path = tmp_path / "dev.json"
        path.write_text(json.dumps({"config": {"split": "train-dev"}, "blocks": {}}), "utf-8")
        with pytest.raises(ValueError, match="not an eval answers archive"):
            cli.write_label_sheets(path, tmp_path / "labelling")

    def test_scoring_reads_the_owners_files_end_to_end(self, tmp_path: Path) -> None:
        rows = world()
        archive = self._archive(tmp_path, rows)
        out = tmp_path / "labelling"
        cli.write_label_sheets(archive, out)
        a = json.loads((out / "d179-answer-sheet.json").read_text(encoding="utf-8"))
        b = json.loads((out / "d179-pair-sheet.json").read_text(encoding="utf-8"))
        by_key = {lb.item_id(r["key"]): r for r in rows}
        (out / "d179-answer-labels.csv").write_text(filled(a, ["refused"] * 100), encoding="utf-8")
        (out / "d179-pair-labels.csv").write_text(
            filled(b, [_nli_says(by_key, e) for e in b["entries"]]), encoding="utf-8"
        )
        report = cli.score_labels(archive, out)
        assert report.blocks["bar_1"]["verdict"] == "MET"
        assert report.blocks["bar_2"]["verdict"] == "MAY BE BUILT"
        assert set(report.config["label_files_sha256"]) == {
            "d179-answer-labels.csv",
            "d179-pair-labels.csv",
        }
        assert "MET" in cli._render_labels(report)
