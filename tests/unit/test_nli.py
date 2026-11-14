"""NLI adapter tests (ML-12, D179's clarification of 2026-11-13).

What goes silently wrong in an NLI adapter is the reading, not the model: a label
index assumed rather than read, a threshold where the argmax was fixed, a claim
truncated to fit. Those are decidable without the model, so they are tested here
with stand-ins for `transformers` and `torch`, which CI's unit job does not
install. `TestTheRealModel` checks the pinned checkpoint itself, and only runs
where it is already cached and a GPU exists (NFR-T3: no download in a test).
"""

from __future__ import annotations

import contextlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from scholiast.application.ports.ml import Entailment, EntailmentChecker
from scholiast.infrastructure.ml.nli import (
    DEFAULT_NLI,
    DEFAULT_NLI_REVISION,
    ClaimExceedsWindow,
    LocalEntailmentChecker,
    NliModelUnusable,
    check_window,
    label_indices,
    reading,
)

#: This checkpoint's order, and the `cross-encoder/nli-deberta-v3-*` family's.
MORITZ = {0: "entailment", 1: "neutral", 2: "contradiction"}
CROSS_ENCODER = {0: "contradiction", 1: "entailment", 2: "neutral"}


def entailment(e: float, n: float, c: float) -> Entailment:
    return Entailment(entailment=e, neutral=n, contradiction=c, premise_truncated=False)


class TestSupportIsTheArgmax:
    """D179: supported when entailment is the argmax, with no threshold."""

    def test_entailment_below_one_half_still_supports_when_it_is_the_argmax(self) -> None:
        # A 0.5 threshold, the obvious mistake, would call this unsupported.
        assert entailment(0.40, 0.35, 0.25).supported

    def test_entailment_above_one_half_is_still_only_support_if_it_leads(self) -> None:
        assert not entailment(0.20, 0.70, 0.10).supported

    def test_a_tie_for_first_is_not_support(self) -> None:
        # Not the argmax: the model did not prefer entailment.
        assert not entailment(0.45, 0.45, 0.10).supported
        assert not entailment(0.45, 0.10, 0.45).supported


class TestTheLabelsComeFromTheCheckpoint:
    def test_the_entailment_index_is_read_not_assumed(self) -> None:
        # The two candidate checkpoints disagree on where entailment sits; a
        # hard-coded index would silently read one of them backwards.
        assert label_indices(MORITZ) == (0, 1, 2)
        assert label_indices(CROSS_ENCODER) == (1, 2, 0)

    def test_label_names_are_matched_whatever_their_case_or_key_type(self) -> None:
        assert label_indices({"0": "ENTAILMENT", "1": "Neutral", "2": "contradiction"}) == (
            0,
            1,
            2,
        )

    def test_a_two_way_checkpoint_is_refused(self) -> None:
        with pytest.raises(NliModelUnusable, match="neutral"):
            label_indices({0: "entailment", 1: "not_entailment"})

    def test_the_reading_follows_the_checkpoints_order(self) -> None:
        logits = [3.0, 0.0, -3.0]  # index 0 strongest
        assert reading(logits, label_indices(MORITZ), truncated=False).supported
        assert not reading(logits, label_indices(CROSS_ENCODER), truncated=False).supported

    def test_the_three_probabilities_are_a_distribution(self) -> None:
        r = reading([2.0, 1.0, 0.5], (0, 1, 2), truncated=True)
        assert r.entailment + r.neutral + r.contradiction == pytest.approx(1.0)
        assert r.entailment > r.neutral > r.contradiction
        assert r.premise_truncated

    def test_large_logits_do_not_overflow(self) -> None:
        r = reading([1000.0, 0.0, -1000.0], (0, 1, 2), truncated=False)
        assert r.entailment == pytest.approx(1.0)


class TestOnlyThePremiseIsTruncated:
    def test_a_pair_that_fits_is_not_truncated(self) -> None:
        assert not check_window(400, 100, 3, max_length=512)

    def test_a_long_premise_is_truncated_and_says_so(self) -> None:
        assert check_window(500, 100, 3, max_length=512)

    def test_the_boundary_is_exact(self) -> None:
        assert not check_window(409, 100, 3, max_length=512)
        assert check_window(410, 100, 3, max_length=512)

    def test_a_claim_with_no_room_for_a_premise_is_refused_not_cut(self) -> None:
        with pytest.raises(ClaimExceedsWindow, match="never truncates the claim"):
            check_window(10, 509, 3, max_length=512)

    def test_a_claim_leaving_one_premise_token_is_accepted(self) -> None:
        assert check_window(10, 508, 3, max_length=512)


# --- the adapter, with stand-ins for transformers and torch ----------------------


class _Batch(dict[str, Any]):
    def to(self, device: str) -> _Batch:
        self["device"] = device
        return self


class _Tokenizer:
    """Whitespace tokens; records how the pair batch was built."""

    def __init__(self) -> None:
        self.pair_calls: list[dict[str, Any]] = []

    def num_special_tokens_to_add(self, *, pair: bool) -> int:
        return 3 if pair else 2

    def __call__(self, first: list[str], second: list[str] | None = None, **kw: Any) -> Any:
        if second is None:
            return {"input_ids": [text.split() for text in first]}
        self.pair_calls.append({"premises": list(first), "claims": list(second), **kw})
        return _Batch(rows=len(first))


class _Logits:
    def __init__(self, rows: list[list[float]]) -> None:
        self.rows = rows

    def float(self) -> _Logits:
        return self

    def cpu(self) -> _Logits:
        return self

    def tolist(self) -> list[list[float]]:
        return self.rows


class _Model:
    def __init__(self, id2label: dict[int, str], script: list[list[float]]) -> None:
        self.config = types.SimpleNamespace(id2label=id2label)
        self.script = list(script)
        self.device: str | None = None
        self.evaluated = False

    def to(self, device: str) -> _Model:
        self.device = device
        return self

    def eval(self) -> _Model:
        self.evaluated = True
        return self

    def __call__(self, **batch: Any) -> Any:
        rows, self.script = self.script[: batch["rows"]], self.script[batch["rows"] :]
        return types.SimpleNamespace(logits=_Logits(rows))


class _Library:
    """`transformers` and `torch` as `_load` and `_predict` import them."""

    def __init__(self, id2label: dict[int, str], script: list[list[float]]) -> None:
        self.tokenizer = _Tokenizer()
        self.model = _Model(id2label, script)
        self.loads: list[tuple[str, dict[str, Any]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        library = self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name: str, **kw: Any) -> _Tokenizer:
                library.loads.append(("tokenizer", {"name": name, **kw}))
                return library.tokenizer

        class AutoModelForSequenceClassification:
            @staticmethod
            def from_pretrained(name: str, **kw: Any) -> _Model:
                library.loads.append(("model", {"name": name, **kw}))
                return library.model

        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        transformers.AutoModelForSequenceClassification = AutoModelForSequenceClassification  # type: ignore[attr-defined]
        torch = types.ModuleType("torch")
        torch.float32 = "float32"  # type: ignore[attr-defined]
        torch.inference_mode = contextlib.nullcontext  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.setitem(sys.modules, "torch", torch)


class TestTheAdapter:
    def test_it_satisfies_the_port(self) -> None:
        assert isinstance(LocalEntailmentChecker(), EntailmentChecker)

    def test_construction_loads_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        library = _Library(MORITZ, [])
        library.install(monkeypatch)
        LocalEntailmentChecker()
        assert library.loads == []

    async def test_it_loads_the_pinned_revision_from_the_local_copy_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # D179: downloaded once under its own approval; a missing copy must fail,
        # not fetch (NFR-T3), and a moved `main` must not change the model.
        library = _Library(MORITZ, [[3.0, 0.0, 0.0]])
        library.install(monkeypatch)
        await LocalEntailmentChecker(device="cpu").entails(["p"], ["c"])
        assert {kind for kind, _ in library.loads} == {"tokenizer", "model"}
        for _, kw in library.loads:
            assert kw["name"] == DEFAULT_NLI
            assert kw["revision"] == DEFAULT_NLI_REVISION
            assert kw["local_files_only"] is True
        model_kw = dict(library.loads)["model"]
        assert model_kw["torch_dtype"] == "float32"
        assert library.model.evaluated
        assert library.model.device == "cpu"

    async def test_pairs_are_premise_first_and_truncate_only_the_premise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = _Library(MORITZ, [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
        library.install(monkeypatch)
        await LocalEntailmentChecker(device="cpu").entails(["p1 p1", "p2"], ["c1", "c2 c2"])
        (call,) = library.tokenizer.pair_calls
        assert call["premises"] == ["p1 p1", "p2"]
        assert call["claims"] == ["c1", "c2 c2"]
        assert call["truncation"] == "only_first"
        assert call["max_length"] == 512

    async def test_each_pair_gets_its_own_reading_across_batches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        script = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
        library = _Library(CROSS_ENCODER, script)
        library.install(monkeypatch)
        checker = LocalEntailmentChecker(device="cpu", batch_size=2)
        got = await checker.entails(["a", "b", "c"], ["x", "y", "z"])
        # CROSS_ENCODER: index 0 is contradiction, 1 entailment, 2 neutral.
        assert [g.supported for g in got] == [False, True, False]
        assert got[0].contradiction > 0.9 and got[2].neutral > 0.9
        assert len(library.tokenizer.pair_calls) == 2

    async def test_a_premise_past_the_window_is_reported_as_truncated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        library = _Library(MORITZ, [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        library.install(monkeypatch)
        long_premise = " ".join(["w"] * 600)
        got = await LocalEntailmentChecker(device="cpu").entails(
            [long_premise, "short"], ["claim", "claim"]
        )
        assert [g.premise_truncated for g in got] == [True, False]

    async def test_a_truncation_flag_stays_with_its_pair_in_a_later_batch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Flags are computed once for all pairs and readings per batch; an offset
        # error would move a truncation onto the wrong pair, or drop it.
        library = _Library(MORITZ, [[3.0, 0.0, 0.0]] * 3)
        library.install(monkeypatch)
        long_premise = " ".join(["w"] * 600)
        got = await LocalEntailmentChecker(device="cpu", batch_size=2).entails(
            ["short", "short", long_premise], ["claim"] * 3
        )
        assert [g.premise_truncated for g in got] == [False, False, True]

    async def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(ValueError, match="2 premises for 1 claims"):
            await LocalEntailmentChecker().entails(["a", "b"], ["c"])

    async def test_no_pairs_loads_no_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        library = _Library(MORITZ, [])
        library.install(monkeypatch)
        assert await LocalEntailmentChecker().entails([], []) == []
        assert library.loads == []

    def test_the_version_names_checkpoint_revision_and_precision(self) -> None:
        version = LocalEntailmentChecker().version
        assert version.startswith(f"{DEFAULT_NLI}@{DEFAULT_NLI_REVISION[:12]}/fp32/")


# --- the real checkpoint ------------------------------------------------------------


def _pinned_copy_cached() -> bool:
    repo = Path.home() / ".cache" / "huggingface" / "hub"
    snapshot = repo / f"models--{DEFAULT_NLI.replace('/', '--')}" / "snapshots"
    return (snapshot / DEFAULT_NLI_REVISION / "model.safetensors").exists()


def _has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # any import failure means "no usable CUDA"
        return False


@pytest.mark.gpu
@pytest.mark.skipif(
    not (_pinned_copy_cached() and _has_cuda()), reason="needs the pinned NLI copy + CUDA"
)
class TestTheRealModel:
    async def test_a_paraphrase_is_supported_and_a_contradiction_is_not(self) -> None:
        # The only test that the pinned checkpoint, read through its own labels,
        # actually judges entailment rather than returning some numbers.
        checker = LocalEntailmentChecker()
        premise = "The model was trained for three days on eight GPUs."
        got = await checker.entails(
            [premise, premise],
            ["Training used eight GPUs.", "Training ran on a single CPU."],
        )
        assert got[0].supported
        assert not got[1].supported
        assert got[1].contradiction > got[1].entailment
