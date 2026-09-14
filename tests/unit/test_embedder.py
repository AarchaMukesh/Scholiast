"""Embedder adapter tests that need no model loaded.

The prefix and precision logic is where this adapter goes silently wrong, and
both are decidable without a GPU or a 440 MB download — so they are tested here,
in the suite that always runs. `tests/integration/test_embedding.py` covers what
genuinely needs the hardware.
"""

from __future__ import annotations

import pytest

from scholiast.application.ports.ml import Precision, TextRole
from scholiast.infrastructure.ml.embedder import BGE_QUERY_PREFIX, LocalEmbedder


def bge(**kw: object) -> LocalEmbedder:
    return LocalEmbedder("BAAI/bge-base-en-v1.5", **kw)  # type: ignore[arg-type]


class TestInstructionPrefixes:
    """§6.4 calls this non-negotiable, because getting it wrong is invisible."""

    def test_a_query_gets_the_retrieval_instruction(self) -> None:
        # Prevents the classic asymmetric-encoder mistake. There is no error
        # when it is wrong — retrieval is just quietly worse, which is the
        # hardest kind of regression to notice.
        prepared = bge()._apply_prefix(["what is RRF?"], TextRole.QUERY)
        assert prepared == [f"{BGE_QUERY_PREFIX}what is RRF?"]

    def test_a_passage_gets_no_instruction(self) -> None:
        # The other half, and the more damaging direction: prefixing passages
        # too would make every stored vector wrong, requiring a full re-index to
        # fix rather than just a restart.
        text = "Reciprocal rank fusion combines ranked lists."
        assert bge()._apply_prefix([text], TextRole.PASSAGE) == [text]

    def test_an_unknown_model_family_gets_no_guessed_prefix(self) -> None:
        # A wrong instruction is worse than none. If the model is not one whose
        # convention we know, we stay out of the way.
        embedder = LocalEmbedder("some-org/mystery-encoder")
        assert embedder._apply_prefix(["q"], TextRole.QUERY) == ["q"]

    def test_e5_gets_its_own_convention(self) -> None:
        # E5's prefix discipline differs from BGE's; §6.4 names both, so the
        # mapping is per-family rather than one hardcoded string.
        assert LocalEmbedder("intfloat/e5-base-v2")._apply_prefix(["q"], TextRole.QUERY) == [
            "query: q"
        ]


class TestPrecisionIsPartOfTheContract:
    """D18: fp16 is specification, not optimisation."""

    def test_gpu_defaults_to_fp16(self) -> None:
        assert bge(device="cuda").precision is Precision.FP16

    def test_cpu_falls_back_to_fp32(self) -> None:
        # fp16 on CPU is usually slower and partly unsupported. §5.7 requires the
        # CPU path to keep working, so it runs fp32 and reports fp32 rather than
        # claiming a precision it is not using.
        assert bge(device="cpu", precision=Precision.FP16).precision is Precision.FP32

    def test_precision_appears_in_the_version_stamp(self) -> None:
        # NFR-M5: fp16 and fp32 do not produce identical vectors, so "re-embed
        # everything done at fp32" is only expressible if the stamps differ.
        assert bge(device="cuda").version != bge(device="cpu").version
        assert "fp16" in bge(device="cuda").version
        assert "fp32" in bge(device="cpu").version

    def test_the_version_names_the_model(self) -> None:
        # A stamp that did not identify the model would make targeted
        # re-indexing impossible after a model swap — the exact scenario NFR-M5
        # exists for.
        assert "bge-base-en-v1.5" in bge().version


class TestEmptyInput:
    async def test_embedding_nothing_returns_nothing_without_loading_a_model(self) -> None:
        # Guards the ingestion path: a paper whose sections all got skipped must
        # not pay a model load to embed zero chunks.
        embedder = bge()
        assert await embedder.embed([], role=TextRole.PASSAGE) == []
        assert embedder._model is None, "an empty batch loaded the model anyway"


class TestLazyLoading:
    def test_construction_does_not_load_the_model(self) -> None:
        # Wiring happens at import time; a constructor that costs two seconds
        # and half a gigabyte of VRAM would make every test that never embeds
        # anything pay for one.
        assert bge()._model is None

    def test_an_unknown_model_fails_only_when_used(self) -> None:
        embedder = LocalEmbedder("definitely/not-a-real-model-xyz", device="cpu")
        # Deliberately broad: sentence-transformers raises different types for a
        # missing repo depending on whether the hub is reachable, and pinning one
        # would make this test fail offline for the wrong reason. What matters is
        # that construction succeeded and the failure arrived only on use.
        with pytest.raises(Exception):  # noqa: B017
            embedder._load()
