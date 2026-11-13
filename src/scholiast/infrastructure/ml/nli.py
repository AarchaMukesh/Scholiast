"""Local NLI cross-encoder for claim-level citation verification (ML-12, D16, D179).

Layer 3 of doc 6 §6.11: does the passage a claim cites actually support it? A
DeBERTa-v3 NLI model reads (premise, hypothesis) jointly and returns entailment,
neutral and contradiction. M5 phase 2 uses it to measure groundedness, citation
support and coverage offline; whether it may also flag claims in the product
waits on its agreement with the owner's labels (D179's bar 2).

**The checkpoint is pinned, not chosen at runtime** (D179's clarification,
2026-11-13): `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` at one revision, loaded
from the local copy only. It was downloaded once under its own approval, so a
missing copy is an error here rather than a quiet download (NFR-T3).

**Support is the argmax, read through the checkpoint's own labels.** Checkpoints
order their labels differently (this one puts entailment at 0, the
`cross-encoder/nli-deberta-v3-*` family at 1), so the index comes from the model's
`id2label`, never from a constant.

**Only the premise is truncated.** A claim cut to fit would be a different claim;
a long passage cut at the window loses only text the model cannot read anyway.
Every truncated pair is reported, and a claim that cannot fit beside at least one
premise token is refused outright.

**fp32, not D18's fp16.** D18 exists for serving latency; this check runs offline,
and DeBERTa-v3 is prone to fp16 overflow. Precision is in the version stamp.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Mapping, Sequence
from typing import Any

from scholiast.application.ports.ml import Entailment, Precision

DEFAULT_NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEFAULT_NLI_REVISION = "6f5cf0a2b59cabb106aca4c287eed12e357e90eb"

#: DeBERTa-v3's window, which this checkpoint was trained at.
MAX_LENGTH = 512

#: Pairs per forward pass. Offline and fp32 on a ~184M model: memory, not latency,
#: sets it, and 32 pairs of 512 tokens fit with room beside the other models.
BATCH_SIZE = 32

_LABELS = ("entailment", "neutral", "contradiction")


class NliModelUnusable(RuntimeError):
    """The checkpoint cannot be read the way D179 requires."""


class ClaimExceedsWindow(ValueError):
    """A claim too long to sit beside any premise text in the model's window."""


def label_indices(id2label: Mapping[Any, str]) -> tuple[int, int, int]:
    """Entailment, neutral and contradiction's logit indices, from the model's config."""
    by_name = {str(name).strip().lower(): int(index) for index, name in id2label.items()}
    missing = [name for name in _LABELS if name not in by_name]
    if missing:
        raise NliModelUnusable(
            f"the checkpoint's labels {sorted(by_name)} lack {missing}; "
            "D179 needs entailment, neutral and contradiction"
        )
    return by_name["entailment"], by_name["neutral"], by_name["contradiction"]


def reading(
    logits: Sequence[float], indices: tuple[int, int, int], *, truncated: bool
) -> Entailment:
    """One pair's softmax over the three labels, as the port reports it."""
    peak = max(logits)
    weights = [math.exp(value - peak) for value in logits]
    total = sum(weights)
    entail, neutral, contradict = (weights[i] / total for i in indices)
    return Entailment(
        entailment=entail,
        neutral=neutral,
        contradiction=contradict,
        premise_truncated=truncated,
    )


def check_window(
    premise_tokens: int, claim_tokens: int, special_tokens: int, *, max_length: int
) -> bool:
    """Whether the premise must be cut; raises if the claim alone cannot fit.

    The claim needs its own tokens, the pair's special tokens and at least one
    premise token beside it, or there is nothing left to verify it against.
    """
    if claim_tokens + special_tokens + 1 > max_length:
        raise ClaimExceedsWindow(
            f"a claim of {claim_tokens} tokens leaves no room for a premise in a "
            f"{max_length}-token window; D179 never truncates the claim"
        )
    return premise_tokens + claim_tokens + special_tokens > max_length


class LocalEntailmentChecker:
    """`EntailmentChecker` adapter over a pinned Hugging Face NLI checkpoint."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI,
        *,
        revision: str = DEFAULT_NLI_REVISION,
        device: str = "cuda",
        max_length: int = MAX_LENGTH,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self._model_name = model_name
        self._revision = revision
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        self._loaded: tuple[Any, Any, tuple[int, int, int]] | None = None

    @property
    def version(self) -> str:
        return f"{self._model_name}@{self._revision[:12]}/{Precision.FP32.value}/1.0.0"

    async def entails(self, premises: Sequence[str], claims: Sequence[str]) -> list[Entailment]:
        if len(premises) != len(claims):
            raise ValueError(f"{len(premises)} premises for {len(claims)} claims")
        if not premises:
            return []
        return await asyncio.to_thread(self._predict, list(premises), list(claims))

    def _load(self) -> tuple[Any, Any, tuple[int, int, int]]:
        """Load on first use, from the local copy only; construction costs nothing."""
        if self._loaded is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, revision=self._revision, local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                revision=self._revision,
                local_files_only=True,
                torch_dtype=torch.float32,
            )
            model.to(self._device)
            model.eval()
            self._loaded = (tokenizer, model, label_indices(model.config.id2label))
        return self._loaded

    def _predict(self, premises: list[str], claims: list[str]) -> list[Entailment]:
        import torch

        tokenizer, model, indices = self._load()
        special = tokenizer.num_special_tokens_to_add(pair=True)
        premise_lengths = [
            len(ids) for ids in tokenizer(premises, add_special_tokens=False)["input_ids"]
        ]
        claim_lengths = [
            len(ids) for ids in tokenizer(claims, add_special_tokens=False)["input_ids"]
        ]
        truncated = [
            check_window(p, c, special, max_length=self._max_length)
            for p, c in zip(premise_lengths, claim_lengths, strict=True)
        ]
        readings: list[Entailment] = []
        for start in range(0, len(premises), self._batch_size):
            stop = start + self._batch_size
            batch = tokenizer(
                premises[start:stop],
                claims[start:stop],
                truncation="only_first",
                max_length=self._max_length,
                padding=True,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                logits = model(**batch).logits.float().cpu().tolist()
            readings.extend(
                reading(row, indices, truncated=cut)
                for row, cut in zip(logits, truncated[start:stop], strict=True)
            )
        return readings
