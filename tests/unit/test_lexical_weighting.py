"""Production's lexical weighting is the one D168's gate measured (D172).

Production fixes BM25's average length at D-RET's mean instead of measuring its
own corpus, so the arms, gate and published numbers describe its weighting
exactly. That claim rests on three facts these tests pin, offline, against the
vendored D-RET snapshot:

- the constant IS D-RET's mean, under the chunker and tokenizer it was taken with;
- production's weights for every D-RET chunk equal what the eval's lexical index
  wrote, bit for bit;
- the weighting cannot change without `LEXICAL_VERSION` changing with it, because
  a hybrid read is refused until every point carries the current version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Final

import pytest

from scholiast.application.ports.vector import SparseRecord
from scholiast.domain import lexical
from scholiast.domain.chunking import CHUNKER_VERSION
from scholiast.eval.dret import DRet, build_dret
from scholiast.eval.harness import build_lexical_index
from scholiast.eval.snapshot import load_snapshot


@pytest.fixture(scope="module")
def dret() -> DRet:
    return build_dret(load_snapshot())


class _Recorder:
    def __init__(self) -> None:
        self.records: list[SparseRecord] = []

    async def upsert_sparse(self, records: Sequence[SparseRecord]) -> int:
        self.records.extend(records)
        return len(records)

    async def search_lexical(self, *args: object, **kwargs: object) -> list[object]:
        raise AssertionError("not used")


class TestTheFixedAverageIsDRetsMean:
    def test_the_constant_equals_the_mean_over_the_vendored_corpus(self, dret: DRet) -> None:
        # Exact equality, not approximate: the claim is that on D-RET production's
        # weights ARE the eval's, which a rounded constant would make false.
        tokens = [lexical.tokenize(chunk.text) for chunk in dret.chunks.values()]
        assert len(tokens) == 7854
        assert lexical.average_length(tokens) == lexical.AVERAGE_LENGTH

    def test_it_was_taken_under_todays_chunker(self, dret: DRet) -> None:
        # The chunker bounds chunk length, so the average mostly describes it. A
        # new chunker version is a decision about this constant, made on purpose:
        # this fails until someone makes it.
        assert dret.versions["chunker"] == CHUNKER_VERSION == "structure-aware/1.2.0"


class TestProductionWeightsAreTheMeasuredWeights:
    async def test_every_dret_chunk_weighs_what_the_eval_index_wrote(self, dret: DRet) -> None:
        recorder = _Recorder()
        await build_lexical_index(dret, recorder)
        assert len(recorder.records) == len(dret.chunks)
        for record in recorder.records:
            assert lexical.chunk_weights(dret.chunks[record.chunk_id].text) == record.weights


#: The weighting's identity at each LEXICAL_VERSION. A change to the constant, the
#: BM25 parameters or the tokenizer moves the fingerprint; the fix is to bump
#: LEXICAL_VERSION and add its fingerprint here, which is what forces the full
#: reindex a hybrid read then waits for (D172).
FINGERPRINTS: Final = {
    "bm25-identifier/1.1.0": "055adbb6f45fd0da1e2be38d620e20d591e07b6310ef5aa583b7e829678e37e0",
}


class TestTheWeightingCannotChangeWithoutItsVersion:
    def test_the_current_weighting_matches_its_version(self) -> None:
        identity = json.dumps(
            {
                "avgdl": repr(lexical.AVERAGE_LENGTH),
                "k1": lexical.BM25_K1,
                "b": lexical.BM25_B,
                "token": lexical._TOKEN.pattern,
                "split": lexical._SPLIT.pattern,
                "min_subtoken": lexical._MIN_SUBTOKEN,
                "golden": lexical.tokenize("LLaMA-2-70B reaches nDCG@10 of 0.847 on SQuAD v1.1."),
                "ids": [lexical.term_id("llama-2-70b"), lexical.term_id("ndcg@10")],
                "weights": sorted(lexical.chunk_weights("BM25 BM25 ranks LLaMA-2-70B").items()),
            },
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(identity.encode()).hexdigest()
        assert FINGERPRINTS.get(lexical.LEXICAL_VERSION) == fingerprint
