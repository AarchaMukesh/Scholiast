"""BM25 term weighting, tested on what doc 6 §6.5 says the lexical arm is FOR.

The arm exists because dense retrieval compresses `LLaMA-2-70B` and `LLaMA-3-8B`
into the same neighbourhood while BM25 knows they are different strings. So the
tests are about identifiers surviving tokenisation and about the weighting having
the properties BM25 is chosen for — not about `tokenize` returning a list.

The `[SPIKE]` doc 6 attached to this decision is resolved here in assertion form
(D73): if a future change shatters `nDCG@10` into `ndcg` and `10`, these fail.
"""

from __future__ import annotations

import pytest

from scholiast.domain.errors import InvariantViolation
from scholiast.domain.lexical import (
    average_length,
    document_weights,
    query_weights,
    term_id,
    tokenize,
)

#: The identifier classes §6.5 names as the risk the spike had to close.
IDENTIFIERS = [
    "LLaMA-2-70B",
    "nDCG@10",
    "FlashAttention-2",
    "0.847",
    "BAAI/bge-base-en-v1.5",
    "BERT-base-uncased",
    "F1-score",
    "GPT-4o",
    "sm_120",
]


class TestTheTokenizerKeepsWhatTheArmExistsToMatch:
    @pytest.mark.parametrize("identifier", IDENTIFIERS)
    def test_a_scientific_identifier_survives_whole(self, identifier: str) -> None:
        # A naive \w+ split shatters every one of these, discarding exactly the
        # rare-term signal that justifies having a lexical arm at all.
        assert identifier.lower() in tokenize(identifier)

    def test_a_compound_also_emits_its_parts(self) -> None:
        # So a query naming a model family still matches a document naming a
        # specific size of it. Without this the tokenizer would be MORE exact than
        # is useful.
        tokens = tokenize("LLaMA-2-70B")
        assert "llama-2-70b" in tokens
        assert "llama" in tokens

    def test_one_character_fragments_are_dropped(self) -> None:
        # The `2` in `LLaMA-2-70B` matches every version number in the corpus and
        # disambiguates nothing; the compound already carries that information.
        assert "2" not in tokenize("FlashAttention-2")
        assert "flashattention-2" in tokenize("FlashAttention-2")

    def test_a_decimal_number_is_not_split_into_two_integers(self) -> None:
        assert tokenize("0.847") == ["0.847"]

    def test_repeats_are_preserved_because_bm25_needs_term_frequency(self) -> None:
        # Deduplicating here would silently turn BM25 into a binary model.
        assert tokenize("bert bert bert").count("bert") == 3


class TestTermIds:
    def test_the_id_is_stable_across_calls(self) -> None:
        assert term_id("llama-2-70b") == term_id("llama-2-70b")

    def test_the_id_does_not_use_python_hash(self) -> None:
        # PYTHONHASHSEED randomises str hashing per interpreter, so an index built
        # in one process would be silently unqueryable from another — every lookup
        # simply missing, with no error anywhere. This pins the actual value.
        assert term_id("llama-2-70b") == 2819683187

    def test_different_terms_get_different_ids(self) -> None:
        ids = {term_id(t) for t in IDENTIFIERS}
        assert len(ids) == len(IDENTIFIERS)


class TestBm25Weighting:
    def test_term_frequency_saturates(self) -> None:
        # The property BM25 is chosen for: the tenth occurrence adds far less than
        # the second. Linear TF would let a keyword-stuffed chunk dominate.
        one = document_weights(["a"], average_length=10.0)[term_id("a")]
        two = document_weights(["a", "a"], average_length=10.0)[term_id("a")]
        ten = document_weights(["a"] * 10, average_length=10.0)[term_id("a")]
        assert one < two < ten
        assert (ten - two) < (two - one)

    def test_a_longer_document_is_penalised_for_the_same_term_count(self) -> None:
        # Length normalisation: otherwise a long section outranks a short one
        # merely by containing more words.
        short = document_weights(["a", "b"], average_length=10.0)[term_id("a")]
        long = document_weights(["a", *[f"w{i}" for i in range(50)]], average_length=10.0)
        assert long[term_id("a")] < short

    def test_idf_is_absent_because_the_store_supplies_it(self) -> None:
        # A term appearing once in a one-word document has weight independent of
        # how rare it is corpus-wide — that factor is Qdrant's `Modifier.IDF`
        # (D73). If IDF leaked in here, weights would freeze against the corpus as
        # it stood at index time and every later insert would make them wrong.
        rare = document_weights(["zzz"], average_length=1.0)[term_id("zzz")]
        common = document_weights(["the"], average_length=1.0)[term_id("the")]
        assert rare == common

    def test_the_query_side_is_one_per_distinct_term(self) -> None:
        # Term frequency on both sides would square the contribution of a term the
        # user happened to repeat.
        weights = query_weights(tokenize("bert bert scaling"))
        assert set(weights.values()) == {1.0}
        assert weights[term_id("bert")] == 1.0

    def test_an_empty_document_has_no_weights(self) -> None:
        assert document_weights([], average_length=10.0) == {}

    def test_a_zero_average_length_is_refused(self) -> None:
        # It would divide by zero inside the normaliser; failing here names the
        # cause instead.
        with pytest.raises(InvariantViolation, match="average document length"):
            document_weights(["a"], average_length=0.0)


class TestAverageLength:
    def test_it_is_the_mean_token_count(self) -> None:
        assert average_length([["a"], ["a", "b", "c"]]) == 2.0

    def test_an_empty_corpus_is_refused_rather_than_returning_zero(self) -> None:
        # Returning 0.0 would poison every weight downstream with a division by
        # zero one call later, far from the cause.
        with pytest.raises(InvariantViolation, match="empty corpus"):
            average_length([])
