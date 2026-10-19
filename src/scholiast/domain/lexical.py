"""BM25 term weighting for the lexical arm (doc 6 §6.5, ML-6, FR-Q3).

**Why a lexical arm exists at all**, in this domain specifically: dense retrieval
compresses `LLaMA-2-70B` and `LLaMA-3-8B` into the same semantic neighbourhood,
and this corpus is made of queries naming exactly that kind of thing — a dataset,
a metric, a model, a number. BM25 knows they are different strings. Doc 6 calls
this "far stronger than the generic 'hybrid usually helps'", and phase 2's measured
baseline agrees from the other direction: the dense arm's dominant failure mode is
*paper confusion*, and an exact identifier is precisely what disambiguates papers.

**Deliberately pure**, like the chunker beside it: a string in, term weights out.
No tokenizer dependency, no store, no I/O. That is what lets the `[SPIKE]` doc 6
requires be run against real corpus text as a plain function call.

**The split of labour with the store.** Document vectors carry the BM25 *term
frequency* component including length normalisation, which needs a corpus-level
average document length. Query vectors carry 1.0 per distinct term. The **IDF
factor is applied by Qdrant** via `Modifier.IDF`, computed from the collection's
own statistics — which is what doc 6 means by "IDF-aware scoring" living in the
store, and what keeps the ranking correct as the corpus grows without re-weighting
every document.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from scholiast.domain.errors import InvariantViolation

#: Bump on ANY behavioural change here. Stamped alongside the chunker and embedder
#: versions (NFR-M5): a change to tokenisation or weighting changes every sparse
#: vector in the index and therefore every lexical number in every report.
#: Production stamps it on every hybrid point, and a hybrid read is refused until
#: every point carries the current one (D172) - so a bump is a full reindex.
# 1.0.0: initial - identifier-preserving tokenizer, BM25 with sub-token expansion.
# 1.1.0: indexing weights use the fixed AVERAGE_LENGTH below (D172).
LEXICAL_VERSION = "bm25-identifier/1.1.0"

#: Saturation and length-normalisation constants. The standard BM25 defaults; they
#: are not tuned here because tuning them on the eval set before the ablation has
#: measured the arm at all would be fitting a hyperparameter to the test.
BM25_K1 = 1.2
BM25_B = 0.75

#: BM25's `avgdl` for everything production indexes, FIXED rather than measured on
#: the live corpus (D172). It is D-RET's mean over 7,854 chunks, under chunker
#: structure-aware/1.2.0 and tokenizer bm25-identifier/1.0.0, which is why the
#: published, arms and gate numbers describe production's weighting exactly.
#:
#: Fixed because it is baked into each chunk's stored weights at write time,
#: unlike IDF, which the store applies to every chunk alike at query time. A
#: running average would weight two identical chunks differently by ingestion
#: date, and a rebuild would no longer reproduce the index (NFR-R6, D101). The
#: cost is calibration: production's own chunks may average something else. The
#: chunker bounds chunk length, so the value mostly describes the chunker; a
#: chunker or tokenizer change therefore means revisiting it, and changing it
#: means a decisions-log entry, a LEXICAL_VERSION bump and a full reindex.
AVERAGE_LENGTH = 144.8731856378915

#: A token is a run of alphanumerics, optionally joined by the punctuation that
#: appears *inside* scientific identifiers rather than between words. This is the
#: whole reason for a custom tokenizer: a naive `\w+` split shatters `LLaMA-2-70B`
#: into `llama`, `2`, `70b`, destroys `nDCG@10`, and turns `0.847` into two
#: integers - discarding exactly the rare-term signal the lexical arm exists to
#: capture.
_TOKEN = re.compile(r"[a-z0-9]+(?:[-._@/][a-z0-9]+)*")

#: Compounds also emit their parts, so a query for `LLaMA` still matches a document
#: saying `LLaMA-2-70B`. Both forms are indexed: the compound carries the precise
#: signal and the parts carry the recall. Without this the tokenizer would be
#: *more* exact than is useful - a query naming a model family would miss every
#: document naming a specific size of it.
_SPLIT = re.compile(r"[-._@/]")

#: Below this, a sub-token is noise rather than signal: the `2` in `LLaMA-2-70B`
#: matches every version number in the corpus and carries no disambiguating power.
#: The compound form keeps that information; the part does not need to.
_MIN_SUBTOKEN = 2


def tokenize(text: str) -> list[str]:
    """Lowercase tokens, preserving scientific identifiers and emitting their parts.

    Order is preserved and duplicates are kept, because term *frequency* is the
    input to BM25 - deduplicating here would silently turn it into a binary model.
    """
    tokens: list[str] = []
    for match in _TOKEN.finditer(text.lower()):
        compound = match.group()
        tokens.append(compound)
        parts = [p for p in _SPLIT.split(compound) if len(p) >= _MIN_SUBTOKEN]
        if len(parts) > 1:
            tokens.extend(parts)
    return tokens


def term_id(term: str) -> int:
    """Stable 32-bit id for a term. Qdrant sparse vectors are indexed by integer.

    A hash rather than a learned vocabulary, so indexing needs no fitted
    dictionary and a term unseen at build time still matches at query time.
    Collisions are possible and are the accepted cost: at 32 bits over a corpus of
    this size they are rare, and a collision costs a little precision on two
    unrelated rare terms rather than any correctness property.

    **Must stay stable across processes**, which Python's `hash()` is not for
    strings - `PYTHONHASHSEED` randomises it per interpreter, so an index built in
    one process would be unqueryable from another. That failure is silent: every
    lookup simply misses.
    """
    # FNV-1a, 32-bit. Chosen for being three lines and deterministic, not for
    # cryptographic properties - it is a dictionary key, not a digest.
    value = 0x811C9DC5
    for byte in term.encode("utf-8"):
        value = ((value ^ byte) * 0x01000193) & 0xFFFFFFFF
    return value


def document_weights(
    tokens: Sequence[str], *, average_length: float, k1: float = BM25_K1, b: float = BM25_B
) -> dict[int, float]:
    """BM25 term-frequency weights for one document.

    `tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))` - saturation so the
    tenth occurrence of a term adds far less than the second, and length
    normalisation so a long section does not outrank a short one merely by
    containing more words.

    The IDF factor is deliberately absent: Qdrant applies it from collection
    statistics (`Modifier.IDF`). Computing it here would freeze each document's
    weights against the corpus as it stood at index time, so every later insert
    would quietly make the older documents' scores wrong.
    """
    if average_length <= 0:
        raise InvariantViolation(f"average document length must be positive, got {average_length}")
    if not tokens:
        return {}
    length = len(tokens)
    norm = k1 * (1.0 - b + b * length / average_length)
    weights: dict[int, float] = {}
    for term, count in Counter(tokens).items():
        weights[term_id(term)] = count * (k1 + 1.0) / (count + norm)
    return weights


def chunk_weights(text: str) -> dict[int, float]:
    """Production's lexical weights for one chunk: its text, at the fixed `AVERAGE_LENGTH`.

    The one function ingestion and the rebuild both call, so the two cannot weight
    a chunk differently. It reads the chunk's own text, not the contextual header
    the dense arm embeds (ML-3), exactly as the eval's lexical index did.
    """
    return document_weights(tokenize(text), average_length=AVERAGE_LENGTH)


def query_weights(tokens: Sequence[str]) -> dict[int, float]:
    """Query-side sparse vector: 1.0 per distinct term.

    The document side already carries saturation and length normalisation, and the
    store supplies IDF, so the dot product of the two *is* the BM25 score. Putting
    term frequency on the query side as well would square the contribution of a
    term the user happened to repeat.
    """
    return {term_id(term): 1.0 for term in dict.fromkeys(tokens)}


def average_length(documents: Iterable[Sequence[str]]) -> float:
    """Mean token count over the corpus - BM25's `avgdl`.

    Computed once over the whole corpus at index time and passed in, rather than
    estimated per batch: a per-batch average would make a document's weight depend
    on which batch it happened to land in, and the same corpus indexed with a
    different batch size would produce different scores.
    """
    total = 0
    count = 0
    for tokens in documents:
        total += len(tokens)
        count += 1
    if not count:
        raise InvariantViolation("cannot compute average document length of an empty corpus")
    return total / count
