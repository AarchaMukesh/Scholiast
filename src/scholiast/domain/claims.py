"""D179's Rule 2 (`claims/1.0.0`): an answer cut into claims, each with the labels it cites.

Groundedness, citation support and coverage are all rates over claims (ML-12,
FR-Q11), so what counts as one claim, and which passage it cites, decides every
number phase 2 reports. D179 fixed the rule in one sentence: a claim is a
sentence; a citation label belongs to the sentence it sits in, and a label
standing alone attaches to the sentence before it; a sentence is non-trivial
unless it is a refusal statement or carries no content beyond framing. This
module is that sentence as code, built on the 167 train answers alone and frozen
by hash before any eval answer exists (`eval.answers.FROZEN_RULE_HASHES`).

What the train answers forced the rule to say that D179's sentence did not:

- **A line is a boundary.** The model writes lists and headed blocks with no
  terminal punctuation, so a line break ends a sentence as a full stop does.
- **The model echoes passage headings.** It often writes `[C3] <title> ·
  <section>` on a line of its own before the passage's content, because that is
  how `EvidenceBlock` renders a passage. The shown headings are removed first, and
  a label left alone at the start of a line is a heading: it labels the rest of
  its paragraph, or the next paragraph when nothing follows it in its own. A
  label standing alone after a sentence on the same line attaches to that
  sentence, as D179 says.
- **A colon introduces a list.** A sentence ending in a colon is framing, and its
  labels carry to the list items directly after it.
- **A refusal can share a sentence with a claim.** "X is Y, but the passage does
  not say Z" states a refusal and makes a claim. Such a sentence is a claim that
  also refuses; only a sentence whose every clause refuses is a refusal statement.

Only the `[Cn]` markers layer 2 audits are citations (`audit_citations`): a bare
`C4` or `(C3)` is prose, so a claim's labels and the answer's citation validity
are read off the same marks.

Does not establish: a sentence is not an atomic claim, so a sentence carrying two
claims counts once (D179); refusal statements are recognised by a closed list of
English phrasings, so a refusal worded outside it reads as a claim.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, unique

# Rule 2 must see exactly the labels layer 2 audits, or coverage and validity
# would be counted over different marks. One pattern, owned by retrieval.
from scholiast.domain.retrieval import _CITATION_MARKER, _LABEL

CLAIMS_VERSION = "claims/1.0.0"

#: What a heading echo may carry after the heading itself (`EvidenceBlock.render`).
OTHER_WORK_SUFFIX = " — describes other work, not this paper's contribution"


@unique
class ClaimKind(StrEnum):
    """What a sentence of an answer is, for D179's metrics."""

    CLAIM = "claim"  # non-trivial: groundedness and coverage are counted over these
    REFUSAL = "refusal"  # every clause states the evidence does not answer
    FRAMING = "framing"  # no content beyond framing, or a colon introducing a list


@dataclass(frozen=True, slots=True)
class Claim:
    """One sentence of an answer, with the labels Rule 2 attaches to it."""

    index: int
    #: The sentence as the model wrote it, headings removed: what a labeller reads.
    source: str
    #: The sentence with its citation markers and attributions removed: what NLI reads.
    text: str
    #: Distinct labels, in the order they were attached.
    labels: tuple[str, ...]
    kind: ClaimKind
    #: Whether any clause states the evidence does not answer (Rule 1 reads this).
    refuses: bool

    @property
    def trivial(self) -> bool:
        return self.kind is not ClaimKind.CLAIM


# --- refusal statements -----------------------------------------------------------

#: Curly quotes, by code point: the model writes them, and the source should not hide them.
_LEFT_DOUBLE, _RIGHT_DOUBLE, _RIGHT_SINGLE = chr(0x201C), chr(0x201D), chr(0x2019)

_ADV = r"(?:\w+ly\s+)?"
_NOT = r"(?:\b(?:does|do|did)\s*(?:not|n't)|\bfail(?:s|ed)?\s+to)"
#: Verbs about what a text says: negated, they are about the evidence whatever the subject.
_SAYING = (
    r"(?:state|mention|specify|say|describe|define|detail|explain|discuss|clarify|"
    r"indicate|address|reveal|tell)"
)
#: Verbs that are about the evidence only when the evidence is their subject, or hedged.
_GIVING = (
    r"(?:provide|give|include|report|contain|list|present|compare|offer|support|answer|cover|show)"
)
_EVIDENCE = (
    r"(?:evidence|passages?|text|context|excerpts?|documents?|sources?|paper|article|abstract|"
    r"information)"
)
#: Participles about what a text says: negated, they are about the evidence ("is not stated").
_SAID = (
    r"(?:stated|mentioned|specified|described|defined|discussed|detailed|explained|addressed|"
    r"clarified|clear|explicit|answered|quantified)"
)
#: Participles that can be a fact about the work ("the reward is not provided at each step"):
#: about the evidence only when hedged, or when the evidence is named after them.
_GIVEN = (
    r"(?:provided|given|included|reported|listed|presented|indicated|covered|found|available|"
    r"known)"
)
_NOT_BE = (
    r"\b(?:(?:is|are|was|were)\s+not|isn't|aren't|wasn't|weren't|"
    r"(?:has|have|had)\s*(?:not|n't)\s+been)"
)
_ABSENT = (
    r"(?:mention|information|indication|statement|passage|definition|description|explanation|"
    r"details?|evidence|reference|discussion)"
)

#: A clause that states the evidence does not answer. Closed list, from the train answers.
_REFUSAL = re.compile(
    "|".join(
        [
            rf"{_NOT}\s+{_ADV}{_SAYING}\b",
            rf"\b{_EVIDENCE}\b[^.;]{{0,40}}?{_NOT}\s+{_ADV}{_GIVING}\b",
            rf"{_NOT}\s+(?:explicitly|directly|specifically|clearly|further)\s+{_GIVING}\b",
            rf"{_NOT}\s+{_ADV}(?:provide|give|include|offer|contain)\s+(?:any\s+|an?\s+|the\s+)?"
            r"(?:further|more|additional|specific|explicit|detailed|exact|precise|direct|clear)\b",
            rf"{_NOT_BE}\s+{_ADV}{_SAID}\b",
            rf"{_NOT_BE}\s+(?:explicitly|directly|specifically|clearly)\s+{_GIVEN}\b",
            rf"{_NOT_BE}\s+{_ADV}{_GIVEN}\b[^.;]{{0,40}}?\b(?:in|by|from)\s+"
            rf"(?:the|this|these|any)\s+(?:\w+\s+)?{_EVIDENCE}\b",
            r"\bno\s+(?:\w+\s+){0,6}?(?:is|are|was|were)\s+"
            rf"{_ADV}(?:mentioned|stated|given|provided|reported|specified|described|found)\b",
            rf"\b(?:there\s+(?:is|are|was|were)\s+no|no|without(?:\s+any)?)\s+(?:\w+\s+){{0,2}}?"
            rf"{_ABSENT}s?\b",
            r"\bnone\s+of\s+the\s+(?:\w+\s+)?(?:passages|evidence|excerpts|documents|sources|texts)",
            r"\bno\s+(?:clear\s+|direct\s+|definitive\s+|explicit\s+|specific\s+)?answer\b",
            r"\bnot\s+(?:directly\s+)?answerable\b",
            r"\b(?:I|we)\s+(?:could\s*(?:not|n't)|can\s*(?:not|'t)|cannot|(?:was|am|were)\s+"
            r"unable\s+to|did\s*(?:not|n't)|do\s*(?:not|n't))\s+"
            rf"{_ADV}(?:find|locate|determine|identify|answer|tell|see|know)\b",
            r"\b(?:cannot|can\s*(?:not|'t)|could\s*(?:not|n't))\s+be\s+"
            r"(?:determined|answered|found|inferred|established|identified)\b",
            r"\bunclear\b",
            r"\b(?:difficult|impossible|hard)\s+to\s+(?:determine|say|tell|know|answer|infer|establish)\b",
            r"\b(?:more|further|additional)\s+(?:information|context|details|evidence)\s+"
            r"(?:is|are|would\s+be|will\s+be)\s+(?:needed|required|necessary)\b",
        ]
    ),
    re.IGNORECASE,
)

#: Where a sentence divides into clauses that can disagree: "X, but the text does not say Y".
_CLAUSE_BREAK = re.compile(
    r",?\s+but\s+|;\s*|,\s*however,?\s+|,?\s+although\s+|,?\s+though\s+|,\s*while\s+|"
    r",?\s+whereas\s+|,\s*yet\s+",
    re.IGNORECASE,
)
_SUBORDINATE_FIRST = re.compile(r"^(?:while|although|though|whereas)\b[^,]*,", re.IGNORECASE)
_WORD = re.compile(r"[A-Za-z0-9][\w'-]*")
#: Fewer words than this cannot carry a claim of their own ("though", "but not Estonian").
CLAIM_CLAUSE_WORDS = 3


def states_refusal(text: str) -> bool:
    """Whether `text` contains a statement that the evidence does not answer."""
    return _REFUSAL.search(text.replace(_RIGHT_SINGLE, "'")) is not None


def _clauses(sentence: str) -> list[str]:
    head = _SUBORDINATE_FIRST.match(sentence)
    parts = [head.group(0), sentence[head.end() :]] if head else [sentence]
    return [clause for part in parts for clause in _CLAUSE_BREAK.split(part)]


def _claims_something(sentence: str) -> bool:
    """Whether any clause of `sentence` says something other than a refusal."""
    return any(
        not states_refusal(clause) and len(_WORD.findall(clause)) >= CLAIM_CLAUSE_WORDS
        for clause in _clauses(sentence)
    )


# --- sentences --------------------------------------------------------------------

_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)]|\(?[a-z]\))\s+")
#: A full stop, a quote or bracket closing after it, then a capital, digit, quote or label.
_SENTENCE_END = re.compile(
    rf"[.!?]+[\"{_RIGHT_DOUBLE}{_RIGHT_SINGLE})\]]*\s+(?=[A-Z0-9\"{_LEFT_DOUBLE}(\[])"
)
#: Words a full stop follows without ending a sentence.
_ABBREVIATIONS = frozenset(
    {"e.g", "i.e", "al", "etc", "vs", "cf", "fig", "figs", "eq", "eqs", "tab", "no", "approx"}
    | {"resp", "incl", "sec", "dr", "mr", "ms", "st", "viz", "ca"}
)
_LAST_WORD = re.compile(r"(\S+)$")
_ALNUM = re.compile(r"[A-Za-z0-9]")


def _inside_quotes(text: str) -> bool:
    return text.count('"') % 2 == 1 or text.count(_LEFT_DOUBLE) > text.count(_RIGHT_DOUBLE)


def _split_sentences(line: str) -> list[str]:
    """A line's sentences. A quotation is part of the sentence that introduces it."""
    pieces: list[str] = []
    start = 0
    for end in _SENTENCE_END.finditer(line):
        before = line[start : end.start()]
        word = _LAST_WORD.search(before)
        if word is not None:
            token = word.group(1).lstrip('("' + _LEFT_DOUBLE).lower()
            if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue
        if _inside_quotes(line[start : end.end()].rstrip()):
            continue
        pieces.append(line[start : end.end()].strip())
        start = end.end()
    pieces.append(line[start:].strip())
    return [piece for piece in pieces if piece]


def _labels(text: str) -> list[str]:
    return [
        label
        for marker in _CITATION_MARKER.finditer(text)
        for label in _LABEL.findall(marker.group(1))
    ]


def _unlabelled(text: str) -> str:
    return _CITATION_MARKER.sub(" ", text)


#: Words that join labels without saying anything: "[C1], [C2] and [C3]." is labels alone.
_JOINERS = re.compile(r"\b(?:and|or|also)\b", re.IGNORECASE)


def _has_content(text: str) -> bool:
    return _ALNUM.search(_JOINERS.sub(" ", _unlabelled(text))) is not None


# --- what NLI reads ---------------------------------------------------------------

_MARKERS = r"\[C\d+(?:\s*,\s*C\d+)*\](?:\s*(?:,\s*and|,|and)\s*\[C\d+(?:\s*,\s*C\d+)*\])*"
#: "According to [C1]," or "as mentioned in [C2]": where a claim came from, not what it says.
_ATTRIBUTION_PHRASE = re.compile(
    r"(?:,\s*)?\b(?:(?:as\s+)?(?:stated|mentioned|described|shown|noted|reported)\s+in|"
    r"according\s+to|from|in|based\s+on)\s+(?:the\s+)?(?:passages?\s+)?" + _MARKERS + r"\s*,?",
    re.IGNORECASE,
)
#: "[C5] mentions that": the passage as the sentence's subject.
_ATTRIBUTION_VERB = re.compile(
    r"(?:\bthe\s+)?(?:\bpassages?\s+)?"
    + _MARKERS
    + r"\s*(?:(?:also|further|explicitly|specifically|does|do)\s+)*"
    r"(?:states?|mentions?|notes?|reports?|says?|explains?|describes?|indicates?|shows?|"
    r"suggests?|highlights?|presents?|lists?|provides?)\b\s*(?:that\s+)?:?\s*",
    re.IGNORECASE,
)


def _for_nli(source: str) -> str:
    """The sentence as a hypothesis: no markers, no "According to [C1]," attribution."""
    text = _ATTRIBUTION_PHRASE.sub(" ", source.strip())
    text = _ATTRIBUTION_VERB.sub(" ", text)
    text = _CITATION_MARKER.sub(" ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r",(?:\s*,)+", ",", text)
    text = re.sub(r"^[\s,;:]+", "", text).strip()
    return text[:1].upper() + text[1:]


# --- the rule ---------------------------------------------------------------------


@dataclass(slots=True)
class _Segment:
    source: str
    labels: list[str]
    paragraph: int
    line: int
    item: bool
    first_on_line: bool
    content: bool
    #: The colon sentence that introduced this list item, if one did.
    intro: str | None = None


def _segments(answer: str, headings: Sequence[str]) -> list[_Segment]:
    text = answer.replace("\r\n", "\n")
    echoes = {h + OTHER_WORK_SUFFIX for h in headings if h} | {h for h in headings if h}
    for echo in sorted(echoes, key=len, reverse=True):
        text = text.replace(echo, "")
    segments: list[_Segment] = []
    line_number = 0
    for paragraph, block in enumerate(re.split(r"\n[ \t]*\n", text)):
        for raw in block.split("\n"):
            line_number += 1
            item = _LIST_ITEM.match(raw)
            body = raw[item.end() :] if item else raw
            pieces = _split_sentences(body)
            for position, piece in enumerate(pieces):
                segments.append(
                    _Segment(
                        source=piece,
                        labels=_labels(piece),
                        paragraph=paragraph,
                        line=line_number,
                        item=item is not None,
                        first_on_line=position == 0,
                        content=_has_content(piece),
                    )
                )
    return segments


def _attach(segments: list[_Segment]) -> list[_Segment]:
    """Move every label that stands alone onto the sentences it labels; return the sentences."""
    content = [s for s in segments if s.content]

    def add(targets: Sequence[_Segment], labels: Sequence[str]) -> None:
        for target in targets:
            target.labels.extend(labels)

    for i, segment in enumerate(segments):
        if segment.content or not segment.labels:
            continue
        if not segment.first_on_line:
            # "Claim. [C1]": standing alone after a sentence, it is that sentence's (D179).
            before = [s for s in segments[:i] if s.content and s.line == segment.line]
            add(before[-1:], segment.labels)
            continue
        # A heading: the rest of its paragraph, up to the next heading.
        after: list[_Segment] = []
        for later in segments[i + 1 :]:
            if later.paragraph != segment.paragraph or _is_heading(later):
                break
            if later.content:
                after.append(later)
        earlier = [s for s in content if s.paragraph == segment.paragraph and s.line < segment.line]
        following = [s for s in content if s.paragraph == segment.paragraph + 1]
        previous = [s for s in content if s.line < segment.line]
        if after:
            add(after, segment.labels)
        elif earlier:
            add(earlier[-1:], segment.labels)
        elif following:
            add(_until_heading(following, segments), segment.labels)
        else:
            add(previous[-1:], segment.labels)

    # A colon introduces a list: its labels carry to the items directly after it, and
    # an item that is not a sentence of its own is read as the end of its introduction
    # ("... datasets: CTB5").
    for i, segment in enumerate(content):
        if not _introduces(segment):
            continue
        for later in content[i + 1 :]:
            if not later.item:
                break
            later.labels.extend(segment.labels)
            if not _FULL_STOP.search(later.source):
                later.intro = segment.source
    return content


_FULL_STOP = re.compile(r"[.!?][\"')\]]*\s*$")


def _is_heading(segment: _Segment) -> bool:
    """A line that is only labels, once the headings it echoed are gone."""
    return not segment.content and bool(segment.labels) and segment.first_on_line


def _until_heading(following: list[_Segment], segments: list[_Segment]) -> list[_Segment]:
    """A paragraph's sentences before its own first heading, if it has one."""
    paragraph = following[0].paragraph
    headings = [s.line for s in segments if s.paragraph == paragraph and _is_heading(s)]
    stop = min(headings, default=None)
    return [s for s in following if stop is None or s.line < stop]


def _introduces(segment: _Segment) -> bool:
    """A sentence ending in a colon introduces what follows it.

    Only a line can end a sentence without a full stop, so such a sentence always
    ends its line.
    """
    return segment.source.rstrip().endswith(":")


#: A whole sentence that only frames ("Based on the evidence provided."): D179's example.
_FRAMING_ONLY = re.compile(
    r"^\s*(?:based\s+on|according\s+to)\s+(?:the\s+)?(?:provided\s+)?"
    r"(?:evidence|passages?|text|information)(?:\s+provided)?[\s.,:;]*$",
    re.IGNORECASE,
)


def _kind(segment: _Segment) -> tuple[ClaimKind, bool]:
    plain = _unlabelled(segment.source)
    refuses = states_refusal(plain)
    if refuses and not _claims_something(plain):
        return ClaimKind.REFUSAL, True
    if _introduces(segment) or _FRAMING_ONLY.match(plain):
        return ClaimKind.FRAMING, refuses
    return ClaimKind.CLAIM, refuses


def split_claims(answer: str, *, headings: Sequence[str] = ()) -> tuple[Claim, ...]:
    """Rule 2: every sentence of `answer`, its labels, and whether it is a claim.

    `headings` are the `title · section path` lines of the passages the model was
    shown, which it sometimes copies back; they are removed before anything else.
    """
    content = _attach(_segments(answer, headings))
    claims: list[Claim] = []
    for index, segment in enumerate(content):
        kind, refuses = _kind(segment)
        claims.append(
            Claim(
                index=index,
                source=segment.source,
                text=_for_nli(
                    segment.source if segment.intro is None else f"{segment.intro} {segment.source}"
                ),
                labels=tuple(dict.fromkeys(segment.labels)),
                kind=kind,
                refuses=refuses,
            )
        )
    return tuple(claims)
