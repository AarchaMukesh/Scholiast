"""Grounded question answering (§5.4.3, FR-Q1, FR-Q2, FR-Q5, FR-Q6, D56-D60).

Retrieve, assemble, generate, audit, persist. Phase 7 turns the resulting events
into `text/event-stream`; the streaming *contract* is here, because FR-Q5 is a
property of the answer and not of HTTP, and a use case that returned a finished
string would make the SSE endpoint a lie.

`stream()` is a coroutine that returns an iterator, **not** an async generator —
see its docstring. The distinction is the whole reason validation lives where it
does: a generator's body does not run until its first `__anext__()`, so an error
raised inside one arrives after the response has already committed `200 OK`.

**This is M1's ablation baseline, not the adopted design** (D56). Dense-only,
child-only context, relevance-descending order. The lexical arm, RRF fusion, the
cross-encoder, parent expansion and sandwich placement are M2's — deliberately,
so each one's contribution is a measured number rather than an assumption. Every
one of them is a configuration change here, not a restructuring.

Three properties are load-bearing:

**Never answer from parametric memory.** A workspace with nothing indexed raises
rather than answering (FR-W5), and a retrieval that finds nothing returns an
explicit refusal without spending a generation call. Those are the degenerate
cases; the calibrated abstention threshold τ is M2's, because §6.12 fits it on
labelled data and reports the curve rather than picking a number (D60).

**The citation surface is closed.** The model is shown `[C1]…[Cn]` and nothing
else, and every label it emits is checked against that set (`audit_citations`,
layer 2). Fabricated labels are stripped from the prose and reported — not
silently dropped, because a disclosure that disappears is not one (NFR-U4).

**Degradation is disclosed, never silent.** Every rung of §5.11's ladder that
gets used is named in the final event and persisted on the message.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from scholiast.application.ports.llm import GenerationUnavailable, LLMProvider, TokenUsage
from scholiast.application.ports.ml import Embedder, TextRole
from scholiast.application.ports.repositories import (
    ChunkRepository,
    ConversationRepository,
    PaperRepository,
    WorkspaceRepository,
)
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.prompts import (
    ANSWER_PROMPT_VERSION,
    EvidenceBlock,
    build_answer_prompt,
)
from scholiast.domain.conversation import StageTimings, Turn, TurnRole
from scholiast.domain.errors import WorkspaceNotAvailable, WorkspaceNotQueryable
from scholiast.domain.retrieval import (
    Citation,
    EvidenceSet,
    Query,
    ScoredChunk,
    audit_citations,
    strip_labels,
)

#: Dense candidates fetched before assembly. M2 replaces this with hybrid
#: retrieval into a ~50-candidate pool for the cross-encoder; 20 is what a
#: dense-only arm can usefully offer an 8-slot budget.
DEFAULT_CANDIDATES = 20

#: Evidence passages placed in front of the model. Matches M2's post-rerank
#: budget on purpose, so the ablation compares like with like, and fits §6.9's
#: 4-6k token guidance — "more context is not better context".
DEFAULT_EVIDENCE_LIMIT = 8

#: Said verbatim when retrieval finds nothing. Not a generation call: there is
#: nothing to ground an answer in, and asking the model anyway is precisely the
#: parametric-memory failure FR-Q4 exists to prevent.
NO_EVIDENCE_ANSWER = (
    "I could not find anything in this workspace's papers that speaks to that "
    "question, so I have nothing to ground an answer in."
)

#: §5.11's ladder, named. These strings reach the user and the message row.
DEGRADED_PROVIDER_FAILOVER = "provider_failover"
DEGRADED_GENERATION_UNAVAILABLE = "generation_unavailable"
#: A provider died after tokens had already reached the client. Not a
#: failover case: the answer stops where it stopped.
DEGRADED_GENERATION_TRUNCATED = "generation_truncated"


# --- events ---------------------------------------------------------------
#
# A small closed set rather than dicts: phase 7 has to map each one to an SSE
# event name, and a typed union makes an unhandled case a type error there
# instead of a missing branch nobody notices.


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One passage, resolved far enough for the UI to make it clickable.

    Carries everything the M1 acceptance criterion names — "the correct paper,
    section, page, and character span" — so the client never needs a second
    round trip to render a citation, and the span is the one FR-I5's round trip
    already proved indexes `sections.text`.
    """

    label: str
    chunk_id: str
    paper_id: str
    paper_title: str
    section_path: str
    page: int
    char_start: int
    char_end: int
    text: str
    score: float
    #: Whether this passage reports someone else's contribution. §6.10 rule 5
    #: is unenforceable without it: a model cannot respect related-work
    #: provenance it was never shown, and misattributing a baseline's result to
    #: the paper under discussion is the failure doc 2 §2.3 opens with.
    describes_other_work: bool = False


@dataclass(frozen=True, slots=True)
class EvidenceReady:
    """Emitted before the first token, so the UI can render the panel early."""

    items: tuple[EvidenceItem, ...]
    excluded_paper_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerToken:
    """One streamed fragment of the answer (FR-Q5)."""

    text: str


@dataclass(frozen=True, slots=True)
class AnswerComplete:
    """The final event: the audited answer and everything that explains it."""

    answer: str
    citations: tuple[Citation, ...]
    degraded: tuple[str, ...]
    timings: StageTimings
    prompt_version: str
    provider: str | None = None
    model: str | None = None
    message_id: str | None = None
    fabricated_labels: tuple[str, ...] = ()


AnswerEvent = EvidenceReady | AnswerToken | AnswerComplete


@dataclass
class _Generated:
    """Mutable accumulator for one generation attempt."""

    text: str = ""
    provider: str | None = None
    model: str | None = None
    usage: TokenUsage | None = None
    degraded: list[str] = field(default_factory=list)
    first_token_ms: int = 0
    #: Set immediately *before* the write is attempted, not after it lands —
    #: "attempted", not "persisted", despite the name. Deliberate: a `_persist`
    #: that raises on the ordinary path must not be retried from the cleanup
    #: path, where it would raise again and mask the original error. Do not
    #: move it after the await; that turns a failed write into a double write.
    persisted: bool = False


@dataclass(frozen=True, slots=True)
class _Prepared:
    """Everything settled before a single token is generated.

    Exists so validation and retrieval happen when `stream()` is *awaited*
    rather than when its result is first iterated. An async generator body does
    not run until the first `__anext__()`, so raising inside one means the
    failure surfaces after the HTTP response has already committed `200 OK` —
    and phase 7 could then never answer an unqueryable workspace with the
    RFC 9457 problem document §5.5 specifies.
    """

    conversation_id: str
    question: str
    evidence: EvidenceSet
    items: tuple[EvidenceItem, ...]
    retrieval_ms: int
    assembly_ms: int


class AnswerQuestionUseCase:
    """One grounded, streamed, cited answer."""

    def __init__(
        self,
        *,
        workspaces: WorkspaceRepository,
        papers: PaperRepository,
        chunks: ChunkRepository,
        conversations: ConversationRepository,
        embedder: Embedder,
        index: VectorIndex,
        providers: Sequence[LLMProvider],
        candidates: int = DEFAULT_CANDIDATES,
        evidence_limit: int = DEFAULT_EVIDENCE_LIMIT,
    ) -> None:
        self._workspaces = workspaces
        self._papers = papers
        self._chunks = chunks
        self._conversations = conversations
        self._embedder = embedder
        self._index = index
        self._providers = tuple(providers)
        self._candidates = candidates
        self._evidence_limit = evidence_limit

    async def stream(
        self, *, workspace_id: str, conversation_id: str, question: str
    ) -> AsyncIterator[AnswerEvent]:
        """Validate and retrieve, then hand back the token stream.

        A plain `async def` returning an iterator, deliberately, rather than an
        async generator. The shapes look alike and behave differently at exactly
        the moment that matters: a generator's body does not run until its first
        `__anext__()`, so `WorkspaceNotQueryable` raised inside one would surface
        *after* a streaming response had already committed its status line.
        Phase 7 needs these as ordinary errors it can render as RFC 9457
        problem documents (§5.5, FR-A2) — which means they have to happen while
        the caller is still awaiting, before a single byte is written.

        Everything expensive but non-streaming happens here too: retrieval and
        assembly are awaited eagerly, so `EvidenceReady` is already decided when
        the stream opens.
        """
        prepared = await self._prepare(
            workspace_id=workspace_id, conversation_id=conversation_id, question=question
        )
        return self._events(prepared)

    async def _prepare(
        self, *, workspace_id: str, conversation_id: str, question: str
    ) -> _Prepared:
        workspace = await self._workspaces.get(workspace_id)
        if workspace is None:
            raise WorkspaceNotAvailable(workspace_id)
        if not workspace.is_queryable:
            # FR-W5: answering here would mean answering from the model's own
            # knowledge of the literature, which is the failure this whole
            # system exists to avoid.
            raise WorkspaceNotQueryable(len(workspace.in_flight_paper_ids))

        started = time.perf_counter()
        scored = await self._retrieve(workspace_id, question)
        retrieval_ms = _elapsed_ms(started)

        assembly_started = time.perf_counter()
        evidence = EvidenceSet.assemble(
            scored,
            limit=self._evidence_limit,
            excluded_paper_ids=workspace.in_flight_paper_ids,
        )
        items = await self._resolve(evidence)
        return _Prepared(
            conversation_id=conversation_id,
            question=question,
            evidence=evidence,
            items=items,
            retrieval_ms=retrieval_ms,
            assembly_ms=_elapsed_ms(assembly_started),
        )

    async def _events(self, prepared: _Prepared) -> AsyncIterator[AnswerEvent]:
        """The streamed half: evidence, tokens, and the audited result.

        The `finally` is the reason this is not a straight line. A consumer that
        stops iterating — an SSE client closing the tab — throws `GeneratorExit`
        at whichever `yield` is suspended, and everything after it would never
        run: the answer already generated, the evidence it used, the timings,
        and the quota already spent would all be discarded because nobody was
        listening. With generation quota as the scarce resource (D26), that is a
        cost worth not paying twice.
        """
        # Deliberately *outside* the try below: abandoning here means the caller
        # left before anything was generated, so there is no answer to lose and
        # no quota to have wasted. Persisting then would write a blank assistant
        # turn for every reader who changed their mind at the evidence panel.
        # The boundary is "generation was entered", not "the stream opened".
        yield EvidenceReady(
            items=prepared.items, excluded_paper_ids=prepared.evidence.excluded_paper_ids
        )

        result = _Generated()
        generation_started = time.perf_counter()
        try:
            if prepared.evidence.is_empty:
                # D60's degenerate case: nothing to ground an answer in, so no
                # generation call is spent finding that out.
                result.text = NO_EVIDENCE_ANSWER
                yield AnswerToken(text=NO_EVIDENCE_ANSWER)
                yield await self._finalise(prepared, result, generation_started)
                return

            async for token in self._generate(prepared.question, prepared.items, result):
                yield AnswerToken(text=token)
            yield await self._finalise(prepared, result, generation_started)
        finally:
            # Reached on abandonment, and on an exception escaping generation.
            # `persisted` makes it a no-op on the ordinary path, so a completed
            # turn is never written twice.
            if not result.persisted:
                await self._finalise(prepared, result, generation_started)

    async def _finalise(
        self, prepared: _Prepared, result: _Generated, generation_started: float
    ) -> AnswerComplete:
        """Audit what was generated, persist the turn, and describe it.

        Called on the ordinary path and again from the cleanup path, which is
        why it sets `result.persisted` rather than trusting its caller: whichever
        one arrives first is the one that writes.
        """
        # Layer 2 (§6.11): every emitted label checked against the set the model
        # was shown. Resolved labels are UNVERIFIED, not VERIFIED — passing a
        # syntactic check does not establish that the chunk supports the claim,
        # and promoting it here would launder one for the other.
        audit = audit_citations(result.text, prepared.evidence)
        # Stripped before the audit, not after: `strip_labels` trims as a side
        # effect of tidying the gaps it leaves, so trimming only here would make
        # the stored answer's whitespace depend on whether the model hallucinated.
        answer = strip_labels(result.text.strip(), audit.fabricated_labels)

        timings = StageTimings(
            retrieval_ms=prepared.retrieval_ms,
            assembly_ms=prepared.assembly_ms,
            first_token_ms=result.first_token_ms,
            generation_ms=_elapsed_ms(generation_started),
        )
        result.persisted = True
        message_id = await self._persist(
            conversation_id=prepared.conversation_id,
            question=prepared.question,
            answer=answer,
            audit_citations_=audit.citations,
            evidence=prepared.evidence,
            result=result,
            timings=timings,
        )
        return AnswerComplete(
            answer=answer,
            citations=audit.citations,
            degraded=tuple(result.degraded),
            timings=timings,
            prompt_version=ANSWER_PROMPT_VERSION,
            provider=result.provider,
            model=result.model,
            message_id=message_id,
            fabricated_labels=audit.fabricated_labels,
        )

    # -- retrieval ---------------------------------------------------------

    async def _retrieve(self, workspace_id: str, question: str) -> list[ScoredChunk]:
        """Dense search over exactly this workspace's indexed papers (D36).

        The id set comes from a tenant-scoped query rather than from the caller,
        so §5.8 layer 3's filter is *derived from* the tenant predicate instead
        of travelling beside it — a caller cannot widen what it never supplied.
        """
        paper_ids = await self._papers.queryable_paper_ids(workspace_id)
        if not paper_ids:
            return []

        # Constructing a Query validates the scope is non-empty and the text is
        # not blank; it is also where M3's conversational rewriting will land,
        # which is why the original text has a home on it already (FR-Q8).
        query = Query(text=question, workspace_id=workspace_id, paper_ids=paper_ids)
        vectors = await self._embedder.embed([query.text], role=TextRole.QUERY)
        return await self._index.search(
            vectors[0], paper_ids=query.paper_ids, limit=self._candidates
        )

    async def _resolve(self, evidence: EvidenceSet) -> tuple[EvidenceItem, ...]:
        """Turn scored chunk ids into passages a citation can point at.

        Chunks are fetched through the scoped repository, so a chunk the vector
        store returned but this tenant cannot reach is dropped rather than
        served — the two stores disagreeing must not become a leak (§5.6.4).
        """
        if evidence.is_empty:
            return ()

        chunks = {
            c.id: c for c in await self._chunks.get_many([c.chunk_id for c in evidence.chunks])
        }
        titles = await self._paper_titles(evidence.paper_ids)

        items: list[EvidenceItem] = []
        for scored in evidence.chunks:
            chunk = chunks.get(scored.chunk_id)
            if chunk is None:
                continue
            label = evidence.label_for(chunk.id)
            if label is None:  # pragma: no cover - assemble() labels everything
                continue
            items.append(
                EvidenceItem(
                    label=label,
                    chunk_id=chunk.id,
                    paper_id=chunk.provenance.paper_id,
                    paper_title=titles.get(chunk.provenance.paper_id, "Unknown paper"),
                    section_path=chunk.provenance.section_path,
                    page=chunk.provenance.page,
                    char_start=chunk.provenance.char_start,
                    char_end=chunk.provenance.char_end,
                    text=chunk.text,
                    score=scored.score,
                    describes_other_work=chunk.provenance.section_kind.describes_other_work,
                )
            )
        return tuple(items)

    async def _paper_titles(self, paper_ids: Sequence[str]) -> dict[str, str]:
        titles: dict[str, str] = {}
        for paper_id in paper_ids:
            paper = await self._papers.get(paper_id)
            if paper is not None:
                titles[paper_id] = paper.title
        return titles

    # -- generation --------------------------------------------------------

    async def _generate(
        self, question: str, items: Sequence[EvidenceItem], result: _Generated
    ) -> AsyncIterator[str]:
        """Walk §5.11's ladder, disclosing every rung that was used.

        A provider is tried until one yields; `GenerationUnavailable` moves to
        the next and records the failover. If every provider is exhausted the
        evidence still went out in `EvidenceReady`, so the user gets retrieval
        without synthesis — which §5.11 calls out as still having standalone
        value, and which is flagged rather than presented as a normal answer.
        """
        system, user = build_answer_prompt(question, [_to_block(i) for i in items])
        started = time.perf_counter()
        failed_over = False

        for provider in self._providers:
            try:
                async for chunk in provider.stream(system=system, user=user):
                    if chunk.text:
                        if not result.text:
                            result.first_token_ms = _elapsed_ms(started)
                        result.text += chunk.text
                        yield chunk.text
                    if chunk.usage is not None:
                        result.usage = chunk.usage
            except GenerationUnavailable:
                if result.text:
                    # Tokens are already with the client, so there is no failing
                    # over: continuing would splice a second provider's prose
                    # onto the first's mid-sentence. The answer is truncated and
                    # says so — an honest partial beats a seamless-looking
                    # Frankenstein (NFR-R4).
                    result.provider = provider.name
                    result.model = provider.model
                    result.degraded.append(DEGRADED_GENERATION_TRUNCATED)
                    return
                failed_over = True
                continue

            if result.text.strip():
                result.provider = provider.name
                result.model = provider.model
                if failed_over:
                    result.degraded.append(DEGRADED_PROVIDER_FAILOVER)
                return
            # A provider that streamed nothing at all failed while looking like
            # it succeeded. Accepting it would serve a blank answer.
            failed_over = True

        # Every rung exhausted. The evidence already went out, so the user has
        # retrieval without synthesis — which §5.11 says still has standalone
        # value, flagged rather than dressed up as a normal answer.
        result.provider = None
        result.model = None
        result.degraded.append(DEGRADED_GENERATION_UNAVAILABLE)

    # -- persistence -------------------------------------------------------

    async def _persist(
        self,
        *,
        conversation_id: str,
        question: str,
        answer: str,
        audit_citations_: tuple[Citation, ...],
        evidence: EvidenceSet,
        result: _Generated,
        timings: StageTimings,
    ) -> str | None:
        """Record both turns (§5.4.3, D59).

        The question is stored too: an evidence panel that shows what an answer
        was based on, next to a question nobody kept, is half a record.
        """
        await self._conversations.append(
            conversation_id, Turn(role=TurnRole.USER, content=question)
        )
        return await self._conversations.append(
            conversation_id,
            Turn(
                role=TurnRole.ASSISTANT,
                content=answer,
                citations=audit_citations_,
                evidence=tuple(evidence.chunks),
                degraded=tuple(result.degraded),
                prompt_version=ANSWER_PROMPT_VERSION,
                provider=result.provider,
                model=result.model,
                timings=timings,
                prompt_tokens=result.usage.prompt_tokens if result.usage else None,
                completion_tokens=result.usage.completion_tokens if result.usage else None,
            ),
        )


def _to_block(item: EvidenceItem) -> EvidenceBlock:
    return EvidenceBlock(
        label=item.label,
        paper_title=item.paper_title,
        section_path=item.section_path,
        text=item.text,
        describes_other_work=item.describes_other_work,
    )


def _elapsed_ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)
