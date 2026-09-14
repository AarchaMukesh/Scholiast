"""The QA surface: conversations and the streamed answer (FR-Q1, FR-Q2, FR-Q5).

**Validation happens before the response starts.** `AnswerQuestionUseCase.stream`
is awaited here, inside the handler, so `WorkspaceNotQueryable` becomes a 409
problem document — phase 6 made it a coroutine returning an iterator for exactly
this reason. Had it stayed an async generator, the error would have surfaced on
the first `__anext__()`, which `StreamingResponse` reaches only after the status
line and headers are already on the wire, and the client would receive a 200
whose body was an error.

**The transport is POST + SSE framing** (D61). §5.5 specifies POST for this
endpoint and D37 said `EventSource`; the browser API can only GET, so the wire
format stays SSE and the M1 page reads it with `fetch`. Keeping the framing means
the endpoint is still EventSource-shaped for any client that wants it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from http import HTTPStatus

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from scholiast.api import schemas
from scholiast.api.deps import AnswerDep, ConversationsDep, WorkspacesDep
from scholiast.application.use_cases.answer_question import (
    AnswerComplete,
    AnswerEvent,
    AnswerToken,
    EvidenceReady,
)
from scholiast.domain.errors import WorkspaceNotAvailable

router = APIRouter(prefix="/v1/workspaces/{workspace_id}/conversations", tags=["chat"])

SSE_MEDIA_TYPE = "text/event-stream"

#: SSE event names. A closed set, because the client switches on them — and
#: because a typo in a string literal is a silently dropped event rather than an
#: error, which is why they are constants rather than inline.
EVENT_EVIDENCE = "evidence"
EVENT_TOKEN = "token"  # noqa: S105 - an SSE event name, not a credential
EVENT_COMPLETE = "complete"


@router.post("", status_code=HTTPStatus.CREATED, response_model=schemas.ConversationCreated)
async def create_conversation(
    workspace_id: str, body: schemas.CreateConversation, repo: ConversationsDep
) -> schemas.ConversationCreated:
    created = await repo.create(workspace_id, title=body.title)
    if created is None:
        raise WorkspaceNotAvailable(workspace_id)
    return schemas.ConversationCreated(conversation_id=created)


@router.get("/{conversation_id}/messages", response_model=list[schemas.TurnOut])
async def list_messages(
    workspace_id: str, conversation_id: str, repo: ConversationsDep
) -> list[schemas.TurnOut]:
    """Conversation history with its evidence and accounting (FR-Q6, FR-Q9)."""
    conversation = await repo.get(conversation_id)
    if conversation is None or conversation.workspace_id != workspace_id:
        raise WorkspaceNotAvailable(conversation_id)
    return [
        schemas.TurnOut(
            role=turn.role.value,
            content=turn.content,
            citations=[
                schemas.CitationOut(
                    label=c.label,
                    chunk_id=c.chunk_id,
                    status=c.status.value,
                    claim_text=c.claim_text,
                )
                for c in turn.citations
            ],
            degraded=list(turn.degraded),
            provider=turn.provider,
            model=turn.model,
            prompt_version=turn.prompt_version,
            latency_ms=turn.timings.as_dict() if turn.timings else None,
        )
        for turn in conversation.turns
    ]


@router.post("/{conversation_id}/messages")
async def ask(
    workspace_id: str,
    conversation_id: str,
    body: schemas.AskQuestion,
    use_case: AnswerDep,
    workspaces: WorkspacesDep,
) -> StreamingResponse:
    """Ask a question; stream the grounded answer (FR-Q1, FR-Q2, FR-Q5).

    The `await` below is load-bearing: it runs validation, retrieval and evidence
    assembly *before* `StreamingResponse` is constructed, so anything that goes
    wrong there is still an ordinary exception that the problem-document handlers
    can turn into a 404 or a 409.
    """
    if await workspaces.get(workspace_id) is None:
        raise WorkspaceNotAvailable(workspace_id)

    events = await use_case.stream(
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        question=body.question,
    )
    return StreamingResponse(
        _frames(events),
        media_type=SSE_MEDIA_TYPE,
        headers={
            # Streaming dies quietly behind a buffering proxy: the whole response
            # is held until the generator finishes, so the user sees nothing for
            # fifteen seconds and then everything. These two headers are what
            # keep time-to-first-token (NFR-P3) a real number in deployment.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


async def _frames(events: AsyncIterator[AnswerEvent]) -> AsyncIterator[str]:
    """Render typed events as SSE frames.

    The match is exhaustive over a closed union, so adding an event type without
    deciding how it reaches the client is a type error here rather than a frame
    that never arrives.
    """
    async for event in events:
        match event:
            case EvidenceReady():
                yield _frame(
                    EVENT_EVIDENCE,
                    {
                        "items": [
                            schemas.EvidenceOut(
                                label=i.label,
                                chunk_id=i.chunk_id,
                                paper_id=i.paper_id,
                                paper_title=i.paper_title,
                                section_path=i.section_path,
                                page=i.page,
                                char_start=i.char_start,
                                char_end=i.char_end,
                                text=i.text,
                                score=i.score,
                                describes_other_work=i.describes_other_work,
                            ).model_dump()
                            for i in event.items
                        ],
                        "excluded_paper_ids": list(event.excluded_paper_ids),
                    },
                )
            case AnswerToken():
                yield _frame(EVENT_TOKEN, {"text": event.text})
            case AnswerComplete():
                yield _frame(
                    EVENT_COMPLETE,
                    {
                        "answer": event.answer,
                        "citations": [
                            schemas.CitationOut(
                                label=c.label,
                                chunk_id=c.chunk_id,
                                status=c.status.value,
                                claim_text=c.claim_text,
                            ).model_dump()
                            for c in event.citations
                        ],
                        "fabricated_labels": list(event.fabricated_labels),
                        "degraded": list(event.degraded),
                        "provider": event.provider,
                        "model": event.model,
                        "prompt_version": event.prompt_version,
                        "message_id": event.message_id,
                        "latency_ms": event.timings.as_dict(),
                    },
                )


def _frame(event: str, payload: dict[str, object]) -> str:
    """One SSE frame.

    `json.dumps` without newlines is what keeps this a single `data:` line — an
    embedded newline would split the frame and the client would parse half a
    document. The blank line terminates the event, per the SSE grammar.
    """
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n"
