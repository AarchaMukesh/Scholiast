"""Liveness (FR-O2's M1 subset).

Only `/health/live`, and the restraint is deliberate. §5.5 also lists
`/health/ready` with dependency status, and FR-O2 wants "DB, vector store,
queue, model availability" — but FR-O1 through FR-O3 sit in M3's requirement
list, and a readiness probe that reports "ready" without actually checking its
dependencies is worse than no probe at all: it is a green light nobody verified.

What this answers is the one question M1 needs: is the process up. That is what
`docker compose up` plus one command has to be able to confirm.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health/live")
async def live() -> dict[str, str]:
    """The process is running and the event loop is turning."""
    return {"status": "ok"}
