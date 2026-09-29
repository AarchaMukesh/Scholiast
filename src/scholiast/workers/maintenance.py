"""The cascade and the sweep, wired to real stores (§5.6.4, FR-W3, D99-D102).

The counterpart to `workers.ingestion`: same shape, same session discipline,
different work. Both are constructed with a session factory and the adapters
they need, and both open one transaction per unit of work so a crash rolls back
to a state the next attempt can resume from.

**Why this runs under `SystemContext` rather than the owner's scope.** The
ingestion worker resolves the owning user and re-enters layer 2 for everything
after the claim (D50), and this deliberately does not. A cascade runs after its
owner's request has already returned 202, and it has to run *even if the owner
cannot be resolved* — a workspace nobody can be found for would otherwise become
residue no sweep could ever remove, which is the one outcome FR-W3 forbids.

That is a real cost, stated rather than glossed: with no tenant predicate, the
workspace id is trusted input, and it arrives over a broker. `DeleteWorkspace
UseCase` is what pays it back — it refuses any workspace that is not already
marked deleted, so the only thing this process can destroy is work an
authenticated owner already asked for.

**One transaction per workspace, not one per sweep.** A sweep that deleted ten
workspaces in one transaction would roll all ten back because the eleventh
failed, and would hold locks on every row it touched for the duration. Per
workspace, a crash keeps whatever finished and leaves the rest marked — which
is exactly the state `list_deleting` is built to find.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from scholiast.application.ports.storage import ObjectStore
from scholiast.application.ports.tenancy import (
    CASCADE_REASON,
    SWEEPER_REASON,
    SystemContext,
)
from scholiast.application.ports.vector import VectorIndex
from scholiast.application.use_cases.delete_workspace import (
    DeleteWorkspaceResult,
    DeleteWorkspaceUseCase,
)
from scholiast.application.use_cases.reconcile import ReconcileResult, ReconcileUseCase
from scholiast.infrastructure.db.repositories import SqlPaperRepository, SqlWorkspaceRepository

log = structlog.get_logger(__name__)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


class MaintenanceWorker:
    """Sweeps deleted workspaces, and reconciles the three stores."""

    def __init__(
        self,
        *,
        sessions: SessionFactory,
        index: VectorIndex,
        store: ObjectStore,
    ) -> None:
        self._sessions = sessions
        self._index = index
        self._store = store

    async def delete_workspace(self, workspace_id: str) -> DeleteWorkspaceResult:
        """Run one workspace's cascade (§5.6.4 step 2)."""
        scope = SystemContext(reason=CASCADE_REASON)
        async with self._sessions() as session:
            use_case = DeleteWorkspaceUseCase(
                workspaces=SqlWorkspaceRepository(session, scope),
                papers=SqlPaperRepository(session, scope),
                index=self._index,
                store=self._store,
            )
            result = await use_case.execute(workspace_id)
            await session.commit()

        if result.swept:
            log.info(
                "cascade.swept",
                workspace_id=workspace_id,
                papers_purged=result.papers_purged,
                vectors_deleted=result.vectors_deleted,
                blobs_deleted=result.blobs_deleted,
            )
        else:
            # Not an error. The workspace was never marked, or a concurrent
            # sweep finished it first — both are states this converges to
            # rather than fails on (D99).
            log.info("cascade.nothing_to_sweep", workspace_id=workspace_id)
        return result

    async def reconcile(self) -> ReconcileResult:
        """Run one reconciliation pass (§5.6.4 step 4).

        **Logged at warning when it finds anything**, because it should find
        nothing: every count above zero is residue some earlier step failed to
        remove, and a sweeper that quietly repairs drift is indistinguishable
        from one that has no drift to repair. The two mean opposite things about
        the rest of the system, so the log has to tell them apart.
        """
        scope = SystemContext(reason=SWEEPER_REASON)
        async with self._sessions() as session:
            use_case = ReconcileUseCase(
                workspaces=SqlWorkspaceRepository(session, scope),
                papers=SqlPaperRepository(session, scope),
                index=self._index,
                store=self._store,
            )
            result = await use_case.execute()
            await session.commit()

        found = (
            result.cascades_resumed
            or result.papers_purged
            or result.vectors_deleted
            or result.blobs_deleted
        )
        # Distinct event names, not one event with a level: "the sweep found
        # residue" is the thing worth alerting on, and it has to be greppable
        # separately from the clean pass that runs every five minutes.
        emit = log.warning if found else log.info
        emit(
            "reconcile.found_residue" if found else "reconcile.clean",
            cascades_resumed=result.cascades_resumed,
            papers_purged=result.papers_purged,
            vectors_deleted=result.vectors_deleted,
            blobs_deleted=result.blobs_deleted,
        )
        return result
