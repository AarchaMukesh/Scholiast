"""Tenant scoping — the type-level half of the isolation design.

Doc 5 §5.8 enforces isolation at three layers. Layer 2 (the data-access layer) is
the one that matters most, because the realistic failure is not a missing login
page — it is a developer adding a query and forgetting the `WHERE owner_id = ?`.

The mechanism here is deliberately awkward on purpose: every scoped repository
requires a `TenantContext` in its constructor, so a repository that cannot filter
by tenant is one the type checker refuses to build. Forgetting the filter stops
being a code-review question and becomes a compile-time one.

`SystemContext` is the single, named, greppable exception — used by the
reconciliation sweeper and migrations, which legitimately operate across tenants.
Making the escape hatch explicit and rare is what keeps it auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable


@runtime_checkable
class Scope(Protocol):
    """Anything that can constrain a query to a set of rows."""

    @property
    def is_system(self) -> bool:
        """True only for cross-tenant maintenance work."""
        ...


@dataclass(frozen=True, slots=True)
class TenantContext:
    """The authenticated principal, threaded through every data access.

    Immutable and hashable so it can be cached and compared, and so no code path
    can widen its own scope after construction.
    """

    user_id: str

    def __post_init__(self) -> None:
        if not self.user_id or not self.user_id.strip():
            # An empty tenant id would silently match nothing — or, worse,
            # everything, depending on how the adapter builds its predicate.
            raise ValueError("TenantContext requires a non-empty user_id")

    @property
    def is_system(self) -> bool:
        return False

    def owns(self, owner_id: str | None) -> bool:
        return owner_id is not None and owner_id == self.user_id


@dataclass(frozen=True, slots=True)
class SystemContext:
    """Unscoped access, for maintenance paths only.

    Legitimate users: the orphaned-vector reconciliation sweeper (§5.6.4), index
    rebuilds, and migrations. Nothing that serves an HTTP request should ever
    construct one — and because it is a distinct type with a distinct name, a
    reviewer can grep for it.
    """

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            # Forces the caller to state *why* it is bypassing tenant scoping,
            # which turns an invisible privilege escalation into a readable one.
            raise ValueError("SystemContext requires a stated reason")

    @property
    def is_system(self) -> bool:
        return True

    def owns(self, owner_id: str | None) -> bool:  # noqa: ARG002 - system sees all
        return True


class TenantIsolationViolation(RuntimeError):
    """Raised when a scoped read returns data the scope does not own.

    This is a defence-in-depth assertion, not a control-flow branch: if it ever
    fires, layer 2 has a bug and the correct response is to fail loudly rather
    than to filter the row out and continue serving.
    """

    def __init__(self, entity: str, entity_id: str, scope: str) -> None:
        super().__init__(
            f"{entity} {entity_id!r} is not owned by scope {scope!r} — "
            "tenant filter was not applied"
        )


#: Reason strings for the sanctioned system-context uses. Keeping them as
#: constants makes the full set of cross-tenant operations enumerable.
SWEEPER_REASON: Final = "reconciliation sweeper: remove residue orphaned by deletion"
#: D99. A workspace cascade runs after its owner's request has already returned,
#: and it must run even if the owner cannot be resolved — a workspace nobody can
#: be found for would otherwise become permanently unsweepable residue, which is
#: the one outcome FR-W3 forbids. The cost of not being tenant-scoped is that
#: the workspace id is trusted input, which is why `DeleteWorkspaceUseCase`
#: refuses any workspace that is not already marked deleted.
CASCADE_REASON: Final = "workspace cascade: delete on behalf of an owner who has gone"
REBUILD_REASON: Final = "index rebuild from source of truth"
MIGRATION_REASON: Final = "schema migration"
#: D50. A worker cannot know whose job it is until it has claimed one, so the
#: claim itself is genuinely cross-tenant. It is also the *only* cross-tenant
#: step: once the owning user is known the pipeline re-enters layer 2 under a
#: `TenantContext`, so every write the worker makes is tenant-filtered.
WORKER_CLAIM_REASON: Final = "ingestion worker: claim a job before its owner is known"
#: D84. Authentication is the bootstrap case: resolving an account by email, or a
#: session by refresh fingerprint, is what *produces* a principal, so it cannot
#: already be holding one. The reach is deliberately narrow — these paths touch
#: `users` and `refresh_tokens` and nothing else, never workspace data — and the
#: moment a request has a principal it re-enters layer 2 under a `TenantContext`.
AUTHENTICATION_REASON: Final = "authentication: resolve a principal before one exists"
