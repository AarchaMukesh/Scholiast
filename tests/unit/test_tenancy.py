"""Tenant scoping tests.

Small module, disproportionate importance: this is the type-level half of the
isolation design (doc 5 §5.8 layer 2). The full guarantee needs the repository
negative suite against a real database (NFR-T5, arriving with M3) — these tests
cover the part that can be checked without one.

The property worth protecting is that widening scope is impossible after
construction and that the one legitimate bypass is explicit, named, and greppable.
"""

from __future__ import annotations

import dataclasses

import pytest

from scholiast.application.ports.tenancy import (
    MIGRATION_REASON,
    REBUILD_REASON,
    SWEEPER_REASON,
    Scope,
    SystemContext,
    TenantContext,
    TenantIsolationViolation,
)


class TestTenantContext:
    def test_owns_only_its_own_rows(self) -> None:
        ctx = TenantContext(user_id="user-1")
        assert ctx.owns("user-1")
        assert not ctx.owns("user-2")

    def test_does_not_own_unowned_rows(self) -> None:
        # A NULL owner must not be treated as "mine" — that would expose public
        # rows as tenant-owned and defeat the visibility model.
        assert not TenantContext(user_id="user-1").owns(None)

    @pytest.mark.parametrize("bad", ["", "   "])
    def test_empty_user_id_is_rejected(self, bad: str) -> None:
        # An empty tenant id would build a predicate matching nothing — or, worse,
        # everything, depending on how the adapter composes it.
        with pytest.raises(ValueError, match="non-empty user_id"):
            TenantContext(user_id=bad)

    def test_is_not_system(self) -> None:
        assert not TenantContext(user_id="user-1").is_system

    def test_is_immutable(self) -> None:
        # Nothing may widen its own scope after construction.
        ctx = TenantContext(user_id="user-1")
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.user_id = "user-2"  # type: ignore[misc]

    def test_is_hashable_for_caching(self) -> None:
        assert len({TenantContext("a"), TenantContext("a"), TenantContext("b")}) == 2


class TestSystemContext:
    def test_sees_everything(self) -> None:
        ctx = SystemContext(reason=SWEEPER_REASON)
        assert ctx.is_system
        assert ctx.owns("anyone")
        assert ctx.owns(None)

    def test_requires_a_stated_reason(self) -> None:
        # Forces the caller to say *why* it is bypassing tenant scoping, turning
        # an invisible privilege escalation into a readable one.
        with pytest.raises(ValueError, match="stated reason"):
            SystemContext(reason="  ")

    def test_sanctioned_reasons_are_enumerable(self) -> None:
        # Keeping them as constants means the full set of cross-tenant operations
        # can be listed — and reviewed.
        for reason in (SWEEPER_REASON, REBUILD_REASON, MIGRATION_REASON):
            assert SystemContext(reason=reason).is_system


class TestScopeProtocol:
    def test_both_contexts_satisfy_the_scope_protocol(self) -> None:
        # Repositories accept `Scope`, so both must be structurally compatible —
        # otherwise the maintenance paths need a parallel repository hierarchy.
        assert isinstance(TenantContext("user-1"), Scope)
        assert isinstance(SystemContext(SWEEPER_REASON), Scope)

    def test_a_bare_object_does_not_satisfy_it(self) -> None:
        assert not isinstance(object(), Scope)


class TestIsolationViolation:
    def test_message_names_the_entity_and_the_scope(self) -> None:
        # This is a defence-in-depth assertion, not control flow: if it fires,
        # layer 2 has a bug, and the message has to be enough to find it.
        exc = TenantIsolationViolation("Workspace", "ws-9", "user-1")
        message = str(exc)
        assert "Workspace" in message
        assert "ws-9" in message
        assert "user-1" in message
        assert "tenant filter" in message

    def test_is_a_runtime_error_not_a_domain_error(self) -> None:
        # A leak is a bug to fail loudly on, not an invalid request to reject
        # politely — so it must not be caught by domain-error handlers.
        assert issubclass(TenantIsolationViolation, RuntimeError)
