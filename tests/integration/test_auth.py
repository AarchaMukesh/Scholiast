"""Authentication against a real Postgres (FR-A1, FR-A5, D84, D85).

The unit suite covers policy with fakes. What only a real database can answer is
whether the *storage* half holds up: does the unique index actually refuse a
second registration, does `ON DELETE CASCADE` actually take the refresh tokens
with the user, does migration 0005's backfill actually leave the seeded account
able to log in, and is the refresh secret really absent from every column.

**Postgres only, no Qdrant.** `create_app`'s `lifespan` seam is used to build
just the engine, the hasher and the token service — nothing authentication
touches needs the vector index, and the main CI job has Postgres via
testcontainers but no Qdrant. A `needs_qdrant` mark here would mean these
checks ran nowhere except a developer's machine.

The hasher is the production one at production parameters. It costs ~60 ms per
call and there are a few dozen calls in this module; paying that is the point,
since the cost IS the security property.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import timedelta
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncConnection, create_async_engine
from testcontainers.community.postgres import PostgresContainer

from scholiast.api.app import create_app
from scholiast.application.ports.tenancy import (
    AUTHENTICATION_REASON,
    SWEEPER_REASON,
    SystemContext,
    TenantContext,
)
from scholiast.config.settings import Settings
from scholiast.infrastructure.db.accounts import SqlAccountRepository, SqlSessionRepository
from scholiast.infrastructure.db.engine import build_engine, build_session_factory
from scholiast.infrastructure.security.passwords import Argon2idHasher
from scholiast.infrastructure.security.tokens import JwtTokenService

pytestmark = [pytest.mark.integration]

ROOT = Path(__file__).resolve().parents[2]

PASSWORD = "a-perfectly-adequate-password"
SEED_PASSWORD = "seed-password-for-this-test-run"
SEED_EMAIL = "demo@scholiast.local"
M1_USER_ID = "00000000-0000-4000-8000-000000000001"


@pytest.fixture(scope="module")
def postgres() -> Iterator[str]:
    """A throwaway Postgres with the migrations applied.

    `SEED_USER_PASSWORD` is set explicitly rather than left to migration 0005's
    documented default, so the assertion below tests the mechanism (the env var
    is read and hashed) instead of a constant that could drift.
    """
    with PostgresContainer("postgres:17-alpine", driver="asyncpg") as container:
        url = container.get_connection_url()
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-m", "alembic", "-x", f"db_url={url}", "upgrade", "head"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "SEED_USER_PASSWORD": SEED_PASSWORD},
        )
        if result.returncode != 0:
            pytest.fail(f"alembic upgrade failed:\n{result.stdout}\n{result.stderr}")
        yield url


@contextlib.asynccontextmanager
async def auth_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Everything `api_lifespan` builds that authentication uses, and nothing else."""
    settings: Settings = app.state.settings
    engine = build_engine(settings)
    app.state.engine = engine
    app.state.sessions = build_session_factory(engine)
    app.state.hasher = Argon2idHasher()
    app.state.tokens = JwtTokenService(
        secret=settings.jwt_secret,
        issuer=settings.jwt_issuer,
        access_ttl=timedelta(minutes=settings.access_token_minutes),
        refresh_ttl=timedelta(days=settings.refresh_token_days),
    )
    try:
        yield
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def client(postgres: str, tmp_path: Path) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(
        Settings(
            database_url=postgres,  # type: ignore[arg-type]
            object_store_path=tmp_path / "objects",
            worker_in_process=False,
        ),
        lifespan=auth_lifespan,
    )
    async with LifespanManager(app) as manager:
        transport = httpx.ASGITransport(app=manager.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


@pytest_asyncio.fixture
async def conn(postgres: str) -> AsyncIterator[AsyncConnection]:
    engine = create_async_engine(postgres)
    async with engine.connect() as connection:
        yield connection
        await connection.rollback()
    await engine.dispose()


def _email() -> str:
    return f"user-{uuid.uuid4().hex[:10]}@example.com"


async def _register(client: httpx.AsyncClient, email: str | None = None) -> dict[str, str]:
    response = await client.post(
        "/v1/auth/register", json={"email": email or _email(), "password": PASSWORD}
    )
    assert response.status_code == 201, response.text
    return dict(response.json())


class TestTheFlowAgainstRealTables:
    async def test_register_call_refresh_logout(self, client: httpx.AsyncClient) -> None:
        """The phase's acceptance path, on real rows."""
        first = await _register(client)
        auth = {"Authorization": f"Bearer {first['access_token']}"}

        # An authenticated call to an ordinary router: §5.8's layer 1 producing
        # the principal that layer 2 scopes by.
        created = await client.post("/v1/workspaces", json={"name": "Reading list"}, headers=auth)
        assert created.status_code == 201

        rotated = await client.post(
            "/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )
        assert rotated.status_code == 200
        second = rotated.json()

        # The new access token works, and the old refresh token does not.
        listed = await client.get(
            "/v1/workspaces", headers={"Authorization": f"Bearer {second['access_token']}"}
        )
        assert [w["name"] for w in listed.json()] == ["Reading list"]
        replay = await client.post(
            "/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
        )
        assert replay.status_code == 401

        logged_out = await client.post(
            "/v1/auth/logout", json={"refresh_token": second["refresh_token"]}
        )
        assert logged_out.status_code == 204

    async def test_the_workspace_belongs_to_the_authenticated_user(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        # The join between the two layers, which is the whole reason phase 1
        # comes first: a token's subject becomes a row's owner.
        issued = await _register(client)
        created = await client.post(
            "/v1/workspaces",
            json={"name": "Owned"},
            headers={"Authorization": f"Bearer {issued['access_token']}"},
        )
        owner = await conn.scalar(
            text("SELECT owner_id FROM workspaces WHERE id = CAST(:id AS uuid)"),
            {"id": created.json()["id"]},
        )
        assert str(owner) == issued["user_id"]

    async def test_an_unauthenticated_call_reaches_no_router(
        self, client: httpx.AsyncClient
    ) -> None:
        # M1's routers had no principal and could not refuse anyone. They can now,
        # and none of them had to change to gain it.
        for method, path in (("GET", "/v1/workspaces"), ("POST", "/v1/workspaces")):
            response = await client.request(method, path, json={"name": "x"})
            assert response.status_code == 401
            assert response.headers["WWW-Authenticate"] == "Bearer"

    async def test_the_refresh_secret_is_never_written_down(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        # D84's claim in one assertion: a stolen database yields no usable
        # refresh token, because the secret is not in it.
        issued = await _register(client)
        secret = issued["refresh_token"]
        rows = (await conn.execute(text("SELECT * FROM refresh_tokens"))).mappings().all()
        assert rows
        assert all(secret not in str(dict(row)) for row in rows)
        assert any(len(row["fingerprint"]) == 64 for row in rows)

    async def test_the_password_is_stored_as_an_argon2id_hash(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        email = _email()
        await _register(client, email=email)
        stored = await conn.scalar(
            text("SELECT password_hash FROM users WHERE email = :email"), {"email": email}
        )
        assert isinstance(stored, str)
        assert stored.startswith("$argon2id$v=19$m=65536,t=3,p=4$")
        assert PASSWORD not in stored

    async def test_a_duplicate_registration_is_refused_by_the_index(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        # The repository maps the constraint violation rather than pre-checking
        # with a SELECT, which two concurrent registrations would both pass.
        email = _email()
        await _register(client, email=email)
        again = await client.post(
            "/v1/auth/register", json={"email": email.upper(), "password": PASSWORD}
        )
        assert again.status_code == 409
        count = await conn.scalar(
            text("SELECT count(*) FROM users WHERE email = :email"), {"email": email}
        )
        assert count == 1

    async def test_reuse_revokes_every_row_in_the_family(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        first = await _register(client)
        second = (
            await client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
        ).json()
        await client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})

        live = await conn.scalar(
            text(
                "SELECT count(*) FROM refresh_tokens "
                "WHERE user_id = CAST(:id AS uuid) AND revoked_at IS NULL"
            ),
            {"id": first["user_id"]},
        )
        assert live == 0
        # And the live token really is dead at the HTTP boundary too.
        replay = await client.post(
            "/v1/auth/refresh", json={"refresh_token": second["refresh_token"]}
        )
        assert replay.status_code == 401

    async def test_two_refreshes_racing_on_one_token_produce_one_winner(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        """D87: single use has to hold when the two uses overlap in time.

        This is the shape a thief racing the legitimate client produces, and it
        is the only test here that needs real transactions: both requests read
        the row as usable, and what separates them is whether the revoking
        UPDATE changed anything.
        """
        first = await _register(client)
        both = await asyncio.gather(
            client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}),
            client.post("/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}),
        )

        assert sorted(response.status_code for response in both) == [200, 401]

        # The loser is treated as reuse, so the family dies — including the
        # successor the winner just issued. A forced re-login is the intended
        # price of not being able to tell thief from client.
        live = await conn.scalar(
            text(
                "SELECT count(*) FROM refresh_tokens "
                "WHERE user_id = CAST(:id AS uuid) AND revoked_at IS NULL"
            ),
            {"id": first["user_id"]},
        )
        assert live == 0

    async def test_logout_writes_the_revocation_down(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        issued = await _register(client)
        await client.post("/v1/auth/logout", json={"refresh_token": issued["refresh_token"]})
        revoked = await conn.scalar(
            text(
                "SELECT count(*) FROM refresh_tokens "
                "WHERE user_id = CAST(:id AS uuid) AND revoked_at IS NOT NULL"
            ),
            {"id": issued["user_id"]},
        )
        assert revoked == 1


class TestTheSeededAccount:
    """D85: the M1 row keeps its data AND becomes an ordinary account."""

    async def test_the_seeded_user_can_log_in(self, client: httpx.AsyncClient) -> None:
        # Migration 0005 read `SEED_USER_PASSWORD` and hashed it. If it had
        # written a placeholder instead, the row would be an orphan nobody can
        # reach — the outcome D85 rejected.
        response = await client.post(
            "/v1/auth/login", json={"email": SEED_EMAIL, "password": SEED_PASSWORD}
        )
        assert response.status_code == 200
        assert response.json()["user_id"] == M1_USER_ID

    async def test_the_documented_default_is_not_what_was_set(
        self, client: httpx.AsyncClient
    ) -> None:
        # The other half: the password came from the environment, so the
        # committed default is not a credential to this database.
        response = await client.post(
            "/v1/auth/login", json={"email": SEED_EMAIL, "password": "scholiast-dev-password"}
        )
        assert response.status_code == 401


class TestTheConstraints:
    """Attempted violations — a constraint that was never created looks identical
    to one that works until something tries to break it."""

    async def test_a_user_without_a_password_hash_is_refused(self, conn: AsyncConnection) -> None:
        # The NOT NULL migration 0005 exists to add, and M1's recorded debt.
        with pytest.raises(IntegrityError):
            await conn.execute(
                text("INSERT INTO users (id, email) VALUES (:id, :email)"),
                {"id": uuid.uuid4(), "email": _email()},
            )

    async def test_two_sessions_cannot_share_a_fingerprint(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        # Without this, a collision would silently let one row's revocation
        # apply to another's token.
        issued = await _register(client)
        fingerprint = await conn.scalar(
            text("SELECT fingerprint FROM refresh_tokens WHERE user_id = CAST(:id AS uuid)"),
            {"id": issued["user_id"]},
        )
        with pytest.raises(IntegrityError):
            await conn.execute(
                text(
                    "INSERT INTO refresh_tokens (id, user_id, family_id, fingerprint, expires_at) "
                    "VALUES (:id, CAST(:user AS uuid), :family, :fingerprint, now())"
                ),
                {
                    "id": uuid.uuid4(),
                    "user": issued["user_id"],
                    "family": uuid.uuid4(),
                    "fingerprint": fingerprint,
                },
            )

    async def test_deleting_a_user_takes_their_sessions(
        self, client: httpx.AsyncClient, conn: AsyncConnection
    ) -> None:
        # Otherwise M3 phase 5's account deletion would leave live credentials
        # pointing at a user that no longer exists.
        issued = await _register(client)
        await conn.execute(
            text("DELETE FROM users WHERE id = CAST(:id AS uuid)"), {"id": issued["user_id"]}
        )
        remaining = await conn.scalar(
            text("SELECT count(*) FROM refresh_tokens WHERE user_id = CAST(:id AS uuid)"),
            {"id": issued["user_id"]},
        )
        assert remaining == 0


class TestTheAuthenticationScope:
    async def test_the_account_repositories_refuse_an_ordinary_tenant(self, postgres: str) -> None:
        # These two are the one sanctioned bypass of §5.8 layer 2, and the
        # bypass is narrow by construction: they take `SystemContext` with a
        # named reason, and nothing else.
        engine = create_async_engine(postgres)
        factory = build_session_factory(engine)
        try:
            async with factory() as session:
                for repository in (SqlAccountRepository, SqlSessionRepository):
                    # A tenant principal is a category error here, and so is a
                    # system context borrowed from another sanctioned use: the
                    # reason string is checked too, which is what keeps
                    # `AUTHENTICATION_REASON` an exhaustive grep.
                    with pytest.raises(ValueError, match="authentication"):
                        repository(session, TenantContext(user_id=str(uuid.uuid4())))
                    with pytest.raises(ValueError, match="authentication"):
                        repository(session, SystemContext(SWEEPER_REASON))
                    assert repository(session, SystemContext(AUTHENTICATION_REASON))
        finally:
            await engine.dispose()
