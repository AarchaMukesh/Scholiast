"""Async engine and session factory.

Async all the way down: asyncpg only, no psycopg2. Mixing a sync driver in for
Alembic's convenience would mean two connection stacks with two failure modes
and two sets of pool settings, which is a poor trade for saving a template.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from scholiast.config.settings import Settings


def build_engine(settings: Settings, *, echo: bool = False) -> AsyncEngine:
    return create_async_engine(
        str(settings.database_url),
        echo=echo,
        # Verify a connection before handing it out. Without this, a connection
        # dropped by a restarted container surfaces as a failed request rather
        # than a transparent reconnect.
        pool_pre_ping=True,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False so a returned entity stays readable after commit;
    # otherwise every attribute access after a commit triggers a lazy reload,
    # which in async code raises rather than quietly re-querying.
    return async_sessionmaker(engine, expire_on_commit=False)


class SqlUnitOfWork:
    """The `UnitOfWork` adapter: a named commit point on one session.

    Nothing more, deliberately. The use case needs to say "this stage is
    durable" (§5.4.2); it does not need a transaction manager, and giving it one
    would let business logic start deciding about savepoints and isolation
    levels — decisions that belong to the layer that owns the connection.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def commit(self) -> None:
        await self._session.commit()


@asynccontextmanager
async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """One unit of work: commit on success, roll back on any exception."""
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()
