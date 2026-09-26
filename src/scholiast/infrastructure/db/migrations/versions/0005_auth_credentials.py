"""refresh tokens, and a password for every user

Revision ID: d77b4e2c9a15
Revises: c42a9e1b7d38
Create Date: 2026-09-25 10:20:00.000000

"""

from __future__ import annotations

import os
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from scholiast.infrastructure.security.passwords import Argon2idHasher

# revision identifiers, used by Alembic.
revision: str = "d77b4e2c9a15"
down_revision: str | Sequence[str] | None = "c42a9e1b7d38"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# M1 seeded one user with no password (migration 0003) and recorded the debt that
# M3 must make `users.password_hash` NOT NULL. D85 chose how: the row keeps its
# data and gets a real, environment-supplied development password, because
# `workspaces.owner_id` cascades on delete — dropping the row would take every
# workspace, paper, chunk and conversation in any existing database with it.
M1_USER_ID = "00000000-0000-4000-8000-000000000001"

#: Read at upgrade time. The default is a development credential and is documented
#: as one: it exists so a local database keeps working, and M6's deployment work
#: is where a real environment must set this or delete the row.
SEED_PASSWORD_ENV = "SEED_USER_PASSWORD"  # noqa: S105 - a variable NAME, not a secret
DEFAULT_SEED_PASSWORD = "scholiast-dev-password"  # noqa: S105 - documented dev default


def upgrade() -> None:
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("family_id", sa.dialects.postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column(
            "issued_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("fingerprint", name="uq_refresh_tokens_fingerprint"),
    )
    op.create_index("ix_refresh_tokens_user", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_family", "refresh_tokens", ["family_id"])
    op.create_index("ix_refresh_tokens_expires_at", "refresh_tokens", ["expires_at"])

    # Hash here rather than embedding a constant: a hash committed to the
    # repository would be the same credential in every database that ever ran
    # this migration, which is precisely the property a seeded password must not
    # have. The salt is fresh per run.
    password = os.environ.get(SEED_PASSWORD_ENV, DEFAULT_SEED_PASSWORD)
    op.execute(
        sa.text(
            "UPDATE users SET password_hash = :hashed "
            "WHERE id = CAST(:id AS uuid) AND password_hash IS NULL"
        ).bindparams(hashed=Argon2idHasher().hash(password), id=M1_USER_ID)
    )

    # Deliberately NOT a blanket backfill. Any other password-less row is an
    # account this migration cannot identify, and giving it a known credential
    # would be worse than failing: the ALTER below will refuse, loudly, and an
    # operator can decide what that account is.
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=False)


def downgrade() -> None:
    op.alter_column("users", "password_hash", existing_type=sa.Text(), nullable=True)
    # The seeded row keeps its hash on downgrade. Clearing it would destroy a
    # credential the operator may have set deliberately, and a nullable column
    # tolerates it being present.
    op.drop_index("ix_refresh_tokens_expires_at", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_family", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
