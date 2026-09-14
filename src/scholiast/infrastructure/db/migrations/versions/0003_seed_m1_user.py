"""seed the single M1 user

Revision ID: b31f0c7a52d4
Revises: 7a186c34092b
Create Date: 2026-09-13 10:40:00.000000

"""

# Template edited from Alembic's default so generated migrations satisfy the
# project's lint rules on the first pass. Fixing the template beats fixing each
# generated file: `--autogenerate` would otherwise reintroduce the same
# violations every time, and a lint gate that has to be hand-patched after every
# use is one people learn to skip.
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b31f0c7a52d4"
down_revision: str | Sequence[str] | None = "7a186c34092b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# D54. Doc 8's M1 criterion is "full stack runs from `docker compose up` + one
# command", and migrations already run inside that command — a separate seed
# step would make it two. The id is fixed rather than generated so the API, the
# UI and the tests can name the principal without a lookup, and so re-running
# the migration chain on a fresh database always produces the same row.
#
# Auth is M3 (FR-A1..A5), so this row has no password hash. `users.password_hash`
# is nullable for exactly this reason, and M3 must make it NOT NULL and treat a
# null hash as *unauthenticatable* rather than as a branch — otherwise this row
# becomes a login with no password.
M1_USER_ID = "00000000-0000-4000-8000-000000000001"
M1_USER_EMAIL = "demo@scholiast.local"


def upgrade() -> None:
    """Insert the seed user, if it is not already there."""
    # ON CONFLICT DO NOTHING, because a migration chain may be re-run against a
    # database that already has the row (a restored dump, a re-stamped head) and
    # a seed that fails on its second run is a seed that blocks deploys.
    op.execute(
        sa.text(
            # The cast is required, not decorative: asyncpg binds the parameter as
            # VARCHAR and Postgres will not implicitly cast it to `uuid`, so the
            # uncast form fails with a DatatypeMismatchError on every upgrade.
            "INSERT INTO users (id, email) VALUES (CAST(:id AS uuid), :email) "
            "ON CONFLICT (id) DO NOTHING"
        ).bindparams(id=M1_USER_ID, email=M1_USER_EMAIL)
    )


def downgrade() -> None:
    """Remove the seed user.

    Deletes by id, not by email: an operator who later repointed this address at
    a real account should not lose it to a downgrade. Workspaces cascade from
    `users`, so this takes the demo data with it — which is the intent, since a
    downgrade past this revision is a teardown, not a data migration.
    """
    op.execute(sa.text("DELETE FROM users WHERE id = CAST(:id AS uuid)").bindparams(id=M1_USER_ID))
