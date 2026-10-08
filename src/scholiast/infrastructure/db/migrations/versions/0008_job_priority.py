"""Ingestion jobs carry a priority, and the claim takes interactive work first (D139).

Under D95 a Celery wake-up claims whatever is due, so the only place a priority
can mean anything is the claim's ORDER BY; this column is what it orders on.
Two values, 10 for interactive work (an upload a user is watching) and 0 for
backfill (a discovered paper), with room between them for a third if one is
ever measured to be needed.

**The server default is interactive.** Every job in flight at migration time is
an upload, and a row inserted without the column errs toward not being
deprioritised. The application states it on every enqueue anyway (the port has
no default), so the default is for the migration and for hand-written SQL.

**The CHECK is written by hand**, because Alembic does not compare CHECK
constraints (CLAUDE.md), and test_schema.py attempts a violation of it.

Revision ID: b52e8d3f7a19
Revises: 7c1d4e9a2f60
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b52e8d3f7a19"
down_revision: str | Sequence[str] | None = "7c1d4e9a2f60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("priority", sa.SmallInteger(), nullable=False, server_default="10"),
    )
    op.create_check_constraint(
        "ck_ingestion_jobs_priority", "ingestion_jobs", "priority IN (0, 10)"
    )


def downgrade() -> None:
    op.drop_constraint("ck_ingestion_jobs_priority", "ingestion_jobs", type_="check")
    op.drop_column("ingestion_jobs", "priority")
