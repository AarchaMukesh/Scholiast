"""Carry the correlation id on the job row (FR-O1, D107).

FR-O1 asks for ids "propagated across API → worker", and the obvious vehicle — a
Celery message header — is wrong here because of D95. The message is a *wake-up*:
the task calls `run_once()`, which claims whatever is due, so the job a message
names is frequently not the job that run processes. An id carried on the message
would therefore label a worker run with some other paper's request id, which is
worse than no correlation because it looks authoritative.

On the row, the id follows the WORK: it survives a redelivery, a lease expiry, a
sweep, and a message that never arrived at all.

**Nullable, and not backfilled.** Every job M1 and M2 wrote predates this column
and has no request behind it that a log line could be found for; inventing an id
for those rows would fabricate a link to a record that does not exist. NULL means
"no correlation was captured", which is the truth.

**No index.** The column is written once at enqueue and read once at claim, both
by primary key — it is a payload the row carries, not something queried by.
Indexing it would cost every insert on the hot queue table to serve a query
nothing makes. An operator grepping logs for an id does not touch this table.

Revision ID: e83c5f1a2b64
Revises: d77b4e2c9a15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e83c5f1a2b64"
down_revision: str | Sequence[str] | None = "d77b4e2c9a15"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_jobs",
        sa.Column("correlation_id", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("ingestion_jobs", "correlation_id")
