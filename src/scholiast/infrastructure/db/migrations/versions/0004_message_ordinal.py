"""order messages within a conversation

Revision ID: c42a9e1b7d38
Revises: b31f0c7a52d4
Create Date: 2026-09-13 19:40:00.000000

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
revision: str = "c42a9e1b7d38"
down_revision: str | Sequence[str] | None = "b31f0c7a52d4"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# §5.6.3's `messages` table has no ordering column, and `created_at` cannot serve
# as one: its server default is `now()`, which in Postgres is **transaction
# start time**, so the question and the answer of a single QA turn — written in
# one transaction — get byte-identical timestamps. Ordering then falls through to
# a random UUID, and a conversation comes back with the answer above the
# question about half the time.
#
# Found by the phase 6 acceptance test, which asserted the roles in order. The
# fix is the same one `chunks.ordinal` already uses: a sequence is stored as a
# sequence rather than inferred from a clock. The unique constraint is what makes
# a concurrent double-append fail loudly instead of silently producing two
# messages that claim the same position.


def upgrade() -> None:
    """Add the ordering column and backfill existing rows."""
    op.add_column("messages", sa.Column("ordinal", sa.Integer(), nullable=True))

    # Backfill in the only order the existing rows can offer. Where timestamps
    # tie, `id` at least makes the result deterministic — this cannot recover the
    # true order, which is precisely why the column now exists.
    op.execute(
        sa.text(
            "UPDATE messages SET ordinal = numbered.position FROM ("
            "  SELECT id, row_number() OVER ("
            "    PARTITION BY conversation_id ORDER BY created_at, id"
            "  ) - 1 AS position FROM messages"
            ") AS numbered WHERE messages.id = numbered.id"
        )
    )
    op.alter_column("messages", "ordinal", nullable=False)

    op.create_index(
        "uq_messages_conversation_ordinal",
        "messages",
        ["conversation_id", "ordinal"],
        unique=True,
    )
    op.create_check_constraint("ck_messages_ordinal_non_negative", "messages", "ordinal >= 0")


def downgrade() -> None:
    """Drop the ordering column.

    Irreversible in substance, like migration 0002's: the position information
    is discarded and `created_at` cannot reconstruct it.
    """
    op.drop_constraint("ck_messages_ordinal_non_negative", "messages", type_="check")
    op.drop_index("uq_messages_conversation_ordinal", table_name="messages")
    op.drop_column("messages", "ordinal")
