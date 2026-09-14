"""parse status vocabulary

Revision ID: 7a186c34092b
Revises: e4ee83af756c
Create Date: 2026-08-28 20:23:53.078383

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
revision: str = "7a186c34092b"
down_revision: str | Sequence[str] | None = "e4ee83af756c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # NOTE: `--autogenerate` detected the column but NOT the CHECK constraints —
    # Alembic does not compare CHECKs, and neither does `alembic check`. They are
    # written by hand below, and each is covered by an attempted violation in
    # tests/integration/test_schema.py::TestParseStatusVocabulary, because a
    # constraint that silently failed to be created looks identical to one that
    # is working until the day something bad gets in.
    op.add_column("papers", sa.Column("parse_failure_class", sa.Text(), nullable=True))

    # Existing rows predate the vocabulary: the initial migration had no server
    # default and the repository wrote ad-hoc strings. Normalise before the
    # constraints go on, or the ALTERs fail on legacy values.
    op.execute(
        "UPDATE papers SET parse_status = 'parsed' "
        "WHERE parse_status NOT IN ('pending', 'parsing', 'parsed', 'failed')"
    )
    op.alter_column("papers", "parse_status", server_default="pending")

    op.create_check_constraint(
        "ck_papers_parse_status",
        "papers",
        "parse_status IN ('pending', 'parsing', 'parsed', 'failed')",
    )
    op.create_check_constraint(
        "ck_papers_parse_failure_class",
        "papers",
        "parse_failure_class IS NULL OR parse_failure_class IN ('source_unavailable', "
        "'transient', 'unparseable', 'scanned_image', 'too_large', 'resource_exhausted')",
    )
    op.create_check_constraint(
        "ck_papers_parse_failure_iff_failed",
        "papers",
        "(parse_status = 'failed') = (parse_failure_class IS NOT NULL)",
    )
    # Same gap, same fix: failure_class on membership was unconstrained too.
    op.create_check_constraint(
        "ck_workspace_papers_failure_class",
        "workspace_papers",
        "failure_class IS NULL OR failure_class IN ('source_unavailable', 'transient', "
        "'unparseable', 'scanned_image', 'too_large', 'resource_exhausted')",
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("ck_workspace_papers_failure_class", "workspace_papers")
    op.drop_constraint("ck_papers_parse_failure_iff_failed", "papers")
    op.drop_constraint("ck_papers_parse_failure_class", "papers")
    op.drop_constraint("ck_papers_parse_status", "papers")
    op.alter_column("papers", "parse_status", server_default=None)
    op.drop_column("papers", "parse_failure_class")
