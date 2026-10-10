"""A paper may exist before its PDF, and is identified by DOI or arXiv id (D132-D134).

Phase 6 adds discovered papers by identifier: the row exists when the user clicks
add, and its bytes arrive when the FETCHING stage downloads them. So:

**`content_hash` becomes nullable, under a CHECK that says exactly when.** NULL is
allowed only while nothing has been parsed and the row has a DOI or an arXiv id to
be found by. Alembic does not compare CHECK constraints (CLAUDE.md), so each one
here is written by hand and covered by an attempted violation in test_schema.py.

**DOI and arXiv id are columns, unique per owner among private rows.** Not paths
into `external_ids`, because the parse write-back rewrites that bag on every run,
and identity under a unique index must sit where an update cannot move it. Per
owner because discovered papers are private to whoever added them (D131). Two
CHECKs pin one spelling per identity, since a unique index over two spellings of
one id enforces nothing.

**`origin`** records whether the source's metadata or GROBID's wins (D135). Every
existing row is an upload, which the server default says.

**`ingestion_jobs.source_links`** carries the links a FETCHING job tries (D133).
Nullable: uploads have their bytes already.

**The downgrade refuses while any row still awaits its PDF**, on 0005's rule of
refusing rather than guessing (its upgrade refuses a password-less account it
cannot identify): SET NOT NULL would fail anyway, and deleting those rows would
cascade users' queued papers away. An operator can decide; a migration should not.

Revision ID: 7c1d4e9a2f60
Revises: e83c5f1a2b64
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers, used by Alembic.
revision: str = "7c1d4e9a2f60"
down_revision: str | Sequence[str] | None = "e83c5f1a2b64"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("papers", sa.Column("doi", sa.Text(), nullable=True))
    op.add_column("papers", sa.Column("arxiv_id", sa.Text(), nullable=True))
    op.add_column(
        "papers",
        sa.Column("origin", sa.Text(), nullable=False, server_default="upload"),
    )
    op.alter_column("papers", "content_hash", existing_type=sa.String(64), nullable=True)

    op.create_check_constraint("ck_papers_origin", "papers", "origin IN ('upload', 'discovery')")
    op.create_check_constraint(
        "ck_papers_hash_or_awaiting",
        "papers",
        "content_hash IS NOT NULL OR "
        "(parse_status = 'pending' AND (doi IS NOT NULL OR arxiv_id IS NOT NULL))",
    )
    op.create_check_constraint(
        "ck_papers_doi_canonical",
        "papers",
        "doi IS NULL OR (doi = lower(doi) AND doi LIKE '10.%/%' "
        "AND doi NOT LIKE '10.48550/arxiv.%')",
    )
    op.create_check_constraint(
        "ck_papers_arxiv_id_canonical",
        "papers",
        "arxiv_id IS NULL OR (arxiv_id = lower(arxiv_id) AND arxiv_id !~ 'v[0-9]+$')",
    )
    op.create_index(
        "uq_papers_private_owner_doi",
        "papers",
        ["owner_id", "doi"],
        unique=True,
        postgresql_where=sa.text("visibility = 'private' AND doi IS NOT NULL"),
    )
    op.create_index(
        "uq_papers_private_owner_arxiv",
        "papers",
        ["owner_id", "arxiv_id"],
        unique=True,
        postgresql_where=sa.text("visibility = 'private' AND arxiv_id IS NOT NULL"),
    )

    op.add_column("ingestion_jobs", sa.Column("source_links", JSONB(), nullable=True))


def downgrade() -> None:
    awaiting = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM papers WHERE content_hash IS NULL"))
        .scalar_one()
    )
    if awaiting:
        raise RuntimeError(
            f"{awaiting} paper(s) still await their PDF (content_hash IS NULL); "
            "remove them or let them finish ingesting before downgrading, because "
            "content_hash cannot become NOT NULL while they exist"
        )
    op.drop_column("ingestion_jobs", "source_links")
    op.drop_index("uq_papers_private_owner_arxiv", table_name="papers")
    op.drop_index("uq_papers_private_owner_doi", table_name="papers")
    op.drop_constraint("ck_papers_arxiv_id_canonical", "papers", type_="check")
    op.drop_constraint("ck_papers_doi_canonical", "papers", type_="check")
    op.drop_constraint("ck_papers_hash_or_awaiting", "papers", type_="check")
    op.drop_constraint("ck_papers_origin", "papers", type_="check")
    op.alter_column("papers", "content_hash", existing_type=sa.String(64), nullable=False)
    op.drop_column("papers", "origin")
    op.drop_column("papers", "arxiv_id")
    op.drop_column("papers", "doi")
