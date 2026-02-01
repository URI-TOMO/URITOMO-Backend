"""remove live table and ai_events source_live_id

Revision ID: 006_remove_live_table
Revises: b502c0ce3b3e
Create Date: 2026-01-29 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "006_remove_live_table"
down_revision: Union[str, None] = "b502c0ce3b3e"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("ai_events") as batch:
        batch.drop_constraint("fk_ai_source_live", type_="foreignkey")
        batch.drop_index("idx_ai_source_live")
        batch.drop_column("source_live_id")

    op.drop_table("live")


def downgrade() -> None:
    op.create_table(
        "live",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("room_id", sa.String(length=64), nullable=False),
        sa.Column("member_id", sa.String(length=64), nullable=False),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("lang", sa.String(length=8), nullable=True),
        sa.Column("start_ms", sa.Integer(), nullable=True),
        sa.Column("end_ms", sa.Integer(), nullable=True),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["member_id"], ["room_members.id"], name="fk_live_member"),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], name="fk_live_room"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("room_id", "seq", name="uq_live_room_seq"),
    )
    op.create_index("idx_live_member_seq", "live", ["member_id", "seq"], unique=False)
    op.create_index("idx_live_room_created", "live", ["room_id", "created_at"], unique=False)

    with op.batch_alter_table("ai_events") as batch:
        batch.add_column(sa.Column("source_live_id", sa.String(length=64), nullable=True))
        batch.create_index("idx_ai_source_live", ["source_live_id"], unique=False)
        batch.create_foreign_key("fk_ai_source_live", "live", ["source_live_id"], ["id"])
