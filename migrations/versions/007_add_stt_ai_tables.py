"""add stt results and ai responses tables

Revision ID: 007_add_stt_ai_tables
Revises: 006_remove_live_table
Create Date: 2026-01-29 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007_add_stt_ai_tables"
down_revision: Union[str, None] = "006_remove_live_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "room_stt_results",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("room_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("member_id", sa.String(length=64), nullable=False),
        sa.Column("user_lang", sa.String(length=8), nullable=False),
        sa.Column("stt_text", sa.Text(), nullable=False),
        sa.Column("translated_text", sa.Text(), nullable=True),
        sa.Column("translated_lang", sa.String(length=8), nullable=True),
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["member_id"], ["room_members.id"], name="fk_room_stt_member"),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], name="fk_room_stt_room"),
        sa.ForeignKeyConstraint(["session_id"], ["room_live_sessions.id"], name="fk_room_stt_session"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("session_id", "seq", name="uq_room_stt_session_seq"),
    )
    op.create_index(
        "idx_room_stt_room_session_created",
        "room_stt_results",
        ["room_id", "session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_room_stt_member_seq",
        "room_stt_results",
        ["member_id", "seq"],
        unique=False,
    )

    op.create_table(
        "room_ai_responses",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("room_id", sa.String(length=64), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("lang", sa.String(length=8), nullable=False),
        sa.Column("stt_text", sa.Text(), nullable=False),
        sa.Column("stt_seq_end", sa.BigInteger(), nullable=False),
        sa.Column("answer_text", sa.Text(), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["room_id"], ["rooms.id"], name="fk_room_ai_room"),
        sa.ForeignKeyConstraint(["session_id"], ["room_live_sessions.id"], name="fk_room_ai_session"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "idx_room_ai_room_session_created",
        "room_ai_responses",
        ["room_id", "session_id", "created_at"],
        unique=False,
    )
    op.create_index(
        "idx_room_ai_session_stt_end",
        "room_ai_responses",
        ["session_id", "stt_seq_end"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("idx_room_ai_session_stt_end", table_name="room_ai_responses")
    op.drop_index("idx_room_ai_room_session_created", table_name="room_ai_responses")
    op.drop_table("room_ai_responses")

    op.drop_index("idx_room_stt_member_seq", table_name="room_stt_results")
    op.drop_index("idx_room_stt_room_session_created", table_name="room_stt_results")
    op.drop_table("room_stt_results")
