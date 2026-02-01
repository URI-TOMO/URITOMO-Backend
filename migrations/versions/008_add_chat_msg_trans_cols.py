"""add translation columns to chat_messages

Revision ID: 008_add_chat_msg_trans_cols
Revises: 007_add_stt_ai_tables
Create Date: 2026-01-30 12:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "008_add_chat_msg_trans_cols"
down_revision: Union[str, None] = "007_add_stt_ai_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_cols = {col["name"] for col in inspector.get_columns("chat_messages")}

    if "translated_text" not in existing_cols:
        op.add_column("chat_messages", sa.Column("translated_text", sa.Text(), nullable=True))
    if "translated_lang" not in existing_cols:
        op.add_column("chat_messages", sa.Column("translated_lang", sa.String(length=8), nullable=True))


def downgrade() -> None:
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_cols = {col["name"] for col in inspector.get_columns("chat_messages")}

    if "translated_lang" in existing_cols:
        op.drop_column("chat_messages", "translated_lang")
    if "translated_text" in existing_cols:
        op.drop_column("chat_messages", "translated_text")
