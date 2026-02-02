"""Add_MeetingSummaryRecord

Revision ID: 011
Revises: 010
Create Date: 2026-02-02 23:35:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '011'
down_revision: Union[str, None] = '010'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('meeting_summary_records',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('room_id', sa.String(length=64), nullable=False),
    sa.Column('main_point', sa.Text(), nullable=False),
    sa.Column('task', sa.Text(), nullable=False),
    sa.Column('decided', sa.Text(), nullable=False),
    sa.Column('meeting_date', sa.DateTime(), nullable=False),
    sa.Column('past_time', sa.String(length=32), nullable=False),
    sa.Column('member_count', sa.Integer(), nullable=False),
    sa.Column('message_count', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('meeting_summary_records')
