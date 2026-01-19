"""add user_friends table

Revision ID: 002
Revises: 001
Create Date: 2026-01-20 01:28:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- user_friends ---
    op.create_table(
        'user_friends',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('requester_id', sa.String(length=64), nullable=False),
        sa.Column('addressee_id', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), server_default='pending', nullable=False),
        sa.Column('requested_at', sa.DateTime(), nullable=False),
        sa.Column('responded_at', sa.DateTime(), nullable=True),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('friend_name', sa.String(length=128), nullable=True),
        sa.CheckConstraint('requester_id <> addressee_id', name='chk_user_friends_not_self'),
        sa.ForeignKeyConstraint(['addressee_id'], ['users.id'], name='fk_friends_addressee'),
        sa.ForeignKeyConstraint(['requester_id'], ['users.id'], name='fk_friends_requester'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('requester_id', 'addressee_id', name='uq_user_friends_pair_direction')
    )
    op.create_index('idx_user_friends_addressee', 'user_friends', ['addressee_id', 'status'], unique=False)
    op.create_index('idx_user_friends_requester', 'user_friends', ['requester_id', 'status'], unique=False)
    op.create_index('idx_user_friends_status', 'user_friends', ['status', 'requested_at'], unique=False)


def downgrade() -> None:
    op.drop_table('user_friends')
