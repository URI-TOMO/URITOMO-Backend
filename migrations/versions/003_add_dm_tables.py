"""add dm tables

Revision ID: 003
Revises: 002
Create Date: 2026-01-20 19:55:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '003'
down_revision: Union[str, None] = '002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- dm_threads ---
    op.create_table(
        'dm_threads',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_friend_id', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False, server_default='active'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_friend_id'], ['user_friends.id'], name='fk_dm_threads_user_friend', onupdate='CASCADE', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_friend_id', name='uq_dm_threads_user_friend')
    )
    op.create_index('idx_dm_threads_status_created', 'dm_threads', ['status', 'created_at'], unique=False)

    # --- dm_participants ---
    op.create_table(
        'dm_participants',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('thread_id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('joined_at', sa.DateTime(), nullable=False),
        sa.Column('left_at', sa.DateTime(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['thread_id'], ['dm_threads.id'], name='fk_dm_participants_thread', onupdate='CASCADE', ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_dm_participants_user', onupdate='CASCADE', ondelete='RESTRICT'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id', 'user_id', name='uq_dm_participants_thread_user')
    )
    op.create_index('idx_dm_participants_thread', 'dm_participants', ['thread_id'], unique=False)
    op.create_index('idx_dm_participants_user', 'dm_participants', ['user_id'], unique=False)

    # --- dm_messages ---
    op.create_table(
        'dm_messages',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('thread_id', sa.String(length=64), nullable=False),
        sa.Column('seq', sa.BigInteger(), nullable=False),
        sa.Column('sender_type', sa.String(length=16), nullable=False),
        sa.Column('sender_user_id', sa.String(length=64), nullable=True),
        sa.Column('message_type', sa.String(length=32), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('lang', sa.String(length=8), nullable=True),
        sa.Column('start_ms', sa.Integer(), nullable=True),
        sa.Column('end_ms', sa.Integer(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint("(sender_type <> 'human') OR (sender_user_id IS NOT NULL)", name='chk_dm_messages_sender_user_when_human'),
        sa.ForeignKeyConstraint(['sender_user_id'], ['users.id'], name='fk_dm_messages_sender_user', onupdate='CASCADE', ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['thread_id'], ['dm_threads.id'], name='fk_dm_messages_thread', onupdate='CASCADE', ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('thread_id', 'seq', name='uq_dm_messages_thread_seq')
    )
    op.create_index('idx_dm_messages_sender_user', 'dm_messages', ['sender_user_id'], unique=False)
    op.create_index('idx_dm_messages_thread_created', 'dm_messages', ['thread_id', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_dm_messages_thread_created', table_name='dm_messages')
    op.drop_index('idx_dm_messages_sender_user', table_name='dm_messages')
    op.drop_table('dm_messages')
    op.drop_index('idx_dm_participants_user', table_name='dm_participants')
    op.drop_index('idx_dm_participants_thread', table_name='dm_participants')
    op.drop_table('dm_participants')
    op.drop_index('idx_dm_threads_status_created', table_name='dm_threads')
    op.drop_table('dm_threads')
