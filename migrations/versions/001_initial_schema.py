"""initial schema

Revision ID: 001
Revises: 
Create Date: 2026-01-12 01:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=True),
        sa.Column('display_name', sa.String(length=128), nullable=False),
        sa.Column('locale', sa.String(length=8), nullable=True),
        sa.Column('status', sa.String(length=16), server_default='active', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('email')
    )

    # --- rooms ---
    op.create_table(
        'rooms',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=True),
        sa.Column('created_by', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), server_default='active', nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('ended_at', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], name='fk_rooms_creator'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_rooms_status_created', 'rooms', ['status', 'created_at'], unique=False)

    # --- room_members ---
    op.create_table(
        'room_members',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('room_id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=True),
        sa.Column('display_name', sa.String(length=128), nullable=False),
        sa.Column('role', sa.String(length=16), server_default='member', nullable=False),
        sa.Column('joined_at', sa.DateTime(), nullable=False),
        sa.Column('left_at', sa.DateTime(), nullable=True),
        sa.Column('client_meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['room_id'], ['rooms.id'], name='fk_members_room'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_members_user'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_members_room_joined', 'room_members', ['room_id', 'joined_at'], unique=False)
    op.create_index('idx_members_room_user', 'room_members', ['room_id', 'user_id'], unique=False)

    # --- chat_messages ---
    op.create_table(
        'chat_messages',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('room_id', sa.String(length=64), nullable=False),
        sa.Column('seq', sa.BigInteger(), nullable=False),
        sa.Column('sender_type', sa.String(length=16), nullable=False),
        sa.Column('sender_member_id', sa.String(length=64), nullable=True),
        sa.Column('message_type', sa.String(length=16), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('lang', sa.String(length=8), nullable=True),
        sa.Column('start_ms', sa.Integer(), nullable=True),
        sa.Column('end_ms', sa.Integer(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['room_id'], ['rooms.id'], name='fk_msg_room'),
        sa.ForeignKeyConstraint(['sender_member_id'], ['room_members.id'], name='fk_msg_sender_member'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('room_id', 'seq', name='uq_room_seq')
    )
    op.create_index('idx_room_created', 'chat_messages', ['room_id', 'created_at'], unique=False)
    op.create_index('idx_room_sender_seq', 'chat_messages', ['room_id', 'sender_member_id', 'seq'], unique=False)

    # --- auth_tokens ---
    op.create_table(
        'auth_tokens',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('token_hash', sa.CHAR(length=64), nullable=False),
        sa.Column('issued_at', sa.DateTime(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('session_id', sa.String(length=64), nullable=True),
        sa.Column('device_meta', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], name='fk_tokens_user'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash')
    )
    op.create_index('idx_auth_tokens_user_expires', 'auth_tokens', ['user_id', 'expires_at'], unique=False)

    # --- live ---
    op.create_table(
        'live',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('room_id', sa.String(length=64), nullable=False),
        sa.Column('member_id', sa.String(length=64), nullable=False),
        sa.Column('seq', sa.BigInteger(), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('lang', sa.String(length=8), nullable=True),
        sa.Column('start_ms', sa.Integer(), nullable=True),
        sa.Column('end_ms', sa.Integer(), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['member_id'], ['room_members.id'], name='fk_live_member'),
        sa.ForeignKeyConstraint(['room_id'], ['rooms.id'], name='fk_live_room'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('room_id', 'seq', name='uq_live_room_seq')
    )
    op.create_index('idx_live_member_seq', 'live', ['member_id', 'seq'], unique=False)
    op.create_index('idx_live_room_created', 'live', ['room_id', 'created_at'], unique=False)

    # --- ai_events ---
    op.create_table(
        'ai_events',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('room_id', sa.String(length=64), nullable=False),
        sa.Column('seq', sa.BigInteger(), nullable=False),
        sa.Column('event_type', sa.String(length=16), nullable=False),
        sa.Column('source_live_id', sa.String(length=64), nullable=True),
        sa.Column('original_text', sa.Text(), nullable=True),
        sa.Column('original_lang', sa.String(length=8), nullable=True),
        sa.Column('translated_text', sa.Text(), nullable=True),
        sa.Column('translated_lang', sa.String(length=8), nullable=True),
        sa.Column('text', sa.Text(), nullable=True),
        sa.Column('lang', sa.String(length=8), nullable=True),
        sa.Column('meta', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['room_id'], ['rooms.id'], name='fk_ai_room'),
        sa.ForeignKeyConstraint(['source_live_id'], ['live.id'], name='fk_ai_source_live'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('room_id', 'seq', name='uq_ai_room_seq')
    )
    op.create_index('idx_ai_room_created', 'ai_events', ['room_id', 'created_at'], unique=False)
    op.create_index('idx_ai_source_live', 'ai_events', ['source_live_id'], unique=False)


def downgrade() -> None:
    op.drop_table('ai_events')
    op.drop_table('live')
    op.drop_table('auth_tokens')
    op.drop_table('chat_messages')
    op.drop_table('room_members')
    op.drop_table('rooms')
    op.drop_table('users')

