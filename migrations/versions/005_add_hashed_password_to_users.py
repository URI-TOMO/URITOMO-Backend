"""add hashed_password to users

Revision ID: b502c0ce3b3e
Revises: 004
Create Date: 2026-01-21 19:25:37.057106

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b502c0ce3b3e'
down_revision: Union[str, None] = '004'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('users', sa.Column('hashed_password', sa.String(length=255), nullable=True))


def downgrade() -> None:
    op.drop_column('users', 'hashed_password')
