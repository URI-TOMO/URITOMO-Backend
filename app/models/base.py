from datetime import datetime
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all models"""
    
    # Common columns for all models can go here if needed
    pass


class TimestampMixin:
    """Mixin for adds created_at and updated_at columns"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False), 
        default=datetime.utcnow,
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=False),
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )
