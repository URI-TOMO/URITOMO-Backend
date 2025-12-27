"""
Glossary Model

Organization-specific glossary entries.
"""

from typing import Optional

from sqlalchemy import String, Integer, ForeignKey, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class GlossaryEntry(Base, TimestampMixin):
    __tablename__ = "glossary_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    
    term: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Context
    context_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # For RAG
    embedding_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True) # Qdrant Point ID
    
    # Relationships
    org: Mapped["Organization"] = relationship(back_populates="glossary_entries")
