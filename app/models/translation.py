"""
Translation Segment Model

Stores translated text and cultural explanations.
"""

from typing import Optional

from sqlalchemy import String, Integer, ForeignKey, Text, Float, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class TranslationSegment(Base, TimestampMixin):
    __tablename__ = "translation_segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    original_segment_id: Mapped[int] = mapped_column(ForeignKey("transcript_segments.id"), index=True)
    
    target_lang: Mapped[str] = mapped_column(String(5))
    translated_text: Mapped[str] = mapped_column(Text)
    
    # Explanation
    has_explanation: Mapped[bool] = mapped_column(Boolean, default=False)
    explanation_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # AI Metadata
    provider: Mapped[str] = mapped_column(String(20), default="openai")
    model: Mapped[str] = mapped_column(String(50), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # RAG Context (which cards were used)
    rag_context: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    # Relationships
    original_segment: Mapped["TranscriptSegment"] = relationship(back_populates="translations")
