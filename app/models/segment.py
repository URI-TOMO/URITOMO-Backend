"""
Transcript Segment Model

Stores original transcript segments ingested from clients.
"""

from typing import List, Optional

from sqlalchemy import String, Integer, ForeignKey, Text, Float, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class TranscriptSegment(Base, TimestampMixin):
    __tablename__ = "transcript_segments"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id"), index=True)
    
    # Temporal info
    timestamp: Mapped[float] = mapped_column(Float, index=True) # Relative timestamp or epoch
    duration: Mapped[float] = mapped_column(Float, default=0.0)
    
    # Content
    speaker_name: Mapped[str] = mapped_column(String(100))
    language: Mapped[str] = mapped_column(String(5))
    text: Mapped[str] = mapped_column(Text)
    
    # Metadata
    is_final: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="segments")
    translations: Mapped[List["TranslationSegment"]] = relationship(back_populates="original_segment", cascade="all, delete-orphan")
