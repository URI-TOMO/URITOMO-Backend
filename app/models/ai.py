from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.live import Live
    from app.models.room import Room


class AIEvent(Base):
    __tablename__ = "ai_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)

    # event_type: translation | moderation | summary | asr | intent | error
    event_type: Mapped[str] = mapped_column(String(16), nullable=False)
    source_live_id: Mapped[Optional[str]] = mapped_column(ForeignKey("live.id"), nullable=True)

    # translation columns
    original_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    original_lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    translated_lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    # assistant columns
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        UniqueConstraint("room_id", "seq", name="uq_ai_room_seq"),
        Index("idx_ai_room_created", "room_id", "created_at"),
        Index("idx_ai_source_live", "source_live_id"),
    )

    # Relationships
    room: Mapped["Room"] = relationship("Room", back_populates="ai_events")
    source_live: Mapped[Optional["Live"]] = relationship("Live", back_populates="ai_events")
