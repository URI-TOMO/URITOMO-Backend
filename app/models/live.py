from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ai import AIEvent
    from app.models.room import Room, RoomMember


class Live(Base):
    __tablename__ = "live"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    member_id: Mapped[str] = mapped_column(ForeignKey("room_members.id"), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    start_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    end_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    # Indexes
    __table_args__ = (
        UniqueConstraint("room_id", "seq", name="uq_live_room_seq"),
        Index("idx_live_room_created", "room_id", "created_at"),
        Index("idx_live_member_seq", "member_id", "seq"),
    )

    # Relationships
    room: Mapped["Room"] = relationship("Room", back_populates="live_events")
    member: Mapped["RoomMember"] = relationship("RoomMember", back_populates="live_utterances")
    ai_events: Mapped[List["AIEvent"]] = relationship("AIEvent", back_populates="source_live")
