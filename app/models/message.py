from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.room import Room, RoomMember


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    # sender_type: human | ai | system
    sender_type: Mapped[str] = mapped_column(String(16), nullable=False)
    sender_member_id: Mapped[Optional[str]] = mapped_column(ForeignKey("room_members.id"), nullable=True)
    
    # message_type: text | translation | notice | error | tool
    message_type: Mapped[str] = mapped_column(String(16), nullable=False)
    
    text: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    start_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    end_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint("room_id", "seq", name="uq_room_seq"),
        Index("idx_room_created", "room_id", "created_at"),
        Index("idx_room_sender_member", "sender_member_id"),
    )

    # Relationships
    room: Mapped["Room"] = relationship("Room", back_populates="messages")
    sender_member: Mapped[Optional["RoomMember"]] = relationship("RoomMember", back_populates="sent_messages")
