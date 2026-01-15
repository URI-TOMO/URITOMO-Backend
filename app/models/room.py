from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.ai import AIEvent
    from app.models.live import Live
    from app.models.message import ChatMessage
    from app.models.user import User


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_by: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_rooms_status_created", "status", "created_at"),
    )

    # Relationships
    creator: Mapped["User"] = relationship("User", back_populates="created_rooms")
    members: Mapped[List["RoomMember"]] = relationship("RoomMember", back_populates="room")
    messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="room")
    live_events: Mapped[List["Live"]] = relationship("Live", back_populates="room")
    ai_events: Mapped[List["AIEvent"]] = relationship("AIEvent", back_populates="room")


class RoomMember(Base):
    __tablename__ = "room_members"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # participantId
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id"), nullable=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    left_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    client_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_members_room_joined", "room_id", "joined_at"),
        Index("idx_members_room_user", "room_id", "user_id"),
    )

    # Relationships
    room: Mapped["Room"] = relationship("Room", back_populates="members")
    user: Mapped[Optional["User"]] = relationship("User", back_populates="memberships")
    sent_messages: Mapped[List["ChatMessage"]] = relationship("ChatMessage", back_populates="sender_member")
    live_utterances: Mapped[List["Live"]] = relationship("Live", back_populates="member")
