from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import BigInteger, CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.friend import UserFriend
    from app.models.user import User


class DmThread(Base):
    __tablename__ = "dm_threads"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_friend_id: Mapped[str] = mapped_column(ForeignKey("user_friends.id", onupdate="CASCADE", ondelete="RESTRICT"), nullable=False, unique=True)
    
    # status: active | archived | ended | deleted
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_dm_threads_status_created", "status", "created_at"),
    )

    # Relationships
    friend_relationship: Mapped["UserFriend"] = relationship("UserFriend", back_populates="dm_thread")
    participants: Mapped[List["DmParticipant"]] = relationship("DmParticipant", back_populates="thread", cascade="all, delete-orphan")
    messages: Mapped[List["DmMessage"]] = relationship("DmMessage", back_populates="thread", cascade="all, delete-orphan")


class DmParticipant(Base):
    __tablename__ = "dm_participants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("dm_threads.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", onupdate="CASCADE", ondelete="RESTRICT"), nullable=False)
    
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    left_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint("thread_id", "user_id", name="uq_dm_participants_thread_user"),
        Index("idx_dm_participants_thread", "thread_id"),
        Index("idx_dm_participants_user", "user_id"),
    )

    # Relationships
    thread: Mapped["DmThread"] = relationship("DmThread", back_populates="participants")
    user: Mapped["User"] = relationship("User", back_populates="dm_participations")


class DmMessage(Base):
    __tablename__ = "dm_messages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(ForeignKey("dm_threads.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)
    
    # sender_type: human | ai | system
    sender_type: Mapped[str] = mapped_column(String(16), nullable=False)
    
    sender_user_id: Mapped[Optional[str]] = mapped_column(ForeignKey("users.id", onupdate="CASCADE", ondelete="SET NULL"), nullable=True)
    
    # message_type: text | translation | notice | error | tool
    message_type: Mapped[str] = mapped_column(String(32), nullable=False)
    
    text: Mapped[str] = mapped_column(Text, nullable=False)
    lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    start_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    end_ms: Mapped[Optional[int]] = mapped_column(nullable=True)
    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    # Constraints and Indexes
    __table_args__ = (
        UniqueConstraint("thread_id", "seq", name="uq_dm_messages_thread_seq"),
        Index("idx_dm_messages_thread_created", "thread_id", "created_at"),
        Index("idx_dm_messages_sender_user", "sender_user_id"),
    )

    # Relationships
    thread: Mapped["DmThread"] = relationship("DmThread", back_populates="messages")
    sender_user: Mapped[Optional["User"]] = relationship("User")
