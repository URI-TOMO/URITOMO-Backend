from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.room import Room, RoomLiveSession, RoomMember


class RoomSttResult(Base):
    __tablename__ = "room_stt_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("room_live_sessions.id"), nullable=False)
    member_id: Mapped[str] = mapped_column(ForeignKey("room_members.id"), nullable=False)

    user_lang: Mapped[str] = mapped_column(String(8), nullable=False)
    stt_text: Mapped[str] = mapped_column(Text, nullable=False)
    translated_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    translated_lang: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    seq: Mapped[int] = mapped_column(BigInteger, nullable=False)

    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("session_id", "seq", name="uq_room_stt_session_seq"),
        Index("idx_room_stt_room_session_created", "room_id", "session_id", "created_at"),
        Index("idx_room_stt_member_seq", "member_id", "seq"),
    )

    room: Mapped["Room"] = relationship("Room", back_populates="stt_results")
    session: Mapped["RoomLiveSession"] = relationship("RoomLiveSession", back_populates="stt_results")
    member: Mapped["RoomMember"] = relationship("RoomMember", back_populates="stt_results")


class RoomAiResponse(Base):
    __tablename__ = "room_ai_responses"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    room_id: Mapped[str] = mapped_column(ForeignKey("rooms.id"), nullable=False)
    session_id: Mapped[str] = mapped_column(ForeignKey("room_live_sessions.id"), nullable=False)

    lang: Mapped[str] = mapped_column(String(8), nullable=False)
    stt_text: Mapped[str] = mapped_column(Text, nullable=False)
    stt_seq_end: Mapped[int] = mapped_column(BigInteger, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)

    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_room_ai_room_session_created", "room_id", "session_id", "created_at"),
        Index("idx_room_ai_session_stt_end", "session_id", "stt_seq_end"),
    )

    room: Mapped["Room"] = relationship("Room", back_populates="ai_responses")
    session: Mapped["RoomLiveSession"] = relationship("RoomLiveSession", back_populates="ai_responses")
