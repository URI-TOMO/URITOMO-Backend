"""
Meeting Models

Meeting, Participant, and Setting.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import String, Integer, ForeignKey, DateTime, Boolean, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Meeting(Base, TimestampMixin):
    __tablename__ = "meetings"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    start_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    end_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    
    status: Mapped[str] = mapped_column(String(20), default="scheduled")  # scheduled, active, completed, cancelled
    
    # Foreign Keys
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    host_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    
    # Relationships
    org: Mapped["Organization"] = relationship(back_populates="meetings")
    host: Mapped["User"] = relationship(back_populates="meetings_hosted")
    
    participants: Mapped[List["MeetingParticipant"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    settings: Mapped["MeetingSetting"] = relationship(back_populates="meeting", uselist=False, cascade="all, delete-orphan")
    
    segments: Mapped[List["TranscriptSegment"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")
    summaries: Mapped[List["Summary"]] = relationship(back_populates="meeting", cascade="all, delete-orphan")


class MeetingParticipant(Base, TimestampMixin):
    __tablename__ = "meeting_participants"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id"), index=True)
    
    # Can be a registered user or a guest
    user_id: Mapped[Optional[int]] = mapped_column(ForeignKey("users.id"), nullable=True)
    guest_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    joined_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    role: Mapped[str] = mapped_column(String(20), default="attendee")  # host, presenter, attendee

    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="participants")
    user: Mapped["User"] = relationship()


class MeetingSetting(Base, TimestampMixin):
    __tablename__ = "meeting_settings"

    id: Mapped[int] = mapped_column(primary_key=True)
    meeting_id: Mapped[int] = mapped_column(ForeignKey("meetings.id"), unique=True)
    
    source_lang: Mapped[str] = mapped_column(String(5), default="ja")
    target_lang: Mapped[str] = mapped_column(String(5), default="ko")
    
    # Feature flags
    enable_translation: Mapped[bool] = mapped_column(Boolean, default=True)
    enable_explanation: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Style customization
    style_profile: Mapped[str] = mapped_column(String(20), default="business") # business, casual
    explanation_level: Mapped[str] = mapped_column(String(20), default="normal") # verbose, normal, concise
    
    # Relationships
    meeting: Mapped["Meeting"] = relationship(back_populates="settings")
