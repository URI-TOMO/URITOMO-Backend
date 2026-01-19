from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, JSON, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class UserFriend(Base):
    __tablename__ = "user_friends"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    
    requester_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    addressee_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)

    # status: 'pending' | 'accepted' | 'rejected' | 'blocked'
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    
    requested_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    responded_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)

    meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Constraints and Indexes
    __table_args__ = (
        # Prevent self-friending
        CheckConstraint("requester_id <> addressee_id", name="chk_user_friends_not_self"),
        
        # Prevent duplicate active relationship rows between same pair in same direction
        UniqueConstraint("requester_id", "addressee_id", name="uq_user_friends_pair_direction"),
        
        # Helpful indexes
        Index("idx_user_friends_requester", "requester_id", "status"),
        Index("idx_user_friends_addressee", "addressee_id", "status"),
        Index("idx_user_friends_status", "status", "requested_at"),
    )

    # Relationships
    requester: Mapped["User"] = relationship("User", foreign_keys=[requester_id], back_populates="sent_friend_requests")
    addressee: Mapped["User"] = relationship("User", foreign_keys=[addressee_id], back_populates="received_friend_requests")
