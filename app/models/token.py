from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CHAR, DateTime, ForeignKey, Index, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False)
    token_type: Mapped[str] = mapped_column(String(16), nullable=False)  # access | refresh | ws
    scope: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    device_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_tokens_user_type", "user_id", "token_type", "expires_at"),
        Index("idx_tokens_hash", "token_hash"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="tokens")
