from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

from sqlalchemy import CHAR, Boolean, DateTime, ForeignKey, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base

if TYPE_CHECKING:
    from app.models.user import User


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), nullable=False)
    token_hash: Mapped[str] = mapped_column(CHAR(64), nullable=False, unique=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, default=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    device_meta: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Indexes
    __table_args__ = (
        Index("idx_auth_tokens_user_expires", "user_id", "expires_at"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="tokens")


class RefreshToken(Base):
    """
    Refresh Token Model
    
    Stores refresh tokens for JWT authentication system with:
    - Secure bcrypt hashing
    - Expiration tracking
    - Token rotation support
    - Revocation capability
    """
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id"), nullable=False, index=True)
    token_hash: Mapped[str] = mapped_column(String, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    replaced_by_token: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    # Indexes
    __table_args__ = (
        Index("idx_refresh_tokens_user_id", "user_id"),
        Index("idx_refresh_tokens_expires_at", "expires_at"),
        Index("idx_refresh_tokens_revoked", "revoked"),
    )

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="refresh_tokens")
