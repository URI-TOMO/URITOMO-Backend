"""
User Models

User and UserOrg association.
"""

from typing import List, Optional

from sqlalchemy import String, Boolean, ForeignKey, Integer, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_superuser: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    orgs: Mapped[List["UserOrg"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    meetings_hosted: Mapped[List["Meeting"]] = relationship(back_populates="host")


class UserOrg(Base, TimestampMixin):
    """Many-to-Many relationship between User and Organization with role"""
    __tablename__ = "user_orgs"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    org_id: Mapped[int] = mapped_column(ForeignKey("organizations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20), default="member")  # owner, admin, member

    # Relationships
    user: Mapped["User"] = relationship(back_populates="orgs")
    org: Mapped["Organization"] = relationship(back_populates="members")
