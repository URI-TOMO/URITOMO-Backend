from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.room import Room, RoomMember
    from app.models.token import AuthToken


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    email: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    display_name: Mapped[str] = mapped_column(String(128), nullable=False)
    locale: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")

    # Relationships
    created_rooms: Mapped[List["Room"]] = relationship("Room", back_populates="creator")
    memberships: Mapped[List["RoomMember"]] = relationship("RoomMember", back_populates="user")
    tokens: Mapped[List["AuthToken"]] = relationship("AuthToken", back_populates="user")
