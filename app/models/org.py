"""
Organization Model
"""

from typing import List

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Organization(Base, TimestampMixin):
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True, nullable=False)
    slug: Mapped[str] = mapped_column(String(50), unique=True, index=True, nullable=False)
    
    # Config
    default_source_lang: Mapped[str] = mapped_column(String(5), default="ja")
    default_target_lang: Mapped[str] = mapped_column(String(5), default="ko")

    # Relationships
    members: Mapped[List["UserOrg"]] = relationship(back_populates="org")
    meetings: Mapped[List["Meeting"]] = relationship(back_populates="org")
    glossary_entries: Mapped[List["GlossaryEntry"]] = relationship(back_populates="org")
