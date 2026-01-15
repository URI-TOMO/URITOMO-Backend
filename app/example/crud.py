"""
CRUD service layer for User and Room models.
"""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.example.schemas import RoomCreate, RoomUpdate, UserCreate, UserUpdate
from app.models.room import Room
from app.models.user import User


# ============ User CRUD ============

class UserCRUD:
    """CRUD operations for User model"""

    @staticmethod
    async def create(db: AsyncSession, user_data: UserCreate) -> User:
        """Create a new user"""
        user = User(
            id=user_data.id,
            email=user_data.email,
            display_name=user_data.display_name,
            locale=user_data.locale,
            status=user_data.status,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(
        db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[User]:
        """Get all users with pagination"""
        result = await db.execute(select(User).offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def update(
        db: AsyncSession, user_id: str, user_data: UserUpdate
    ) -> Optional[User]:
        """Update user by ID"""
        user = await UserCRUD.get_by_id(db, user_id)
        if not user:
            return None

        # Update only provided fields
        update_data = user_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(user, field, value)

        await db.commit()
        await db.refresh(user)
        return user

    @staticmethod
    async def delete(db: AsyncSession, user_id: str) -> bool:
        """Delete user by ID"""
        user = await UserCRUD.get_by_id(db, user_id)
        if not user:
            return False

        await db.delete(user)
        await db.commit()
        return True


# ============ Room CRUD ============

class RoomCRUD:
    """CRUD operations for Room model"""

    @staticmethod
    async def create(db: AsyncSession, room_data: RoomCreate) -> Room:
        """Create a new room"""
        room = Room(
            id=room_data.id,
            title=room_data.title,
            created_by=room_data.created_by,
            status=room_data.status,
        )
        db.add(room)
        await db.commit()
        await db.refresh(room)
        return room

    @staticmethod
    async def get_by_id(db: AsyncSession, room_id: str) -> Optional[Room]:
        """Get room by ID"""
        result = await db.execute(select(Room).where(Room.id == room_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_all(
        db: AsyncSession, skip: int = 0, limit: int = 100
    ) -> List[Room]:
        """Get all rooms with pagination"""
        result = await db.execute(select(Room).offset(skip).limit(limit))
        return list(result.scalars().all())

    @staticmethod
    async def get_by_creator(
        db: AsyncSession, creator_id: str, skip: int = 0, limit: int = 100
    ) -> List[Room]:
        """Get rooms created by a specific user"""
        result = await db.execute(
            select(Room)
            .where(Room.created_by == creator_id)
            .offset(skip)
            .limit(limit)
        )
        return list(result.scalars().all())

    @staticmethod
    async def update(
        db: AsyncSession, room_id: str, room_data: RoomUpdate
    ) -> Optional[Room]:
        """Update room by ID"""
        room = await RoomCRUD.get_by_id(db, room_id)
        if not room:
            return None

        # Update only provided fields
        update_data = room_data.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(room, field, value)

        await db.commit()
        await db.refresh(room)
        return room

    @staticmethod
    async def delete(db: AsyncSession, room_id: str) -> bool:
        """Delete room by ID (soft delete by setting ended_at)"""
        room = await RoomCRUD.get_by_id(db, room_id)
        if not room:
            return False

        room.ended_at = datetime.utcnow()
        room.status = "ended"
        await db.commit()
        return True

    @staticmethod
    async def hard_delete(db: AsyncSession, room_id: str) -> bool:
        """Permanently delete room by ID"""
        room = await RoomCRUD.get_by_id(db, room_id)
        if not room:
            return False

        await db.delete(room)
        await db.commit()
        return True
