"""
CRUD service layer for User and Mock data generation.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.example.user.schemas import UserCreate
from app.models.friend import UserFriend
from app.models.room import Room, RoomMember
from app.models.user import User


class UserCRUD:
    """CRUD operations for User model and Mock data"""

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
        """Get user by ID (Helper for other operations)"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def create_mock_data(db: AsyncSession, user_id: str) -> dict:
        """
        Sets up mock data for a given user_id:
        - Creates the user if they don't exist
        - Adds 2 friends (already accepted)
        - Adds 2 active rooms the user is a member of
        """
        # 1. Ensure main user exists
        user = await UserCRUD.get_by_id(db, user_id)
        
        if not user:
            user = User(
                id=user_id,
                display_name=f"User_{user_id[:4]}",
                email=f"{user_id}@example.com",
                locale="ja",
                status="active"
            )
            db.add(user)
        
        # 2. Add friends
        for i in range(1, 3):
            friend_id = f"friend_{i}_{uuid.uuid4().hex[:6]}"
            
            # Create friend user
            friend_user = User(
                id=friend_id,
                display_name=f"Friend {i}",
                email=f"friend_{i}_{uuid.uuid4().hex[:6]}@example.com",
                status="active"
            )
            db.add(friend_user)
            
            # Create accepted friendship
            friendship = UserFriend(
                id=f"fs_{uuid.uuid4().hex[:8]}",
                requester_id=user_id,
                addressee_id=friend_id,
                status="accepted",
                friend_name=f"Bestie {i}"
            )
            db.add(friendship)
            
        # 3. Add rooms
        for i in range(1, 3):
            room_id = f"room_{i}_{uuid.uuid4().hex[:6]}"
            
            # Create room
            room = Room(
                id=room_id,
                title=f"Japanese Study {i}",
                created_by=user_id,
                status="active"
            )
            db.add(room)
            
            # Add user as member
            member = RoomMember(
                id=f"rm_{uuid.uuid4().hex[:8]}",
                room_id=room_id,
                user_id=user_id,
                display_name=user.display_name,
                role="owner" if i == 1 else "member"
            )
            db.add(member)

        await db.commit()
        return {
            "status": "success",
            "message": f"Setup mock data for user {user_id}",
            "user_id": user_id
        }
