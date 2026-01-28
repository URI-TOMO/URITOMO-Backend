from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List, Optional
from pydantic import BaseModel, EmailStr

from app.infra.db import get_db
from app.models.user import User
from app.models.friend import UserFriend
from app.models.room import Room, RoomMember
from app.core.token import get_current_user_id

router = APIRouter()

# ============ Schemas ============

class UserMainInfo(BaseModel):
    display_name: str
    email: Optional[str]
    lang: Optional[str]

class FriendInfo(BaseModel):
    id: str
    friend_name: str
    email: Optional[str]

class RoomInfo(BaseModel):
    id: str
    name: str

class MainPageResponse(BaseModel):
    user: UserMainInfo
    friend_count : int
    user_friends: List[FriendInfo]
    rooms: List[RoomInfo]

# ============ Endpoint ============

@router.get("/user/main", response_model=MainPageResponse, tags=["main"])


async def get_main_page_data(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    """
    Get all necessary data for the main page:
    - User info (name, email)
    - Friend list (id, name, total count, email)
    - Joined rooms (id, title)
    """
    
    try:
        # 1. Fetch User
        print(f"DEBUG: Fetching main page data for user_id: {user_id}")
        user_stmt = select(User).where(User.id == user_id)
        user_result = await db.execute(user_stmt)
        user = user_result.scalar_one_or_none()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # 2. Fetch Friends
        print("DEBUG: Fetching friends...")
        friend_stmt = (
            select(UserFriend)
            .where(
                or_(UserFriend.requester_id == user_id, UserFriend.addressee_id == user_id),
                UserFriend.status == "accepted"
            )
        )
        friends_result = await db.execute(friend_stmt)
        friendships = friends_result.scalars().all()

        total_friends = len(friendships)
        friends_list = []
        
        for friendship in friendships:
            friend_id = friendship.addressee_id if friendship.requester_id == user_id else friendship.requester_id
            
            friend_user_stmt = select(User).where(User.id == friend_id)
            friend_user_result = await db.execute(friend_user_stmt)
            friend_user = friend_user_result.scalar_one_or_none()
            
            if friend_user:
                name = friendship.friend_name or friend_user.display_name
                friends_list.append(FriendInfo(
                    id=friend_user.id,
                    friend_name=name,
                    email=friend_user.email
                ))

        # 3. Fetch Rooms
        print("DEBUG: Fetching rooms...")
        room_stmt = (
            select(Room)
            .join(RoomMember, Room.id == RoomMember.room_id)
            .where(RoomMember.user_id == user_id, Room.status == "active")
        )
        rooms_result = await db.execute(room_stmt)
        rooms = rooms_result.scalars().all()
        
        room_list = [
            RoomInfo(id=r.id, name=r.title or "Untitled Room") 
            for r in rooms
        ]

        print("DEBUG: Successfully assembled MainPageResponse")
        return MainPageResponse(
            user=UserMainInfo(display_name=user.display_name, email=user.email, lang=user.locale),
            friend_count=total_friends,
            user_friends=friends_list,
            rooms=room_list
        )
    except Exception as e:
        print(f"ERROR in get_main_page_data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e
