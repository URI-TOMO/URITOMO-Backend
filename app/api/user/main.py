from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from typing import List, Optional
from pydantic import BaseModel, EmailStr

from app.infra.db import get_db
from app.models.user import User
from app.models.friend import UserFriend
from app.models.room import Room, RoomMember

router = APIRouter()

# ============ Schemas ============

class UserMainInfo(BaseModel):
    display_name: str
    email: Optional[EmailStr]

class FriendInfo(BaseModel):
    id: str
    friend_name: str
    email: Optional[EmailStr]

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
    user_id: str = Query(..., description="User ID to fetch data for (Mock for token)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all necessary data for the main page:
    - User info (name, email)
    - Friend list (id, name, total count, email)
    - Joined rooms (id, title)
    """
    
    # 1. Fetch User
    user_stmt = select(User).where(User.id == user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # 2. Fetch Friends
    # We join with User to get friend's email and actual display_name
    # Since UserFriend can have requester as me or addressee as me:
    friend_stmt = (
        select(UserFriend, User)
        .join(User, or_(
            (UserFriend.requester_id == User.id) & (UserFriend.addressee_id == user_id),
            (UserFriend.addressee_id == User.id) & (UserFriend.requester_id == user_id)
        ))
        .where(
            or_(UserFriend.requester_id == user_id, UserFriend.addressee_id == user_id),
            UserFriend.status == "accepted"
        )
    )
    friends_result = await db.execute(friend_stmt)
    friends_rows = friends_result.all()

    total_friends = len(friends_rows)
    friends_list = []
    for f_row, u_row in friends_rows:
        # custom friend_name from UserFriend table or display_name from User table
        name = f_row.friend_name or u_row.display_name
        friends_list.append(FriendInfo(
            id=u_row.id,
            friend_name=name,
            email=u_row.email
        ))

    # 3. Fetch Rooms
    # Rooms where the user is a member
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

    return MainPageResponse(
        user=UserMainInfo(display_name=user.display_name, email=user.email),
        friend_count=total_friends,
        user_friends=friends_list,
        rooms=room_list
    )
