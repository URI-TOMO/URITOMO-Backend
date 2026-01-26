from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from pydantic import BaseModel
from typing import List

from app.infra.db import get_db
from app.models.room import Room, RoomMember
from app.models.user import User
from app.core.token import CurrentUserDep
from app.core.errors import AppError

router = APIRouter(tags=["rooms"])

# ============ Schemas ============

class MemberInfo(BaseModel):
    id: str
    name: str
    status: str

class RoomDetailResponse(BaseModel):
    id: str
    name: str
    members: List[MemberInfo]
    participant_count: int

# ============ Endpoints ============

@router.get("/rooms/{room_id}", response_model=RoomDetailResponse)
async def get_room_detail(
    room_id: str,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    """
    Get room details including members information.
    
    Requires authentication via Bearer token.
    """
    # Query room with members
    stmt = (
        select(Room)
        .options(selectinload(Room.members).selectinload(RoomMember.user))
        .where(Room.id == room_id)
    )
    result = await db.execute(stmt)
    room = result.scalar_one_or_none()
    
    if not room:
        raise AppError(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Room not found"
        )
    
    # Check if current user is a member of the room
    is_member = any(
        member.user_id == current_user_id 
        for member in room.members 
        if member.user_id is not None and member.left_at is None
    )
    
    if not is_member:
        raise AppError(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not a member of this room"
        )
    
    # Build members list (only active members who haven't left)
    members_info = []
    for member in room.members:
        if member.left_at is None:  # Only active members
            # Determine status (online/offline)
            # For now, we'll use a simple logic - you can enhance this with Redis presence tracking
            user_status = "offline"
            if member.user:
                # You can implement actual online status tracking with Redis
                # For now, defaulting to offline
                user_status = "offline"
            
            members_info.append(
                MemberInfo(
                    id=member.user_id or member.id,
                    name=member.display_name,
                    status=user_status
                )
            )
    
    return RoomDetailResponse(
        id=room.id,
        name=room.title or "Untitled Room",
        members=members_info,
        participant_count=len(members_info)
    )
