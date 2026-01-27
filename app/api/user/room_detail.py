from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, EmailStr
from typing import List, Optional

from app.infra.db import get_db
from app.models.room import Room, RoomMember
from app.models.user import User
from app.core.token import CurrentUserDep
from app.core.errors import AppError

router = APIRouter(tags=["main"])

# ============ Schemas ============

class MemberInfo(BaseModel):
    id: str
    name: str
    status: str
    locale: Optional[str] = None

class RoomDetailResponse(BaseModel):
    id: str
    name: str
    members: List[MemberInfo]
    participant_count: int

class AddMemberRequest(BaseModel):
    email: EmailStr

class AddMemberResponse(BaseModel):
    id: str
    name: str
    locale: Optional[str] = None

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
            message="Room not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="ROOM_NOT_FOUND",
        )
    
    # Check if current user is a member of the room
    is_member = any(
        member.user_id == current_user_id 
        for member in room.members 
        if member.user_id is not None and member.left_at is None
    )
    
    if not is_member:
        raise AppError(
            message="You are not a member of this room",
            status_code=status.HTTP_403_FORBIDDEN,
            code="ROOM_MEMBER_FORBIDDEN",
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
                    status=user_status,
                    locale=member.user.locale if member.user else None,
                )
            )
    
    return RoomDetailResponse(
        id=room.id,
        name=room.title or "Untitled Room",
        members=members_info,
        participant_count=len(members_info)
    )


@router.post("/rooms/{room_id}/members", response_model=AddMemberResponse, status_code=status.HTTP_201_CREATED)
async def add_room_member(
    room_id: str,
    payload: AddMemberRequest,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    """
    Add a member to a room by email.

    Requires authentication via Bearer token.
    """
    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalar_one_or_none()
    if not room:
        raise AppError(
            message="Room not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="ROOM_NOT_FOUND",
        )

    if room.created_by != current_user_id:
        owner_result = await db.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id,
                RoomMember.left_at.is_(None),
            )
        )
        owner_member = owner_result.scalar_one_or_none()
        if not owner_member or owner_member.role != "owner":
            raise AppError(
                message="You do not have permission to add members to this room",
                status_code=status.HTTP_403_FORBIDDEN,
                code="ROOM_MEMBER_FORBIDDEN",
            )

    normalized_email = payload.email.strip().lower()
    user_result = await db.execute(
        select(User).where(func.lower(User.email) == normalized_email)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        raise AppError(
            message=f"User with email '{payload.email}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="USER_NOT_FOUND",
        )

    member_result = await db.execute(
        select(RoomMember).where(
            RoomMember.room_id == room_id,
            RoomMember.user_id == user.id,
        )
    )
    existing_member = member_result.scalar_one_or_none()
    if existing_member:
        if existing_member.left_at is None:
            raise AppError(
                message="User is already a member of this room",
                status_code=status.HTTP_409_CONFLICT,
                code="ROOM_MEMBER_EXISTS",
            )
        existing_member.left_at = None
        existing_member.joined_at = datetime.utcnow()
        existing_member.display_name = user.display_name
        await db.commit()
        return AddMemberResponse(
            id=user.id,
            name=user.display_name,
            locale=user.locale,
        )

    new_member = RoomMember(
        id=f"member_{uuid.uuid4().hex}",
        room_id=room_id,
        user_id=user.id,
        display_name=user.display_name,
        role="member",
        joined_at=datetime.utcnow(),
    )
    db.add(new_member)
    await db.commit()

    return AddMemberResponse(
        id=user.id,
        name=user.display_name,
        locale=user.locale,
    )
