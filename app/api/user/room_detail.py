from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from pydantic import BaseModel, EmailStr
from typing import List, Optional

from app.infra.db import get_db
from app.models.room import Room, RoomMember, RoomInvitation
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

class InviteMemberResponse(BaseModel):
    message: str
    invite_id: str
    status: str

class AcceptInviteResponse(BaseModel):
    message: str
    room_id: str

class RejectInviteResponse(BaseModel):
    message: str

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


@router.post("/rooms/{room_id}/members", response_model=InviteMemberResponse, status_code=status.HTTP_200_OK)
async def invite_room_member(
    room_id: str,
    payload: AddMemberRequest,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    """
    Send an invitation to a user to join the room.
    """
    # 1. Check Room existence
    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalar_one_or_none()
    if not room:
        raise AppError(
            message="Room not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="ROOM_NOT_FOUND",
        )

    # 2. Permission check (Creator or Owner? currently just check creator or assume owner role)
    # Reusing existing logic: Check if room.created_by == current or if member role is owner
    has_permission = room.created_by == current_user_id
    if not has_permission:
        # Check ownership via membership
        owner_result = await db.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id,
                RoomMember.left_at.is_(None),
            )
        )
        owner_member = owner_result.scalar_one_or_none()
        if owner_member and owner_member.role == "owner":
            has_permission = True
    
    if not has_permission:
        raise AppError(
            message="You do not have permission to invite members to this room",
            status_code=status.HTTP_403_FORBIDDEN,
            code="ROOM_MEMBER_FORBIDDEN",
        )

    # 3. Find target user
    normalized_email = payload.email.strip().lower()
    user_result = await db.execute(
        select(User).where(func.lower(User.email) == normalized_email)
    )
    target_user = user_result.scalar_one_or_none()
    if not target_user:
        raise AppError(
            message=f"User with email '{payload.email}' not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="USER_NOT_FOUND",
        )

    # 4. Check if already member
    member_result = await db.execute(
        select(RoomMember).where(
            RoomMember.room_id == room_id,
            RoomMember.user_id == target_user.id,
            RoomMember.left_at.is_(None),
        )
    )
    if member_result.scalar_one_or_none():
         raise AppError(
            message="User is already a member of this room",
            status_code=status.HTTP_409_CONFLICT,
            code="ROOM_MEMBER_EXISTS",
        )

    # 5. Check if invitation already exists (pending)
    invite_result = await db.execute(
        select(RoomInvitation).where(
            RoomInvitation.room_id == room_id,
            RoomInvitation.invitee_id == target_user.id,
            RoomInvitation.status == "pending"
        )
    )
    if invite_result.scalar_one_or_none():
        raise AppError(
            message="Invitation already pending for this user.",
            status_code=status.HTTP_409_CONFLICT,
            code="INVITATION_EXISTS",
        )

    # 6. Create Invitation
    new_invite = RoomInvitation(
        id=f"inv_{uuid.uuid4().hex}",
        room_id=room_id,
        inviter_id=current_user_id,
        invitee_id=target_user.id,
        status="pending",
        created_at=datetime.utcnow()
    )
    db.add(new_invite)
    await db.commit()

    return InviteMemberResponse(
        message="Invitation sent successfully.",
        invite_id=new_invite.id,
        status="pending"
    )


@router.post("/rooms/invite/{invite_id}/accept", response_model=AcceptInviteResponse)
async def accept_room_invite(
    invite_id: str,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch Invitation
    stmt = select(RoomInvitation).where(RoomInvitation.id == invite_id)
    result = await db.execute(stmt)
    invite = result.scalar_one_or_none()

    if not invite:
        raise AppError(
            message="Invitation not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            code="INVITATION_NOT_FOUND",
        )

    # 2. Check if it's for current user
    if invite.invitee_id != current_user_id:
        raise AppError(
            message="You are not authorized to accept this invitation.",
            status_code=status.HTTP_403_FORBIDDEN,
            code="INVITATION_FORBIDDEN",
        )

    if invite.status != "pending":
        raise AppError(
            message=f"Invitation is already {invite.status}.",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVITATION_INVALID_STATUS",
        )

    # 3. Update Invitation Status
    invite.status = "accepted"
    invite.responded_at = datetime.utcnow()

    # 4. Add Member to Room
    # Check if a (left) member record exists to reuse
    member_result = await db.execute(
        select(RoomMember).where(
            RoomMember.room_id == invite.room_id,
            RoomMember.user_id == current_user_id
        )
    )
    existing_member = member_result.scalar_one_or_none()
    
    # We need user display name
    user_result = await db.execute(select(User).where(User.id == current_user_id))
    user = user_result.scalar_one()

    if existing_member:
        existing_member.left_at = None
        existing_member.joined_at = datetime.utcnow()
        existing_member.display_name = user.display_name # Update name
        existing_member.role = "member" # Reset role to member just in case
    else:
        new_member = RoomMember(
            id=f"member_{uuid.uuid4().hex}",
            room_id=invite.room_id,
            user_id=current_user_id,
            display_name=user.display_name,
            role="member",
            joined_at=datetime.utcnow(),
        )
        db.add(new_member)

    await db.commit()

    return AcceptInviteResponse(
        message="Joined room successfully.",
        room_id=invite.room_id
    )


@router.post("/rooms/invite/{invite_id}/reject", response_model=RejectInviteResponse)
async def reject_room_invite(
    invite_id: str,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    # 1. Fetch Invitation
    stmt = select(RoomInvitation).where(RoomInvitation.id == invite_id)
    result = await db.execute(stmt)
    invite = result.scalar_one_or_none()

    if not invite:
        raise AppError(
            message="Invitation not found.",
            status_code=status.HTTP_404_NOT_FOUND,
            code="INVITATION_NOT_FOUND",
        )

    # 2. Check if it's for current user
    if invite.invitee_id != current_user_id:
        raise AppError(
            message="You are not authorized to reject this invitation.",
            status_code=status.HTTP_403_FORBIDDEN,
            code="INVITATION_FORBIDDEN",
        )

    if invite.status != "pending":
         raise AppError(
            message=f"Invitation is already {invite.status}.",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="INVITATION_INVALID_STATUS",
        )

    # 3. Update Status
    invite.status = "rejected"
    invite.responded_at = datetime.utcnow()
    
    await db.commit()

    return RejectInviteResponse(
        message="Invitation rejected."
    )
