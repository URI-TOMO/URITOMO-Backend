from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from pydantic import BaseModel, EmailStr
from uuid import uuid4
from datetime import datetime
from typing import Optional

from app.infra.db import get_db
from app.models.user import User
from app.models.friend import UserFriend
from app.core.token import get_current_user_id

router = APIRouter(tags=["friends"])

# ============ Schemas ============

class FriendAddRequest(BaseModel):
    email: EmailStr

class FriendAddResponse(BaseModel):
    name: str
    email: str
    lang: Optional[str]

# ============ Endpoints ============

@router.post("/user/friend/add", response_model=FriendAddResponse)
async def add_friend_by_email(
    data: FriendAddRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Search for a user by email and add them as a friend immediately.
    """
    # 1. Search for the user by email
    stmt = select(User).where(User.email == data.email)
    result = await db.execute(stmt)
    target_user = result.scalar_one_or_none()

    if not target_user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    target_user_id = target_user.id

    # 2. Guard: Cannot add self
    if current_user_id == target_user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="You cannot add yourself as a friend"
        )

    # 3. Guard: Check if a relationship already exists
    friend_stmt = select(UserFriend).where(
        or_(
            (UserFriend.requester_id == current_user_id) & (UserFriend.addressee_id == target_user_id),
            (UserFriend.requester_id == target_user_id) & (UserFriend.addressee_id == current_user_id)
        )
    )
    friend_result = await db.execute(friend_stmt)
    existing_relationship = friend_result.scalar_one_or_none()

    if existing_relationship:
        if existing_relationship.status == "accepted":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You are already friends"
            )
        elif existing_relationship.status == "pending":
            # If there's a pending request, we might want to just accept it? 
            # Or currently just error out as requested to 'add' via email search.
            # Assuming simple logic for now: error if already pending.
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A friend request is already pending"
            )
        elif existing_relationship.status == "blocked":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot add this user"
            )

    # 4. create new friend entry (Accepted immediately)
    new_friendship = UserFriend(
        id=str(uuid4()),
        requester_id=current_user_id,
        addressee_id=target_user_id,
        status="accepted",
        requested_at=datetime.utcnow(),
        responded_at=datetime.utcnow()
    )

    db.add(new_friendship)
    await db.commit()

    return FriendAddResponse(
        name=target_user.display_name,
        email=target_user.email,
        lang=target_user.locale
    )
