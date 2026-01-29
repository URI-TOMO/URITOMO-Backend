from uuid import uuid4
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Path
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_
from pydantic import BaseModel, EmailStr

from app.infra.db import get_db
from app.models.user import User
from app.models.friend import UserFriend
from app.core.token import get_current_user_id

router = APIRouter(tags=["friend"])

# --- Schemas ---

class SendFriendRequestPayload(BaseModel):
    email: EmailStr

class SendFriendRequestResponse(BaseModel):
    message: str
    request_id: str
    status: str

class FriendRequestSender(BaseModel):
    id: str
    name: str # display_name
    email: EmailStr
    avatar: Optional[str] = None

class FriendRequest(BaseModel):
    request_id: str
    sender: FriendRequestSender
    status: str
    created_at: datetime

class FriendData(BaseModel):
    id: str
    name: str
    email: EmailStr
    status: str

class AcceptFriendRequestResponse(BaseModel):
    message: str
    friend: FriendData

class RejectFriendRequestResponse(BaseModel):
    message: str

# --- Endpoints ---

@router.post("/user/friend/request", response_model=SendFriendRequestResponse, status_code=status.HTTP_200_OK)
async def send_friend_request(
    payload: SendFriendRequestPayload,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    # 1. Fetch target user
    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    target_user = result.scalar_one_or_none()
    
    if not target_user:
        raise HTTPException(status_code=404, detail="User with this email not found.")
        
    if target_user.id == user_id:
        raise HTTPException(status_code=400, detail="You cannot send a friend request to yourself.")

    # 2. Check existing relationship (pending or accepted)
    # We check if ANY relationship exists that blocks a new request.
    # We should look for:
    # - (A, B) or (B, A) is 'accepted' -> Already friends
    # - (A, B) is 'pending' -> Request already sent by me
    # - (B, A) is 'pending' -> Request already sent by them (should accept that instead?)
    
    stmt_exist = select(UserFriend).where(
        or_(
            and_(UserFriend.requester_id == user_id, UserFriend.addressee_id == target_user.id),
            and_(UserFriend.requester_id == target_user.id, UserFriend.addressee_id == user_id)
        ),
        UserFriend.status.in_(["pending", "accepted"])
    )
    result_exist = await db.execute(stmt_exist)
    existing = result_exist.scalar_one_or_none()
    
    if existing:
        if existing.status == "accepted":
             raise HTTPException(status_code=409, detail="You are already friends.")
        # If pending
        if existing.requester_id == user_id:
             raise HTTPException(status_code=409, detail="You have already sent a friend request.")
        else:
             raise HTTPException(status_code=409, detail="This user has already sent you a friend request. Please check your inbox.")

    # 3. Check for rejected/ended/blocked rows to reuse or create new
    # The unique constraint is (requester_id, addressee_id).
    # If I sent a request before and it was rejected, I have a row (me, target, rejected).
    # I can reuse this row.
    
    stmt_my_req = select(UserFriend).where(
        UserFriend.requester_id == user_id,
        UserFriend.addressee_id == target_user.id
    )
    result_my_req = await db.execute(stmt_my_req)
    my_req_row = result_my_req.scalar_one_or_none()
    
    if my_req_row:
        # Reuse existing row
        my_req_row.status = "pending"
        my_req_row.requested_at = datetime.utcnow()
        my_req_row.responded_at = None
        my_req_row.ended_at = None
        request_obj = my_req_row
    else:
        # Create new
        request_obj = UserFriend(
            id=str(uuid4()),
            requester_id=user_id,
            addressee_id=target_user.id,
            status="pending",
            requested_at=datetime.utcnow()
        )
        db.add(request_obj)

    await db.commit()
    await db.refresh(request_obj)
    
    return SendFriendRequestResponse(
        message="Friend request sent successfully.",
        request_id=request_obj.id,
        status=request_obj.status
    )

@router.get("/user/friend/requests/received", response_model=List[FriendRequest])
async def get_received_friend_requests(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = (
        select(UserFriend, User)
        .join(User, UserFriend.requester_id == User.id)
        .where(
            UserFriend.addressee_id == user_id,
            UserFriend.status == "pending"
        )
    )
    
    result = await db.execute(stmt)
    rows = result.all()
    
    response = []
    for f_row, sender_user in rows:
        response.append(FriendRequest(
            request_id=f_row.id,
            sender=FriendRequestSender(
                id=sender_user.id,
                name=sender_user.display_name,
                email=sender_user.email,
                avatar=None # User model currently has no avatar/profile_image field
            ),
            status=f_row.status,
            created_at=f_row.requested_at
        ))
        
    return response

@router.post("/user/friend/request/{request_id}/accept", response_model=AcceptFriendRequestResponse)
async def accept_friend_request(
    request_id: str = Path(..., description="The ID of the friend request"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = select(UserFriend).where(UserFriend.id == request_id)
    result = await db.execute(stmt)
    friend_request = result.scalar_one_or_none()
    
    if not friend_request:
        raise HTTPException(status_code=404, detail="Friend request not found.")
        
    if friend_request.addressee_id != user_id:
        raise HTTPException(status_code=403, detail="You are not authorized to accept this request.")
        
    if friend_request.status != "pending":
        raise HTTPException(status_code=400, detail=f"Request is already {friend_request.status}.")
        
    friend_request.status = "accepted"
    friend_request.responded_at = datetime.utcnow()
    
    # Send back the friend details (the requester)
    stmt_friend = select(User).where(User.id == friend_request.requester_id)
    result_friend = await db.execute(stmt_friend)
    friend_user = result_friend.scalar_one_or_none()
    
    # We need to commit to save the accepted status
    await db.commit()
    
    # Assuming friend_user logic (if user deleted? unlikely if constraint holds, but safe to check)
    if not friend_user:
         # Should not happen ideally
         raise HTTPException(status_code=500, detail="Friend user data not found.")

    return AcceptFriendRequestResponse(
        message="Friend request accepted.",
        friend=FriendData(
            id=friend_user.id,
            name=friend_user.display_name,
            email=friend_user.email,
            status="online" # Placeholder
        )
    )

@router.post("/user/friend/request/{request_id}/reject", response_model=RejectFriendRequestResponse)
async def reject_friend_request(
    request_id: str = Path(..., description="The ID of the friend request"),
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id)
):
    stmt = select(UserFriend).where(UserFriend.id == request_id)
    result = await db.execute(stmt)
    friend_request = result.scalar_one_or_none()
    
    if not friend_request:
        raise HTTPException(status_code=404, detail="Friend request not found.")
        
    if friend_request.addressee_id != user_id:
        raise HTTPException(status_code=403, detail="You are not authorized to reject this request.")
        
    if friend_request.status != "pending":
         raise HTTPException(status_code=400, detail=f"Request is already {friend_request.status}.")

    friend_request.status = "rejected"
    friend_request.responded_at = datetime.utcnow()
    
    await db.commit()
    
    return RejectFriendRequestResponse(
        message="Friend request rejected."
    )
