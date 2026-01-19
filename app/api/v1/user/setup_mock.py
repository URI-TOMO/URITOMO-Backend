import uuid
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.models.user import User
from app.models.friend import UserFriend
from app.models.room import Room, RoomMember

router = APIRouter()

@router.post("/setup-mock", tags=["debug"])
async def setup_mock_data(
    user_id: str = Query(..., description="User ID to set up mock data for"),
    db: AsyncSession = Depends(get_db)
):
    """
    Sets up mock data for a given user_id:
    - Creates the user if they don't exist
    - Adds 2 friends (already accepted)
    - Adds 2 active rooms the user is a member of
    """
    # 1. Ensure main user exists
    user_stmt = select(User).where(User.id == user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    
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
            email=f"friend_{i}@example.com",
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

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to setup mock data: {str(e)}")

    return {
        "status": "success",
        "message": f"Setup mock data for user {user_id}",
        "user_id": user_id
    }
