from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.infra.db import get_db
from app.core.token import CurrentUserDep
from app.models import User, Room, RoomMember, RoomLiveSession

router = APIRouter(tags=["debug"])

@router.post("/setup-session", status_code=status.HTTP_201_CREATED)
async def setup_debug_session(
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db)
):
    """
    **Debug Session Setup**
    
    1. Uses the token to identify the current user.
    2. Finds an existing room for the user OR creates a new one.
    3. Starts a new active live session in that room.
    4. Returns the user_id, room_id, and session_id.
    
    Useful for quickly jumping into a live session for testing.
    """
    # 1. Get User
    user = await db.get(User, current_user_id)
    if not user:
        return {"error": "User not found in database. Please signup first."}

    # 2. Get or Create a Room
    room_stmt = select(Room).join(RoomMember, Room.id == RoomMember.room_id).where(RoomMember.user_id == current_user_id).limit(1)
    result = await db.execute(room_stmt)
    room = result.scalar_one_or_none()

    if not room:
        room_id = str(uuid4())
        room = Room(
            id=room_id,
            title=f"Debug Room for {user.display_name}",
            created_by=current_user_id,
            status="active",
            created_at=datetime.utcnow()
        )
        db.add(room)
        await db.flush()
        
        member = RoomMember(
            id=str(uuid4()),
            room_id=room.id,
            user_id=current_user_id,
            display_name=user.display_name,
            role="owner",
            joined_at=datetime.utcnow()
        )
        db.add(member)
        await db.flush()
    
    # 3. Create Live Session
    session_id = f"ls_{uuid4().hex[:16]}"
    live_session = RoomLiveSession(
        id=session_id,
        room_id=room.id,
        title=f"Debug Session - {user.display_name}",
        status="active",
        started_by=current_user_id,
        started_at=datetime.utcnow()
    )
    db.add(live_session)
    
    # Optional: We could also automatically join the user to this session as a member record
    # (Optional, but helps with some logic)
    
    await db.commit()

    return {
        "message": "Debug session setup successful",
        "user_id": current_user_id,
        "room_id": room.id,
        "session_id": session_id
    }
