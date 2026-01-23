import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from app.core.errors import AppError
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.deps import SessionDep
from app.core.token import CurrentUserDep
from app.models.room import Room, RoomLiveSession, RoomMember
from app.models.user import User
from app.models.message import ChatMessage
from fastapi import Query

router = APIRouter(prefix="/meeting", tags=["meeting"])



class LiveSessionSchema(BaseModel):
    id: str
    room_id: str
    title: str
    status: str
    started_by: str
    started_at: datetime
    ended_at: Optional[datetime]

class SuccessResponse(BaseModel):
    status: str
    data: dict

@router.post("/{room_id}/live-sessions", response_model=SuccessResponse)
async def start_live_session(
    room_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep
):
    try:
        # 1. Fetch User (for display_name)
        # We know user exists because of CurrentUserDep, but we need the object.
        user_result = await session.execute(select(User).where(User.id == current_user_id))
        user = user_result.scalar_one_or_none()
        
        if not user:
             # Should practically not happen if token is valid
             # Should practically not happen if token is valid
             raise AppError(status_code=401, code="40102", message="Unauthorized")

        # 2. Check Room and Membership
        # We need to verify if the room exists AND if the user is a member.
        # We can do this in one query or separate.
        
        # Check Room
        room_result = await session.execute(select(Room).where(Room.id == room_id))
        room = room_result.scalar_one_or_none()
        
        if not room:
             raise AppError(status_code=404, code="40401", message="Room not found or not a member")

        # Check Membership
        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
             raise AppError(status_code=404, code="40401", message="Room not found or not a member")

        # 3. Create Live Session
        session_id = f"ls_{uuid.uuid4().hex[:16]}" # Example format, matching ls_001 style broadly or just uuid
        # User example: ls_001. I'll use a standard ID generation or just uuid.
        
        session_title = user.display_name
        
        new_session = RoomLiveSession(
            id=session_id,
            room_id=room_id,
            title=session_title,
            status="active",
            started_by=current_user_id,
            started_at=datetime.utcnow(),
            ended_at=None
        )
        
        session.add(new_session)
        await session.commit()
        await session.refresh(new_session)
        
        return SuccessResponse(
            status="success",
            data={
                "session": {
                    "id": new_session.id,
                    "room_id": new_session.room_id,
                    "title": new_session.title,
                    "status": new_session.status,
                    "started_by": new_session.started_by,
                    "started_at": new_session.started_at,
                    "ended_at": new_session.ended_at
                }
            }
        )

    except AppError:
        raise
    except Exception as e:
        # Log generic error?
        print(f"Error starting live session: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")


