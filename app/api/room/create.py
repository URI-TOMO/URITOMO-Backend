from fastapi import APIRouter, Depends, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime
from uuid import uuid4

from app.infra.db import get_db
from app.models.user import User
from app.models.room import Room, RoomMember
from app.core.token import get_current_user_id

router = APIRouter(tags=["room"])

class RoomCreateRequest(BaseModel):
    room_name: str

class RoomCreateResponse(BaseModel):
    room_id: str
    room_name: str
    created_at: datetime

@router.post("/room/create", response_model=RoomCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_room(
    data: RoomCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Create a new room with the given name.
    The creator becomes the owner of the room.
    """
    new_room_id = uuid4().hex
    
    # Create Room
    new_room = Room(
        id=new_room_id,
        title=data.room_name,
        created_by=current_user_id,
        status="active",
        created_at=datetime.utcnow()
    )

    # Create RoomMember for the creator
    new_member = RoomMember(
        id=f"member_{uuid4().hex}",
        room_id=new_room_id,
        user_id=current_user_id,
        display_name="Owner",  # Default, updated below if user found
        role="owner",
        joined_at=datetime.utcnow()
    )

    # Fetch user to get correct display name
    user_stmt = select(User).where(User.id == current_user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    
    if user:
         new_member.display_name = user.display_name
    
    db.add(new_room)
    db.add(new_member)
    await db.commit()

    return RoomCreateResponse(
        room_id=new_room.id,
        room_name=new_room.title,
        created_at=new_room.created_at
    )
