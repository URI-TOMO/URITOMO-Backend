from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import Room, RoomMember

router = APIRouter()

# ============ Schemas ============

class DocumentInfo(BaseModel):
    meeting_date: str
    past_time: str
    meeting_member: str
    meeting_name: str

class SummaryResponse(BaseModel):
    documents: DocumentInfo

# ============ Endpoints ============

@router.post("/summary/{room_id}", response_model=SummaryResponse, tags=["summary"])
async def get_room_summary(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の要約データを取得します。
    """
    room_stmt = select(Room).where(Room.id == room_id)
    room_result = await db.execute(room_stmt)
    room = room_result.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Meeting room not found")

    member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
    member_result = await db.execute(member_stmt)
    members = member_result.scalars().all()
    member_names = [m.display_name for m in members]
    
    meeting_date_str = room.created_at.strftime("%Y-%m-%d")

    return SummaryResponse(
        documents=DocumentInfo(
            meeting_date=meeting_date_str,
            past_time="60 min",
            meeting_member=", ".join(member_names),
            meeting_name=room.title or "Untitled Meeting"
        )
    )
