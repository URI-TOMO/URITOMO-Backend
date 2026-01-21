from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import Room, RoomMember

router = APIRouter()

# ============ Schemas ============

class SummarizationContent(BaseModel):
    main_point: str
    task: str
    decided: str

class SummarizationData(BaseModel):
    summarization: SummarizationContent
    meeting_date: str
    past_time: str
    meeting_member: int

class SummarizationResponse(BaseModel):
    summary: SummarizationData

# ============ Endpoints ============

@router.post("/summarization/{room_id}", response_model=SummarizationResponse, tags=["summary"])
async def get_summarization(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の要約（メインポイント、タスク、決定事項）を取得します。
    """
    room_stmt = select(Room).where(Room.id == room_id)
    room_result = await db.execute(room_stmt)
    room = room_result.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Meeting room not found")

    member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
    member_result = await db.execute(member_stmt)
    member_count = len(member_result.scalars().all())

    meeting_date_str = room.created_at.strftime("%Y-%m-%d")

    # モックデータを使用してレスポンスを構成
    return SummarizationResponse(
        summary=SummarizationData(
            summarization=SummarizationContent(
                main_point="Discussion on project architecture and timelines.",
                task="Complete API implementation by Friday.",
                decided="Use FastAPI for backend and PostgreSQL for DB."
            ),
            meeting_date=meeting_date_str,
            past_time="60 min",
            meeting_member=member_count
        )
    )
