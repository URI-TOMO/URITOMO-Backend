from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import RoomMember

router = APIRouter()

# ============ Schemas ============

class MemberInfo(BaseModel):
    count: int
    names: List[str]

class MemberResponse(BaseModel):
    meeting_member: MemberInfo

# ============ Endpoints ============

@router.post("/meeting_member/{room_id}", response_model=MemberResponse, tags=["summary"])
async def get_meeting_members(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の参加者詳細情報を取得します。
    """
    member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
    member_result = await db.execute(member_stmt)
    members = member_result.scalars().all()
    member_names = [m.display_name for m in members]

    return MemberResponse(
        meeting_member=MemberInfo(
            count=len(members),
            names=member_names
        )
    )
