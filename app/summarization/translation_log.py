from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import RoomMember
from app.models.message import ChatMessage

router = APIRouter()

# ============ Schemas ============

class TranslationLogItem(BaseModel):
    id: str
    timestamp: str
    sender_name: str
    text: str
    original_text: Optional[str] = None

class TranslationLogResponse(BaseModel):
    translation_log: List[TranslationLogItem]

# ============ Endpoints ============

@router.post("/translation_log/{room_id}", response_model=TranslationLogResponse, tags=["summary"])
async def get_translation_log(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の翻訳ログ（チャット履歴）を取得します。
    """
    # メンバー情報を取得（送信者名表示用）
    member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
    member_result = await db.execute(member_stmt)
    members = member_result.scalars().all()
    member_map = {m.id: m.display_name for m in members}

    # メッセージ履歴を取得
    msg_stmt = (
        select(ChatMessage)
        .where(ChatMessage.room_id == room_id)
        .order_by(ChatMessage.seq.asc())
    )
    msg_result = await db.execute(msg_stmt)
    messages = msg_result.scalars().all()

    log_data = [
        TranslationLogItem(
            id=msg.id,
            timestamp=msg.created_at.strftime("%H:%M:%S"),
            sender_name=member_map.get(msg.sender_member_id, "Unknown"),
            text=msg.text
        ) for msg in messages
    ]

    return TranslationLogResponse(translation_log=log_data)
