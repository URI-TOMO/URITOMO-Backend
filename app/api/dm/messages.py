from datetime import datetime
from typing import List, Optional
import uuid
import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy import func, select, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.core.token import CurrentUserDep
from app.core.time import to_jst_iso
from app.infra.db import get_db
from app.models.dm import DmMessage, DmParticipant, DmThread
from app.models.user import User
from app.meeting.ws.manager import manager

router = APIRouter(tags=["dm"])
logger = logging.getLogger("uritomo.dm")

# --- Schemas ---
class DmMessageResponse(BaseModel):
    id: str
    thread_id: str
    seq: int
    sender_type: str
    sender_user_id: Optional[str]
    display_name: Optional[str] = None # Calculated field
    text: str
    lang: Optional[str] = None
    created_at: datetime # or str

class DmMessagesListResponse(BaseModel):
    messages: List[DmMessageResponse]

class SendDmMessagePayload(BaseModel):
    text: str

class SendDmMessageResponse(BaseModel):
    message: DmMessageResponse

# --- Helpers ---
async def get_next_seq(thread_id: str, session: AsyncSession) -> int:
    stmt = select(func.max(DmMessage.seq)).where(DmMessage.thread_id == thread_id)
    result = await session.execute(stmt)
    max_seq = result.scalar()
    return (max_seq or 0) + 1

async def check_participation(thread_id: str, user_id: str, session: AsyncSession):
    stmt = select(DmParticipant).where(
        DmParticipant.thread_id == thread_id,
        DmParticipant.user_id == user_id
    )
    result = await session.execute(stmt)
    part = result.scalars().first()
    if not part:
        raise HTTPException(status_code=403, detail="Not a participant of this thread")
    return part

# --- Endpoints ---

@router.get("/dm/{thread_id}/messages", response_model=DmMessagesListResponse)
async def get_messages(
    thread_id: str,
    current_user: CurrentUserDep,
    limit: int = 50,
    before_seq: Optional[int] = Query(None),
    session: AsyncSession = Depends(get_db)
):
    # current_user is a string (user_id), not a User object
    user_id = current_user
    print(f"📥 [DM] get_messages called | thread_id={thread_id} | user={user_id}", flush=True)
    await check_participation(thread_id, user_id, session)

    query = select(DmMessage).where(DmMessage.thread_id == thread_id)
    
    if before_seq is not None:
        query = query.where(DmMessage.seq < before_seq)
        
    query = query.order_by(desc(DmMessage.seq)).limit(limit).options(joinedload(DmMessage.sender_user))
    
    result = await session.execute(query)
    messages = result.scalars().all()
    
    # Sort by seq asc for frontend
    messages = sorted(messages, key=lambda m: m.seq)
    
    return {
        "messages": [
            {
                "id": m.id,
                "thread_id": m.thread_id,
                "seq": m.seq,
                "sender_type": m.sender_type,
                "sender_user_id": m.sender_user_id,
                "display_name": m.sender_user.display_name if m.sender_user else "Unknown",
                "text": m.text,
                "lang": m.lang,
                "created_at": to_jst_iso(m.created_at)
            }
            for m in messages
        ]
    }

@router.post("/dm/{thread_id}/messages", response_model=SendDmMessageResponse)
async def send_message(
    thread_id: str,
    payload: SendDmMessagePayload,
    current_user: CurrentUserDep,
    session: AsyncSession = Depends(get_db)
):
    # current_user is a string (user_id), not a User object
    user_id = current_user
    print(f"📤 [DM] send_message called | thread_id={thread_id} | user={user_id} | text={payload.text[:50]}", flush=True)
    await check_participation(thread_id, user_id, session)
    
    next_seq = await get_next_seq(thread_id, session)
    msg_id = f"dm_msg_{uuid.uuid4().hex[:16]}"
    
    new_message = DmMessage(
        id=msg_id,
        thread_id=thread_id,
        seq=next_seq,
        sender_type="human",
        sender_user_id=user_id,
        message_type="text",
        text=payload.text,
        created_at=datetime.utcnow()
    )
    
    session.add(new_message)
    await session.commit()
    await session.refresh(new_message)
    
    # Get display_name from User table
    user_stmt = select(User).where(User.id == user_id)
    user_result = await session.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    display_name = user.display_name if user else "Unknown"
    
    resp_data = {
        "id": new_message.id,
        "thread_id": new_message.thread_id,
        "seq": new_message.seq,
        "sender_type": new_message.sender_type,
        "sender_user_id": new_message.sender_user_id,
        "display_name": display_name,
        "text": new_message.text,
        "lang": new_message.lang,
        "created_at": to_jst_iso(new_message.created_at)
    }
    
    # Broadcast via WebSocket
    ws_data = {
        "type": "dm.chat",
        "data": resp_data
    }
    await manager.broadcast(thread_id, ws_data)
    print(f"📡 [DM] Broadcast sent | thread_id={thread_id} | message_id={msg_id}", flush=True)
    
    return {"message": resp_data}
