"""
Room Chat Messages API
- GET /rooms/{room_id}/messages - Fetch chat history for a room
- POST /rooms/{room_id}/messages - Send a chat message to a room (REST, not WebSocket)
"""

from datetime import datetime
from typing import Optional, List
import uuid

from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import joinedload
from pydantic import BaseModel

import logging

from app.infra.db import get_db
from app.models.room import Room, RoomMember
from app.models.message import ChatMessage
from app.models.ai import AIEvent
from app.core.token import CurrentUserDep
from app.core.errors import AppError
from app.core.time import to_jst_iso
from app.translation.deepl_service import deepl_service
from app.translation.openai_service import openai_service
from app.meeting.ws.manager import manager

logger = logging.getLogger("uritomo.ws")

router = APIRouter(tags=["room-chat"])


# ============ Schemas ============

class MessageResponse(BaseModel):
    id: str
    room_id: str
    seq: int
    sender_member_id: Optional[str]
    display_name: str
    text: str
    lang: Optional[str]
    translated_text: Optional[str]
    translated_lang: Optional[str]
    created_at: str


class MessagesListResponse(BaseModel):
    messages: List[MessageResponse]
    total: int
    has_more: bool


class SendMessageRequest(BaseModel):
    text: str
    lang: Optional[str] = "Korean"


class SendMessageResponse(BaseModel):
    id: str
    room_id: str
    seq: int
    text: str
    lang: str
    translated_text: Optional[str]
    translated_lang: Optional[str]
    created_at: str


# ============ Helper Functions ============

def _normalize_lang(lang: Optional[str]) -> str:
    if not lang:
        return "Korean"
    lang_lower = lang.strip().lower()
    if lang_lower in {"ko", "kr", "korean"}:
        return "Korean"
    if lang_lower in {"ja", "jp", "japanese"}:
        return "Japanese"
    return lang


def _looks_like_mock(text: Optional[str]) -> bool:
    if not text:
        return True
    return (
        text.startswith("[KO]")
        or text.startswith("[JA]")
        or text.startswith("[TRANS]")
        or text.startswith("[Korean]")
        or text.startswith("[Japanese]")
    )


async def _translate_with_fallback(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    translated_text: Optional[str] = None
    if deepl_service.enabled:
        try:
            translated_text = deepl_service.translate_text(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        except Exception:
            translated_text = None
    if _looks_like_mock(translated_text):
        try:
            translated_text = await openai_service.translate_text(
                text=text,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        except Exception:
            pass
    if _looks_like_mock(translated_text):
        return None
    return translated_text


# ============ Endpoints ============

@router.get("/rooms/{room_id}/messages", response_model=MessagesListResponse)
async def get_room_messages(
    room_id: str,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
    limit: int = Query(50, ge=1, le=100, description="Number of messages to return"),
    before_seq: Optional[int] = Query(None, description="Fetch messages before this sequence number for pagination"),
):
    """
    Get chat history for a room.
    
    Supports pagination via `before_seq` parameter.
    Messages are returned in chronological order (oldest first within the batch).
    """
    logger.info(f"📥 REST Get Messages | Room: {room_id} | User: {current_user_id} | Limit: {limit} | Before: {before_seq}")
    
    # 1. Check Room existence
    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalar_one_or_none()
    
    if not room:
        raise AppError(
            message="Room not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="ROOM_NOT_FOUND",
        )
    
    # 2. Verify Membership
    member_result = await db.execute(
        select(RoomMember).where(
            RoomMember.room_id == room_id,
            RoomMember.user_id == current_user_id,
            RoomMember.left_at.is_(None),
        )
    )
    member = member_result.scalar_one_or_none()
    
    if not member:
        raise AppError(
            message="Access denied: You are not a member of this room",
            status_code=status.HTTP_403_FORBIDDEN,
            code="ROOM_MEMBER_FORBIDDEN",
        )
    
    # 3. Build Query
    query = (
        select(ChatMessage)
        .options(joinedload(ChatMessage.sender_member))
        .where(ChatMessage.room_id == room_id)
        .order_by(ChatMessage.seq.desc())
        .limit(limit + 1)  # Fetch one extra to check if there are more
    )
    
    if before_seq is not None:
        query = query.where(ChatMessage.seq < before_seq)
    
    # 4. Execute Query
    messages_result = await db.execute(query)
    messages = messages_result.scalars().all()
    
    # Check if there are more messages
    has_more = len(messages) > limit
    if has_more:
        messages = messages[:-1]  # Remove the extra one
    
    # 5. Format Response (reverse to chronological order)
    formatted_messages = []
    for msg in reversed(messages):
        sender_name = "Unknown"
        if msg.sender_member:
            sender_name = msg.sender_member.display_name
        elif msg.sender_type == "ai":
            sender_name = "AI Assistant"
        elif msg.sender_type == "system":
            sender_name = "System"
        
        formatted_messages.append(
            MessageResponse(
                id=msg.id,
                room_id=msg.room_id,
                seq=msg.seq,
                sender_member_id=msg.sender_member_id,
                display_name=sender_name,
                text=msg.text,
                lang=msg.lang,
                translated_text=msg.translated_text,
                translated_lang=msg.translated_lang,
                created_at=to_jst_iso(msg.created_at),
            )
        )
    
    logger.info(f"✅ REST Get Messages Complete | Room: {room_id} | Count: {len(formatted_messages)} | HasMore: {has_more}")
    
    return MessagesListResponse(
        messages=formatted_messages,
        total=len(formatted_messages),
        has_more=has_more,
    )


@router.post("/rooms/{room_id}/messages", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_room_message(
    room_id: str,
    payload: SendMessageRequest,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
):
    """
    Send a chat message to a room via REST API.
    
    This endpoint saves the message, translates it, and broadcasts to WebSocket clients.
    Useful for sending messages without maintaining a WebSocket connection.
    """
    logger.info(f"📤 REST Send Message | Room: {room_id} | User: {current_user_id} | Text: {payload.text[:50]}")
    
    text = payload.text.strip()
    if not text:
        raise AppError(
            message="Message text cannot be empty",
            status_code=status.HTTP_400_BAD_REQUEST,
            code="EMPTY_MESSAGE",
        )
    
    # 1. Check Room existence
    room_result = await db.execute(select(Room).where(Room.id == room_id))
    room = room_result.scalar_one_or_none()
    
    if not room:
        raise AppError(
            message="Room not found",
            status_code=status.HTTP_404_NOT_FOUND,
            code="ROOM_NOT_FOUND",
        )
    
    # 2. Verify Membership and get member info
    member_result = await db.execute(
        select(RoomMember).where(
            RoomMember.room_id == room_id,
            RoomMember.user_id == current_user_id,
            RoomMember.left_at.is_(None),
        )
    )
    member = member_result.scalar_one_or_none()
    
    if not member:
        raise AppError(
            message="Access denied: You are not a member of this room",
            status_code=status.HTTP_403_FORBIDDEN,
            code="ROOM_MEMBER_FORBIDDEN",
        )
    
    # 3. Get next sequence number
    seq_result = await db.execute(
        select(func.max(ChatMessage.seq)).where(ChatMessage.room_id == room_id)
    )
    max_seq = seq_result.scalar() or 0
    next_seq = max_seq + 1
    
    # 4. Create ChatMessage
    source_lang = _normalize_lang(payload.lang)
    message_id = f"msg_{uuid.uuid4().hex[:16]}"
    
    new_message = ChatMessage(
        id=message_id,
        room_id=room_id,
        seq=next_seq,
        sender_type="human",
        sender_member_id=member.id,
        message_type="text",
        text=text,
        lang=source_lang,
        translated_text=None,
        translated_lang=None,
        created_at=datetime.utcnow(),
    )
    
    db.add(new_message)
    await db.commit()
    await db.refresh(new_message)
    
    logger.info(f"💾 REST Message Saved | ID: {message_id} | Room: {room_id} | Seq: {next_seq}")
    
    # 5. Translate
    target_lang = "Japanese" if source_lang == "Korean" else "Korean"
    translated_text: Optional[str] = None
    
    try:
        translated_text = await _translate_with_fallback(
            text=text,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        
        if translated_text:
            # Save translation to AIEvent
            trans_id = f"trans_{uuid.uuid4().hex[:16]}"
            ai_event = AIEvent(
                id=trans_id,
                room_id=room_id,
                seq=next_seq,
                event_type="translation",
                original_text=text,
                original_lang=source_lang,
                translated_text=translated_text,
                translated_lang=target_lang,
                meta={
                    "related_message_id": message_id,
                    "participant_id": current_user_id,
                    "participant_name": member.display_name,
                },
                created_at=datetime.utcnow(),
            )
            
            new_message.translated_text = translated_text
            new_message.translated_lang = target_lang
            db.add(ai_event)
            await db.commit()
    except Exception as e:
        logger.error(f"❌ REST Translation Failed | Room: {room_id} | Error: {e}")
    
    # 6. Broadcast to WebSocket clients
    broadcast_data = {
        "type": "chat",
        "data": {
            "id": new_message.id,
            "room_id": new_message.room_id,
            "seq": new_message.seq,
            "sender_member_id": new_message.sender_member_id,
            "display_name": member.display_name,
            "text": new_message.text,
            "lang": new_message.lang,
            "translated_text": new_message.translated_text,
            "translated_lang": new_message.translated_lang,
            "created_at": to_jst_iso(new_message.created_at),
        }
    }
    logger.info(f"📡 REST Broadcast | Room: {room_id} | Message ID: {new_message.id}")
    await manager.broadcast(room_id, broadcast_data)
    
    logger.info(f"✅ REST Send Message Complete | Room: {room_id} | Message ID: {new_message.id}")
    
    return SendMessageResponse(
        id=new_message.id,
        room_id=new_message.room_id,
        seq=new_message.seq,
        text=new_message.text,
        lang=new_message.lang,
        translated_text=new_message.translated_text,
        translated_lang=new_message.translated_lang,
        created_at=to_jst_iso(new_message.created_at),
    )
