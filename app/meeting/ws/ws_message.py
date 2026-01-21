import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from app.infra.db import AsyncSessionLocal
from app.models.message import ChatMessage
from app.models.room import RoomMember, RoomLiveSession
from app.meeting.ws.manager import manager

async def handle_chat_message(session_id: str, user_id: str, data: dict):
    """
    Handle incoming chat message:
    1. Validate data
    2. Fetch RoomMember and latest sequence
    3. Save to DB
    4. Broadcast to all session members
    """
    text = data.get("text")
    if not text:
        return
    
    async with AsyncSessionLocal() as db_session:
        # 1. Get Session and Room ID
        session_result = await db_session.execute(
            select(RoomLiveSession).where(RoomLiveSession.id == session_id)
        )
        live_session = session_result.scalar_one_or_none()
        if not live_session:
            return
        
        room_id = live_session.room_id

        # 2. Get RoomMember ID for this user
        member_result = await db_session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == user_id
            )
        )
        member = member_result.scalar_one_or_none()
        if not member:
            return

        # 3. Get next sequence number for this room
        seq_result = await db_session.execute(
            select(func.max(ChatMessage.seq)).where(ChatMessage.room_id == room_id)
        )
        max_seq = seq_result.scalar() or 0
        next_seq = max_seq + 1

        # 4. Create ChatMessage
        message_id = f"msg_{uuid.uuid4().hex[:16]}"
        new_message = ChatMessage(
            id=message_id,
            room_id=room_id,
            seq=next_seq,
            sender_type="human",
            sender_member_id=member.id,
            message_type="text",
            text=text,
            lang=data.get("lang", "ja"), # Default or from data
            created_at=datetime.utcnow()
        )

        db_session.add(new_message)
        await db_session.commit()
        await db_session.refresh(new_message)

        # 5. Broadcast
        # We broadcast a formatted message
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
                "created_at": new_message.created_at.isoformat()
            }
        }
        await manager.broadcast(session_id, broadcast_data)
