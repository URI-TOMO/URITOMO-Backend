import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import select, func
from app.infra.db import AsyncSessionLocal
from app.models.message import ChatMessage
from app.models.room import RoomMember, RoomLiveSession
from app.models.ai import AIEvent
from app.meeting.ws.manager import manager
from app.translation.deepl_service import deepl_service
from app.core.logging import get_logger

logger = get_logger(__name__)

async def handle_chat_message(session_id: str, user_id: str, data: dict):
    """
    Handle incoming chat message:
    1. Validate data
    2. Fetch RoomMember and latest sequence
    3. Save to DB (Original)
    4. Broadcast to all session members (Original)
    5. Perform Translation
    6. Save Translation Event
    7. Broadcast Translation
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

        # 4. Create ChatMessage (Original)
        source_lang = data.get("lang", "Korean") # Default to Korean based on user context
        
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
            created_at=datetime.utcnow()
        )

        db_session.add(new_message)
        await db_session.commit()
        await db_session.refresh(new_message)

        # 5. Broadcast Original Message
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

        # 6. Perform Translation (Sync/Async)
        # DeepL translate is synchronous in our service currently, but it's fine for now 
        # as we are in an async handler, though blocking the loop is not ideal if high load.
        # Ideally deepl_service should be async or run in executor.
        # For this implementation, we run it directly.
        
        target_lang = "Japanese" if "Korean" in source_lang else "Korean"
        
        try:
            translated_text = deepl_service.translate_text(
                text=text, 
                source_lang=source_lang, 
                target_lang=target_lang
            )
            
            # 7. Save Translation Event
            # Using specific ID format or UUID
            trans_id = f"trans_{uuid.uuid4().hex[:16]}"
            
            ai_event = AIEvent(
                id=trans_id,
                room_id=room_id,
                seq=next_seq, # Use same seq as message or new seq? 
                              # AIEvent has unique constraint on (room_id, seq). 
                              # ChatMessage also has (room_id, seq).
                              # If they share the same seq namespace, we have a problem.
                              # AIEvent and ChatMessage are different tables.
                              # If the frontend renders by sorting ALL events by seq, then we should probably increment seq.
                              # However, getting a new lock for seq is complex.
                              # Usually AI events are associated with the message or have their own sequence.
                              # Let's check the models.
                              # Message: seq is BigInteger.
                              # AIEvent: seq is BigInteger.
                              # If they are interleaved, they need a shared sequence generator or we reuse the message seq if it's 1:1.
                              # But AIEvent might not be 1:1.
                              # For now, let's assume we reuse the sequence of the message to link them 
                              # OR we just increment if we can.
                              # But since we already committed the message, fetching max_seq again might get the same or next.
                              # Let's use the SAME sequence to indicate they belong together 
                              # (if the uniqueness is per TABLE, then it's fine).
                              # Uniqueness: ChatMessage(room_id, seq) AND AIEvent(room_id, seq).
                              # So we CAN use the same seq for AIEvent as ChatMessage without DB conflict.
                event_type="translation",
                original_text=text,
                original_lang=source_lang,
                translated_text=translated_text,
                translated_lang=target_lang,
                meta={
                     "related_message_id": message_id,
                     "participant_id": user_id,
                     "participant_name": member.display_name
                },
                created_at=datetime.utcnow()
            )
            
            db_session.add(ai_event)
            await db_session.commit()
            
            # 8. Broadcast Translation
            # The structure requested by user initially:
            # {
            #     "room_id": "room_01",
            #     "participant_id": "user_xyz123",
            #     "participant_name": "user",
            #     "Original": "안녕하세요",
            #     "translated": "こんにちは",
            #     "timestamp": "2024-01-01T00:00:00.00Z",
            #     "sequence": "0"
            # }
            # We map this to our websocket message format.
            
            trans_broadcast_data = {
                "type": "translation",
                "data": {
                    "room_id": room_id,
                    "participant_id": user_id,
                    "participant_name": member.display_name,
                    "Original": text,
                    "translated": translated_text,
                    "timestamp": ai_event.created_at.isoformat(),
                    "sequence": str(next_seq),
                    "lang": target_lang
                }
            }
            
            await manager.broadcast(session_id, trans_broadcast_data)
            
        except Exception as e:
            logger.error(f"Translation failed in websocket: {e}")
            # We don't fail the chat, just skip translation broadcast

async def handle_stt_message(session_id: str, user_id: str, data: dict):
    """
    Handle incoming STT (Speech-to-Text) message:
    1. If not is_final, just broadcast to others (partial UI)
    2. If is_final:
        - Save to DB (as ChatMessage with sender_type='human' or 'transcription')
        - Translate
        - Broadcast Final and Translation
    """
    text = data.get("text")
    is_final = data.get("is_final", False)
    if not text:
        return

    # If it's just a partial result, broadcast and return
    if not is_final:
        broadcast_data = {
            "type": "stt",
            "data": {
                "user_id": user_id,
                "text": text,
                "is_final": False,
                "lang": data.get("lang", "Korean")
            }
        }
        await manager.broadcast(session_id, broadcast_data)
        return

    # If it's final, process like a chat message (save & translate)
    async with AsyncSessionLocal() as db_session:
        # 1. Get Session and Room ID
        session_result = await db_session.execute(
            select(RoomLiveSession).where(RoomLiveSession.id == session_id)
        )
        live_session = session_result.scalar_one_or_none()
        if not live_session:
            return
        
        room_id = live_session.room_id

        # 2. Get RoomMember ID
        member_result = await db_session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == user_id
            )
        )
        member = member_result.scalar_one_or_none()
        if not member:
            return

        # 3. Get next sequence
        seq_result = await db_session.execute(
            select(func.max(ChatMessage.seq)).where(ChatMessage.room_id == room_id)
        )
        max_seq = seq_result.scalar() or 0
        next_seq = max_seq + 1

        # 4. Create ChatMessage (Final STT)
        source_lang = data.get("lang", "Korean")
        message_id = f"stt_{uuid.uuid4().hex[:16]}"
        
        new_message = ChatMessage(
            id=message_id,
            room_id=room_id,
            seq=next_seq,
            sender_type="transcription", # Mark as transcription
            sender_member_id=member.id,
            message_type="text",
            text=text,
            lang=source_lang,
            created_at=datetime.utcnow()
        )

        db_session.add(new_message)
        await db_session.commit()
        await db_session.refresh(new_message)

        # 5. Broadcast Final STT
        broadcast_data = {
            "type": "stt",
            "data": {
                "id": new_message.id,
                "room_id": new_message.room_id,
                "seq": new_message.seq,
                "user_id": user_id,
                "display_name": member.display_name,
                "text": new_message.text,
                "lang": new_message.lang,
                "is_final": True,
                "created_at": new_message.created_at.isoformat()
            }
        }
        await manager.broadcast(session_id, broadcast_data)

        # 6. Translate
        target_lang = "Japanese" if "Korean" in source_lang else "Korean"
        try:
            translated_text = deepl_service.translate_text(
                text=text, 
                source_lang=source_lang, 
                target_lang=target_lang
            )
            
            # 7. Save Translation
            trans_id = f"stt_trans_{uuid.uuid4().hex[:16]}"
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
                     "participant_id": user_id,
                     "participant_name": member.display_name,
                     "is_stt": True
                },
                created_at=datetime.utcnow()
            )
            
            db_session.add(ai_event)
            await db_session.commit()
            
            # 8. Broadcast Translation
            trans_broadcast_data = {
                "type": "translation",
                "data": {
                    "room_id": room_id,
                    "participant_id": user_id,
                    "participant_name": member.display_name,
                    "Original": text,
                    "translated": translated_text,
                    "timestamp": ai_event.created_at.isoformat(),
                    "sequence": str(next_seq),
                    "lang": target_lang,
                    "is_stt": True
                }
            }
            await manager.broadcast(session_id, trans_broadcast_data)
        except Exception as e:
            logger.error(f"Translation failed for STT: {e}")
