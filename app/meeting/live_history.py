from typing import Optional
from fastapi import APIRouter, Query
from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.deps import SessionDep
from app.core.token import CurrentUserDep
from app.core.errors import AppError
from app.models.room import RoomLiveSession, RoomMember
from app.models.message import ChatMessage
from app.meeting.schemas import SuccessResponse

router = APIRouter(prefix="/meeting", tags=["meetings"])

@router.get("/{session_id}/messages", response_model=SuccessResponse)
async def get_session_messages(
    session_id: str,
    current_user_id: CurrentUserDep,
    session: SessionDep,
    limit: int = Query(50, ge=1, le=100),
    before_seq: Optional[int] = Query(None, description="Fetch messages strictly before this sequence number")
):
    """
    Fetch chat history for a live session.
    Verifies that the user is a member of the room associated with the session.
    """
    try:
        # 1. Get Session and Room ID
        result = await session.execute(
            select(RoomLiveSession).where(RoomLiveSession.id == session_id)
        )
        live_session = result.scalar_one_or_none()
        
        if not live_session:
             raise AppError(status_code=404, code="40402", message="Live session not found")
        
        room_id = live_session.room_id
        
        # 2. Verify Membership
        member_result = await session.execute(
            select(RoomMember).where(
                RoomMember.room_id == room_id,
                RoomMember.user_id == current_user_id
            )
        )
        member = member_result.scalar_one_or_none()
        
        if not member:
             raise AppError(status_code=403, code="40301", message="Access denied: Not a room member")

        # 3. Query Messages
        # We need to join with RoomMember to get display_name
        query = (
            select(ChatMessage)
            .options(joinedload(ChatMessage.sender_member))
            .where(ChatMessage.room_id == room_id)
            .order_by(ChatMessage.seq.desc())
            .limit(limit)
        )
        
        if before_seq is not None:
            query = query.where(ChatMessage.seq < before_seq)
            
        messages_result = await session.execute(query)
        messages = messages_result.scalars().all()
        
        # 4. Format Response (Reverse to chronological order for client)
        # Client likely wants oldest -> newest. But we queried desc to get *latest*.
        # So we reverse the list in python.
        
        formatted_messages = []
        for msg in reversed(messages):
            sender_name = "Unknown"
            if msg.sender_member:
                sender_name = msg.sender_member.display_name
            elif msg.sender_type == "ai":
                sender_name = "AI Assistant"
            elif msg.sender_type == "system":
                sender_name = "System"

            formatted_messages.append({
                "id": msg.id,
                "room_id": msg.room_id,
                "seq": msg.seq,
                "sender_member_id": msg.sender_member_id,
                "display_name": sender_name,
                "text": msg.text,
                "lang": msg.lang,
                "created_at": msg.created_at
            })

        return SuccessResponse(
            status="success",
            data={
                "messages": formatted_messages,
                "total": len(formatted_messages)
            }
        )

    except AppError:
        raise
    except Exception as e:
        print(f"Error fetching messages: {e}")
        raise AppError(status_code=500, code="50001", message="Internal server error")
