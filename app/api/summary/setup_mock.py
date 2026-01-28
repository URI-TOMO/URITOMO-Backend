import uuid
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.models.room import Room, RoomMember
from app.models.message import ChatMessage

router = APIRouter()

@router.post("/debug/summary/setup-mock", tags=["debug"])
async def setup_summary_mock_data(
    room_id: str = Query("mock-room-summary", description="Room ID to setup mock summary data for"),
    user_id: str = Query("test-user-id", description="Developer user ID"),
    db: AsyncSession = Depends(get_db)
):
    """
    要約画面テスト用のモックデータを作成します。
    """
    room = await db.get(Room, room_id)
    if not room:
        room = Room(
            id=room_id,
            title="Sprint Planning Meeting",
            created_by=user_id,
            status="active",
            created_at=datetime.utcnow() - timedelta(hours=1)
        )
        db.add(room)

    member_names = ["Alice", "Bob", "Charlie"]
    for i, name in enumerate(member_names):
        m_id = f"member_{room_id}_{i}"
        member = await db.get(RoomMember, m_id)
        if not member:
            member = RoomMember(
                id=m_id,
                room_id=room_id,
                display_name=name,
                role="member"
            )
            db.add(member)

    msg_exists = (await db.execute(select(ChatMessage).where(ChatMessage.room_id == room_id))).scalars().first()
    if not msg_exists:
        logs = [
            ("Alice", "Hello everyone, let's start the meeting."),
            ("Bob", "I've finished the frontend design."),
            ("Charlie", "Great, I will start the backend implementation today.")
        ]
        for i, (name, text) in enumerate(logs):
            msg = ChatMessage(
                id=str(uuid.uuid4()),
                room_id=room_id,
                seq=i + 1,
                sender_type="participant",
                sender_member_id=f"member_{room_id}_{i % 3}",
                message_type="stt",
                text=text
            )
            db.add(msg)

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup summary mock data: {str(e)}"
        )

    return {
        "status": "success",
        "room_id": room_id,
        "message": "Mock summary data created successfully"
    }
