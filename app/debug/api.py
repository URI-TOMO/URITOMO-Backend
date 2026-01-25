from datetime import datetime, timedelta
import random
import hashlib
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, status, Query
from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.infra.db import get_db
from app.models import (
    User, Room, RoomMember, ChatMessage, Live, AIEvent,
    UserFriend, DmThread, DmParticipant, DmMessage,
    RoomLiveSession, RoomLiveSessionMember, AuthToken
)

from app.debug.for_live import router as for_live_router
from app.debug.login import router as login_router

router = APIRouter(tags=["debug"])
router.include_router(for_live_router)
router.include_router(login_router)





@router.delete("/clear", status_code=status.HTTP_200_OK)
async def clear_all_data(db: AsyncSession = Depends(get_db)):
    """
    Clear ALL data from all tables.
    """
    # Disable foreign key checks for easy deletion
    await db.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
    
    tables = [
        "ai_events", "live", "chat_messages", 
        "room_live_session_members", "room_live_sessions", 
        "room_members", "rooms", 
        "dm_messages", "dm_participants", "dm_threads", 
        "user_friends", "auth_tokens", "users"
    ]
    
    for table in tables:
        await db.execute(text(f"TRUNCATE TABLE {table}"))
        
    await db.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    await db.commit()
    
    return {"message": "All data cleared successfully!"}
