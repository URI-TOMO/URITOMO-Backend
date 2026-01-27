from datetime import datetime, timedelta
import random
import hashlib
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, status, Query, HTTPException
from pydantic import BaseModel
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

class DebugUserInfo(BaseModel):
    id: str
    email: Optional[str]
    display_name: str
    locale: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime


@router.get("/user_info", response_model=list[DebugUserInfo], status_code=status.HTTP_200_OK)
async def debug_user_info(
    name: str = Query(..., min_length=1, max_length=128),
    db: AsyncSession = Depends(get_db),
):
    """
    Fetch user rows by display_name (debug).
    """
    result = await db.execute(select(User).where(User.display_name == name))
    users = result.scalars().all()
    if not users:
        raise HTTPException(status_code=404, detail=f"User '{name}' not found.")

    return [
        DebugUserInfo(
            id=user.id,
            email=user.email,
            display_name=user.display_name,
            locale=user.locale,
            status=user.status,
            created_at=user.created_at,
            updated_at=user.updated_at,
        )
        for user in users
    ]




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
