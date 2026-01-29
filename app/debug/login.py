from datetime import datetime, timedelta
import hashlib
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.infra.db import get_db
from app.core.token import create_access_token, create_refresh_token
from app.models.user import User
from app.models.token import AuthToken
from app.core.config import settings

router = APIRouter(tags=["debug"])

class DebugLoginRequest(BaseModel):
    username: str

@router.post("/login")
async def debug_login(
    data: DebugLoginRequest = Body(...),
    db: AsyncSession = Depends(get_db)
):
    """
    **Debug Login Endpoint**
    
    Accepts a 'username' (which is treated as the user ID in mock data).
    Checks if the user exists in the database.
    If found, issues and returns tokens directly at the root of the response.
    """
    user_id = data.username
    
    # 1. Fetch User by ID (Since mock data uses simple IDs like "1", "2", "jin", etc.)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        # Try finding by display name as a secondary option
        result = await db.execute(select(User).where(User.display_name == user_id))
        user = result.scalar_one_or_none()
        
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{user_id}' not found in database. Please run /debug/seed first.")

    # 2. Issue Tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})
    
    # 3. Save Refresh Token to DB (Optional but good for consistency)
    refresh_expire = datetime.utcnow() + timedelta(minutes=settings.refresh_token_expire_minutes)
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    db_token = AuthToken(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token_hash=token_hash,
        expires_at=refresh_expire,
        issued_at=datetime.utcnow(),
        session_id=str(uuid.uuid4())
    )
    db.add(db_token)
    await db.commit()
    
    # 4. Return flat response (root level)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user_id": user.id,
        "display_name": user.display_name,
        "locale": user.locale
    }
