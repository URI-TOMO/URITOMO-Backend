from datetime import datetime, timedelta
import random
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.core.token import create_access_token, create_refresh_token
from app.models.user import User
from app.models.token import AuthToken

router = APIRouter(tags=["debug"])

@router.post("/signin-random")
async def signin_random_user(db: AsyncSession = Depends(get_db)):
    """
    Creates a user with a random 2-digit ID (u_XX) or gets existing one,
    and returns auth tokens.
    """
    
    # 1. Generate Random 2-digit ID (10~99)
    # We try a few times to find an available or existing one, or just pick one.
    # The requirement says "random id 2 digits".
    
    random_num = random.randint(10, 99)
    user_id = f"u_{random_num}"
    
    # 2. Check if user exists
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        # Create new user
        user = User(
            id=user_id,
            email=f"{user_id}@example.com",
            display_name=f"User {random_num}",
            locale="en",
            status="active",
            created_at=datetime.utcnow()
        )
        db.add(user)
        # Flush to ensure user exists for token foreign key
        await db.flush()
    
    # 3. Issue Tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})
    
    # 4. Save Refresh Token to DB
    # We parse the refresh token to get exp/iat, or just calculate it again.
    # ideally create_refresh_token could return the object but it returns str string.
    # We'll just assume standard expiry from config for DB record.
    from app.core.config import settings
    refresh_expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    
    # Simple hash simulation or actual hash if needed. 
    # For debug/dev, we might store raw or simple hash. 
    # The existing debug/api.py uses hashlib.sha256.
    import hashlib
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    db_token = AuthToken(
        id=str(uuid.uuid4()),
        user_id=user.id,
        token_hash=token_hash,
        expires_at=refresh_expire,
        issued_at=datetime.utcnow(),
        session_id=str(uuid.uuid4()) # Dummy session ID
    )
    db.add(db_token)
    await db.commit()
    
    return {
        "user_id": user.id,
        "display_name": user.display_name,
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }
