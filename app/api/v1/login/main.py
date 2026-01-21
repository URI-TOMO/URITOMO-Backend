from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional
from pydantic import BaseModel, EmailStr

from app.infra.db import get_db
from app.models.user import User
from app.core.token import create_access_token

router = APIRouter()

# ============ Schemas ============

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class SNSLoginRequest(BaseModel):
    sns_type: str  # e.g., "google", "apple"
    access_token: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str

# ============ Endpoints ============

@router.post("/login", response_model=LoginResponse, tags=["auth"])
async def login(
    request: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Standard login with email and password. (Mocked validation)
    """
    stmt = select(User).where(User.email == request.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password"
        )
    
    access_token = create_access_token(data={"sub": user.id})
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id
    )

@router.post("/login/sns", response_model=LoginResponse, tags=["auth"])
async def login_sns(
    request: SNSLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    SNS login (Google, Apple, etc.) (Mocked)
    """
    user_id = f"sns_{request.sns_type}_{request.access_token[:8]}"
    
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        user = User(
            id=user_id,
            display_name=f"SNS User {request.access_token[:4]}",
            email=f"{user_id}@example.com",
            status="active"
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)

    access_token = create_access_token(data={"sub": user.id})
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        user_id=user.id
    )
