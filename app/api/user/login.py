from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from uuid import uuid4

from app.infra.db import get_db
from app.models.user import User
from app.core.security import get_password_hash, verify_password
from app.core.token import create_access_token, create_refresh_token
from app.core.errors import AppError, AuthenticationError, ValidationError

router = APIRouter(tags=["auth"])

# ============ Schemas ============

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str

# ============ Endpoints ============

@router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(
    data: SignupRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    General Sign-up
    """
    # 1. Check if user already exists
    stmt = select(User).where(User.email == data.email)
    result = await db.execute(stmt)
    if result.scalar_one_or_none():
        raise ValidationError("User with this email already exists")

    # 2. Create new user
    new_user = User(
        id=str(uuid4()),
        email=data.email,
        display_name=data.name,
        hashed_password=get_password_hash(data.password),
        status="active"
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 3. Generate tokens
    access_token = create_access_token(data={"sub": new_user.id})
    refresh_token = create_refresh_token(data={"sub": new_user.id})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=new_user.id
    )


@router.post("/general_login", response_model=TokenResponse)
async def general_login(
    data: LoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    General Login
    """
    # 1. Fetch User
    stmt = select(User).where(User.email == data.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user or not user.hashed_password:
        raise AuthenticationError("Invalid email or password")

    # 2. Verify Password
    if not verify_password(data.password, user.hashed_password):
        raise AuthenticationError("Invalid email or password")

    # 3. Generate tokens
    access_token = create_access_token(data={"sub": user.id})
    refresh_token = create_refresh_token(data={"sub": user.id})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user.id
    )
