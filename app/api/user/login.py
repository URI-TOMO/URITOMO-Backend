from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
from uuid import uuid4
from datetime import datetime, timedelta
import hashlib

from app.infra.db import get_db
from app.models.user import User
from app.models.token import AuthToken
from app.core.security import get_password_hash, verify_password
from app.core.token import create_access_token, create_refresh_token
from app.core.config import settings
from app.core.errors import AppError, AuthenticationError, ValidationError

router = APIRouter(tags=["auth"])

# ============ Schemas ============

class SignupRequest(BaseModel):
    name: str
    email: str
    password: str
    lang: str


def _normalize_lang(value: str) -> str:
    lowered = value.strip().lower()
    if lowered in {"kr", "kor", "korea"}:
        return "ko"
    if lowered in {"jp", "jpn", "japan"}:
        return "ja"
    if lowered.startswith("ko"):
        return "ko"
    if lowered.startswith("ja"):
        return "ja"
    return ""

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

    lang = _normalize_lang(data.lang)
    if not lang:
        raise ValidationError("lang must be one of ko/ja (or kr/jp)")

    # 2. Create new user
    new_user = User(
        id=str(uuid4()),
        email=data.email,
        display_name=data.name,
        locale=lang,
        hashed_password=get_password_hash(data.password),
        status="active"
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    # 3. Generate tokens
    access_token = create_access_token(data={"sub": new_user.id})
    refresh_token = create_refresh_token(data={"sub": new_user.id})
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    # 4. Persist refresh token (non-debug login)
    db.add(AuthToken(
        id=str(uuid4()),
        user_id=new_user.id,
        token_hash=token_hash,
        issued_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(minutes=settings.refresh_token_expire_minutes),
    ))
    await db.commit()

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
    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()

    # 4. Persist refresh token (non-debug login)
    db.add(AuthToken(
        id=str(uuid4()),
        user_id=user.id,
        token_hash=token_hash,
        issued_at=datetime.utcnow(),
        expires_at=datetime.utcnow() + timedelta(minutes=settings.refresh_token_expire_minutes),
    ))
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        user_id=user.id
    )
