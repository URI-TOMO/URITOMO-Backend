"""
Comprehensive Token System Module

This module provides a complete token authentication system including:
- JWT access token generation and verification
- Refresh token management with bcrypt hashing
- Token validation, rotation, and cleanup
- Authentication routes and dependencies
- Pydantic schemas for requests/responses
"""

# ============================================================================
# Imports
# ============================================================================
from datetime import datetime, timezone, timedelta
from typing import Optional, Annotated
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, EmailStr, Field
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
import bcrypt
import secrets

from app.models.token import RefreshToken
from app.models.user import User
from app.infra.db import get_db
from app.core.config import settings


# ============================================================================
# Configuration
# ============================================================================
# Read security configuration from environment
JWT_SECRET_KEY = settings.jwt_secret_key
JWT_ALGORITHM = settings.jwt_algorithm
ACCESS_TOKEN_EXPIRE_MINUTES = settings.access_token_expire_minutes
REFRESH_TOKEN_EXPIRE_DAYS = settings.refresh_token_expire_days


# ============================================================================
# Pydantic Schemas
# ============================================================================

class TokenPayload(BaseModel):
    """JWT token payload schema"""
    sub: str = Field(..., description="Subject (user_id)")
    exp: int = Field(..., description="Expiration time (UNIX timestamp)")
    iat: int = Field(..., description="Issued at (UNIX timestamp)")
    type: str = Field(..., description="Token type (access or refresh)")


class LoginRequest(BaseModel):
    """Login request schema"""
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    """Token response schema"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration (seconds)")


class RefreshRequest(BaseModel):
    """Token refresh request schema"""
    refresh_token: str = Field(..., description="Refresh token")


class RefreshResponse(BaseModel):
    """Token refresh response schema"""
    access_token: str = Field(..., description="New access token")
    refresh_token: Optional[str] = Field(None, description="New refresh token (rotation)")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration (seconds)")


class ErrorResponse(BaseModel):
    """Error response schema"""
    detail: str = Field(..., description="Error message")
    error_code: Optional[str] = Field(None, description="Error code")


class UserResponse(BaseModel):
    """User response schema"""
    id: str = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="User email")
    display_name: str = Field(..., description="Display name")
    locale: Optional[str] = Field(None, description="Locale")
    status: str = Field(..., description="User status")


# ============================================================================
# JWT Handler Functions
# ============================================================================

def get_access_token_expires_delta() -> timedelta:
    """Get access token expiration timedelta"""
    return timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)


def get_refresh_token_expires_delta() -> timedelta:
    """Get the refresh token expiration delta"""
    return timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)


def create_access_token(user_id: str) -> str:
    """
    Generate JWT access token.
    
    Args:
        user_id: User ID to embed in the token
        
    Returns:
        Encoded JWT access token string
    """
    now = datetime.now(timezone.utc)
    expire = now + get_access_token_expires_delta()

    payload: dict[str, str | int] = {
        "sub": str(user_id),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }

    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return token


def verify_access_token(token: str) -> Optional[TokenPayload]:
    """
    Verify JWT access token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        TokenPayload if valid, None if invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return TokenPayload(**payload)
    except JWTError:
        return None


def create_refresh_token() -> str:
    """
    Generate refresh token (not JWT).
    
    Creates a cryptographically secure random token for refresh token usage.
    
    Returns:
        URL-safe base64-encoded random token (48 bytes = 64 characters)
    """
    return secrets.token_urlsafe(48)


# ============================================================================
# Token Service Functions
# ============================================================================

def hash_token(token: str) -> str:
    """
    Hash a token using bcrypt for secure storage.

    Args:
        token: Plain text token to hash

    Returns:
        Hashed token string
    """
    return bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()


def verify_token_hash(token: str, hashed: str) -> bool:
    """
    Verify a token against its hash.

    Args:
        token: Plain text token
        hashed: Hashed token to compare against

    Returns:
        True if token matches hash, False otherwise
    """
    return bcrypt.checkpw(token.encode(), hashed.encode())


async def store_refresh_token(
    db: AsyncSession,
    user_id: str,
    token: str
) -> RefreshToken:
    """
    Store a new refresh token in the database.

    Args:
        db: Database session
        user_id: ID of the user who owns this token (string format)
        token: Plain text refresh token to store (will be hashed)

    Returns:
        Created RefreshToken database record
    """
    token_hash = hash_token(token)
    expires_at = datetime.now(timezone.utc) + get_refresh_token_expires_delta()
    
    db_token = RefreshToken(
        user_id=user_id,
        token_hash=token_hash,
        expires_at=expires_at,
        revoked=False
    )
    db.add(db_token)

    return db_token


async def validate_refresh_token(
    db: AsyncSession,
    token: str
) -> Optional[RefreshToken]:
    """
    Validate a refresh token by checking:
    - Token exists in database
    - Token is not expired
    - Token is not revoked

    Args:
        db: Database session
        token: Plain text refresh token to validate

    Returns:
        RefreshToken record if valid, None otherwise
    """
    token_entry = await find_refresh_token_by_value(db, token)

    if not token_entry:
        return None

    # Check if token is expired
    expires_at = token_entry.expires_at  # type: ignore
    now = datetime.now(timezone.utc)
    # Make both timezone-aware or both naive for comparison
    if expires_at.tzinfo is None:
        now = datetime.now()
    if expires_at < now:  # type: ignore
        return None

    # Check if token is revoked
    if token_entry.revoked:  # type: ignore
        return None

    return token_entry


async def rotate_refresh_token(
    db: AsyncSession,
    old_token: str,
    new_token: str
) -> Optional[RefreshToken]:
    """
    Rotate a refresh token by:
    1. Revoking the old token
    2. Creating a new token
    3. Linking the old token to the new token

    Args:
        db: Database session
        old_token: Plain text old refresh token to revoke
        new_token: Plain text new refresh token to create

    Returns:
        New RefreshToken record if successful, None if old token is invalid
    """
    old_token_entry = await validate_refresh_token(db, old_token)

    if not old_token_entry:
        return None

    # Create new token
    user_id: str = old_token_entry.user_id  # type: ignore
    new_token_entry = await store_refresh_token(db, user_id, new_token)
    
    # Revoke old token and link to new token
    old_token_id: int = old_token_entry.id  # type: ignore
    new_token_id: int = new_token_entry.id  # type: ignore
    old_token_entry.revoked = True  # type: ignore
    old_token_entry.replaced_by_token = new_token_id  # type: ignore

    return new_token_entry


async def cleanup_expired_tokens(db: AsyncSession) -> int:
    """
    Remove expired refresh tokens from the database.

    Args:
        db: Database session

    Returns:
        Number of tokens deleted
    """
    from sqlalchemy import delete
    now = datetime.now(timezone.utc)
    result = await db.execute(
        delete(RefreshToken).where(RefreshToken.expires_at < now)
    )
    await db.commit()
    # rowcount may be None depending on dialect; default to 0
    return result.rowcount or 0


async def find_refresh_token_by_value(
    db: AsyncSession,
    token: str
) -> Optional[RefreshToken]:
    """
    Find a refresh token in the database by its value.

    Args:
        db: Database session
        token: Plain text refresh token to find

    Returns:
        RefreshToken record if found, None otherwise
    """
    from sqlalchemy import select
    now = datetime.now(timezone.utc)

    # Filter to only non-revoked, non-expired tokens, then check hash
    result = await db.execute(
        select(RefreshToken).where(
            RefreshToken.revoked.is_(False),
            RefreshToken.expires_at > now,
        )
    )
    candidates = result.scalars().all()
    for entry in candidates:
        if verify_token_hash(token, entry.token_hash):  # type: ignore[attr-defined]
            return entry
    return None


# ============================================================================
# FastAPI Dependencies
# ============================================================================

security = HTTPBearer()


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)],
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer token credentials
        db: Database session
        
    Returns:
        User: Authenticated user object
        
    Raises:
        HTTPException: 401 if token is invalid or user not found
    """
    from sqlalchemy import select
    
    token = credentials.credentials
    payload = verify_access_token(token)

    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.sub
    result = await db.execute(select(User).filter(User.id == user_id))
    user = result.scalars().first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


# Type alias for dependency injection
CurrentUser = Annotated[User, Depends(get_current_user)]


# ============================================================================
# FastAPI Routes
# ============================================================================

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/login", 
    response_model=TokenResponse, 
    responses={401: {"model": ErrorResponse}}, 
    status_code=status.HTTP_200_OK,
    summary="User login",
    description="Authenticate user with email and password, returns access and refresh tokens"
)
async def login(data: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    User login endpoint.
    
    Validates email and password, then issues:
    - Access token (JWT) - short-lived
    - Refresh token - long-lived, stored in database
    
    Args:
        data: Login credentials (email, password)
        db: Database session
        
    Returns:
        TokenResponse with access_token and refresh_token
        
    Raises:
        HTTPException 401: Invalid credentials
    """
    from sqlalchemy import select
    
    result = await db.execute(select(User).filter(User.email == data.email))
    user = result.scalars().first()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # TODO: パスワード検証を実装
    # password_hash: str = user.password_hash
    # if not verify_password(data.password, password_hash):
    #     raise HTTPException(...)

    user_id: str = user.id
    access_token = create_access_token(user_id)
    refresh_token_value = create_refresh_token()
    token_entry = store_refresh_token(db, user_id, refresh_token_value)
    db.add(token_entry)
    await db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token_value,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/refresh", 
    response_model=RefreshResponse, 
    responses={401: {"model": ErrorResponse}}, 
    status_code=status.HTTP_200_OK,
    summary="Refresh access token",
    description="Exchange refresh token for new access token and refresh token (rotation)"
)
async def refresh_token(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Token refresh endpoint.
    
    Implements refresh token rotation:
    1. Validates old refresh token
    2. Issues new access token
    3. Issues new refresh token
    4. Revokes old refresh token
    
    Args:
        data: Refresh token request
        db: Database session
        
    Returns:
        RefreshResponse with new tokens
        
    Raises:
        HTTPException 401: Invalid or expired refresh token
    """
    token_entry = await find_refresh_token_by_value(db, data.refresh_token)
    
    if not token_entry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = token_entry.user_id  # type: ignore
    new_refresh_token_value = create_refresh_token()
    new_token_entry = await rotate_refresh_token(db, data.refresh_token, new_refresh_token_value)
    
    if not new_token_entry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to rotate refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    new_access_token = create_access_token(user_id)

    return RefreshResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token_value,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


@router.post(
    "/logout", 
    responses={200: {"description": "Success"}, 401: {"model": ErrorResponse}}, 
    status_code=status.HTTP_200_OK,
    summary="User logout",
    description="Revoke refresh token to log out user"
)
async def logout(data: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Logout endpoint.
    
    Revokes the provided refresh token, preventing further use.
    Client should also discard the access token.
    
    Args:
        data: Refresh token to revoke
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException 401: Invalid refresh token
        HTTPException 500: Database error
    """
    token_entry = find_refresh_token_by_value(db, data.refresh_token)
    
    if not token_entry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        token_entry.revoked = True  # type: ignore
        db.add(token_entry)
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}",
        )

    return {"detail": "Logged out successfully"}


@router.get(
    "/me", 
    responses={401: {"model": ErrorResponse}}, 
    status_code=status.HTTP_200_OK,
    summary="Get current user",
    description="Get information about the currently authenticated user"
)
async def get_current_user_info(current_user: CurrentUser):
    """
    Get current user info endpoint.
    
    Returns information about the authenticated user based on the access token.
    
    Args:
        current_user: Injected authenticated user
        
    Returns:
        User information
    """
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name=current_user.display_name,
        locale=current_user.locale,
        status=current_user.status
    )