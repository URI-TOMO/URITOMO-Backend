"""
Google Login System - OAuth2 Authentication

Features:
- Google OAuth2 authentication
- JWT token generation
- User creation/retrieval
- Secure token handling
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from google.auth.transport import requests
from google.oauth2 import id_token
from jose import jwt
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.deps import get_current_user_id
from app.core.security import create_access_token
from app.infra.db import get_db
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["Google Login"])


# ============ Schemas ============

class GoogleTokenRequest(BaseModel):
    """Schema for Google token request"""
    token: str = Field(..., description="Google ID token from client")


class GoogleUserInfo(BaseModel):
    """Schema for Google user information"""
    sub: str = Field(..., description="Google user ID")
    email: EmailStr
    name: str
    picture: Optional[str] = None
    locale: Optional[str] = None


class TokenResponse(BaseModel):
    """Schema for token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: "UserResponse"


class UserResponse(BaseModel):
    """Schema for user response"""
    id: str
    email: Optional[str]
    display_name: str
    locale: Optional[str]
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============ CRUD Operations ============

class GoogleLoginCRUD:
    """CRUD operations for Google Login"""

    @staticmethod
    async def get_or_create_user(
        db: AsyncSession, 
        google_info: GoogleUserInfo
    ) -> User:
        """Get existing user or create new one from Google info"""
        # Check if user exists
        result = await db.execute(
            select(User).where(User.id == google_info.sub)
        )
        user = result.scalar_one_or_none()

        if user:
            # Update user information if changed
            if user.email != google_info.email or user.display_name != google_info.name:
                user.email = google_info.email
                user.display_name = google_info.name
                user.locale = google_info.locale
                await db.commit()
                await db.refresh(user)
            return user

        # Create new user
        new_user = User(
            id=google_info.sub,
            email=google_info.email,
            display_name=google_info.name,
            locale=google_info.locale,
            status="active",
        )
        db.add(new_user)
        await db.commit()
        await db.refresh(new_user)
        return new_user

    @staticmethod
    async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        """Get user by ID"""
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
        """Get user by email"""
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()


# ============ Service Layer ============

class GoogleAuthService:
    """Service for Google authentication"""

    @staticmethod
    def verify_google_token(token: str) -> GoogleUserInfo:
        """Verify Google ID token and return user info"""
        try:
            # For development/testing: try JWT decode first
            print(f"[DEBUG] verify_google_token called")
            print(f"[DEBUG] enable_test_auth: {settings.enable_test_auth}")
            print(f"[DEBUG] jwt_secret_key: {settings.jwt_secret_key[:10]}...{settings.jwt_secret_key[-5:]}")
            print(f"[DEBUG] jwt_algorithm: {settings.jwt_algorithm}")
            print(f"[DEBUG] token (first 50 chars): {token[:50]}...")
            
            if settings.enable_test_auth:
                print(f"[DEBUG] Test auth enabled, attempting JWT decode")
                try:
                    payload = jwt.decode(
                        token,
                        settings.jwt_secret_key,
                        algorithms=[settings.jwt_algorithm]
                    )
                    print(f"[DEBUG] JWT decode successful!")
                    print(f"[DEBUG] Payload: {payload}")
                    # Any valid JWT in test mode with required fields is accepted
                    if payload.get('sub') and payload.get('email'):
                        print(f"[DEBUG] Creating GoogleUserInfo from JWT payload")
                        return GoogleUserInfo(
                            sub=payload.get('sub'),
                            email=payload.get('email'),
                            name=payload.get('name', 'Test User'),
                            picture=payload.get('picture'),
                            locale=payload.get('locale', 'en')
                        )
                except Exception as jwt_err:
                    # Log the JWT decode error for debugging
                    print(f"[DEBUG] JWT decode failed: {type(jwt_err).__name__}: {str(jwt_err)}")
                    print(f"[DEBUG] Full error details:", jwt_err)
                    import traceback
                    traceback.print_exc()
                    # Continue to Google verification if JWT decode fails
                    pass
            
            # Production: Verify with real Google OAuth2
            google_client_id = getattr(settings, 'google_client_id', None)
            if not google_client_id:
                raise ValueError('GOOGLE_CLIENT_ID not configured in settings')
            
            idinfo = id_token.verify_oauth2_token(
                token,
                requests.Request(),
                google_client_id
            )

            # Verify issuer
            if idinfo['iss'] not in ['accounts.google.com', 'https://accounts.google.com']:
                raise ValueError('Wrong issuer.')

            # Extract user info
            return GoogleUserInfo(
                sub=idinfo['sub'],
                email=idinfo['email'],
                name=idinfo.get('name', idinfo['email']),
                picture=idinfo.get('picture'),
                locale=idinfo.get('locale', 'en')
            )

        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid Google token: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {type(e).__name__}: {str(e)}"
            )


# ============ API Endpoints ============

@router.post(
    "/google",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate with Google",
)
async def google_login(
    token_request: GoogleTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Authenticate user with Google OAuth2 token:
    - **token**: Google ID token from client-side authentication
    
    Returns JWT access token and user information.
    """
    # Verify Google token
    google_info = GoogleAuthService.verify_google_token(token_request.token)
    
    # Get or create user
    user = await GoogleLoginCRUD.get_or_create_user(db, google_info)
    
    # Generate JWT access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    
    # Get token expiration time
    expires_in = settings.access_token_expire_minutes * 60  # in seconds
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse.model_validate(user)
    )


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
)
async def get_current_user_info(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Get current authenticated user's information"""
    user = await GoogleLoginCRUD.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return UserResponse.model_validate(user)


@router.post(
    "/refresh",
    response_model=TokenResponse,
    summary="Refresh access token",
)
async def refresh_token(
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    """Refresh JWT access token for authenticated user"""
    user = await GoogleLoginCRUD.get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Generate new access token
    access_token = create_access_token(
        data={"sub": user.id, "email": user.email}
    )
    
    expires_in = settings.access_token_expire_minutes * 60
    
    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=expires_in,
        user=UserResponse.model_validate(user)
    )