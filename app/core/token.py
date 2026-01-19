"""
Token management and validation logic.
All JWT and authentication dependency operations are centralized here.
"""

from datetime import datetime, timedelta
from typing import Annotated, Optional

from fastapi import Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

from app.core.config import settings
from app.core.errors import AuthenticationError

# HTTP Bearer scheme (Only shows a token input box in Swagger)
security_scheme = HTTPBearer()



def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)

    to_encode.update({"exp": expire, "iat": datetime.utcnow()})

    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expire_days)
    to_encode.update({"exp": expire, "iat": datetime.utcnow(), "type": "refresh"})

    encoded_jwt = jwt.encode(
        to_encode, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )

    return encoded_jwt


def decode_token(token: str) -> Optional[dict]:
    """Decode and validate JWT token string"""
    try:
        payload = jwt.decode(
            token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )
        return payload
    except JWTError:
        return None


def verify_token(token: str) -> Optional[str]:
    """Verify token and extract user ID (sub claim)"""
    payload = decode_token(token)
    if payload is None:
        return None

    user_id: Optional[str] = payload.get("sub")
    return user_id


async def get_current_user_id(auth: Annotated[HTTPAuthorizationCredentials, Depends(security_scheme)]) -> str:
    """
    FastAPI dependency to validate token and return current user ID.
    Used in protected routes.
    """
    token = auth.credentials
    user_id = verify_token(token)
    if not user_id:
        raise AuthenticationError("Could not validate credentials")
    return user_id



# Frequently used Dependency Annotation
CurrentUserDep = Annotated[str, Depends(get_current_user_id)]
