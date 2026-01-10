"""
Auth Service

User registration and authentication logic.
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import get_password_hash, verify_password, create_access_token
from app.models.user import User
from app.schemas.auth import UserCreate, UserLogin


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        result = await self.session.execute(select(User).where(User.email == email))
        return result.scalars().first()

    async def authenticate_user(self, login_data: UserLogin) -> Optional[User]:
        """Authenticate user by email and password"""
        user = await self.get_user_by_email(login_data.email)
        if not user:
            return None
        if not verify_password(login_data.password, user.hashed_password):
            return None
        return user

    async def create_user(self, user_in: UserCreate) -> User:
        """Create a new user"""
        user = User(
            email=user_in.email,
            hashed_password=get_password_hash(user_in.password),
            full_name=user_in.full_name,
        )
        self.session.add(user)
        # We don't commit here, usually handled by caller or dependency injection, 
        # but for simple service we can.
        # However, to support transaction blocks, better to let caller commit.
        # But for MVP simplicity, we auto-commit if success.
        await self.session.commit()
        await self.session.refresh(user)
        return user
