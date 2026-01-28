from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.token import get_current_user_id
from app.infra.db import get_db
from app.models.user import User

router = APIRouter(tags=["profile"])


class UserProfile(BaseModel):
    id: str
    email: Optional[EmailStr]
    display_name: str


class UserProfileUpdateRequest(BaseModel):
    display_name: Optional[str] = None


@router.get("/user/profile", response_model=UserProfile)
async def get_user_profile(
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
    )


@router.patch("/user/profile", response_model=UserProfile)
async def update_user_profile(
    data: UserProfileUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: str = Depends(get_current_user_id),
):
    if data.display_name is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No update fields provided",
        )

    if data.display_name is not None and not data.display_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="display_name cannot be empty",
        )

    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if data.display_name is not None:
        user.display_name = data.display_name

    await db.commit()
    await db.refresh(user)

    return UserProfile(
        id=user.id,
        email=user.email,
        display_name=user.display_name,
    )
