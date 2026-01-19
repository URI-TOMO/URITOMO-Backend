"""
API router for User creation and Mock data setup.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.example.user.crud import UserCRUD
from app.example.user.schemas import (
    UserCreate,
    UserResponse,
)
from app.infra.db import get_db

router = APIRouter(prefix="/example", tags=["Example CRUD"])


@router.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new user with the following information:
    - **id**: Unique user identifier (required)
    - **email**: User email address (optional)
    - **display_name**: User display name (required)
    - **locale**: User locale preference (optional)
    - **status**: User status (default: active)
    """
    # Check if user already exists
    existing_user = await UserCRUD.get_by_id(db, user_data.id)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"User with id '{user_data.id}' already exists",
        )

    user = await UserCRUD.create(db, user_data)
    return user


@router.post(
    "/setup-mock",
    status_code=status.HTTP_200_OK,
    summary="Setup mock data for a user",
)
async def setup_mock(
    user_id: str = Query(..., description="User ID to set up mock data for"),
    db: AsyncSession = Depends(get_db),
):
    """
    Sets up mock data for a given user_id:
    - Creates the user if they don't exist
    - Adds 2 friends (already accepted)
    - Adds 2 active rooms the user is a member of
    """
    try:
        result = await UserCRUD.create_mock_data(db, user_id)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup mock data: {str(e)}",
        )
