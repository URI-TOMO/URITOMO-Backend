"""
API router for example CRUD operations.
"""

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.example.crud import RoomCRUD, UserCRUD
from app.example.schemas import (
    MessageResponse,
    RoomCreate,
    RoomResponse,
    RoomUpdate,
    UserCreate,
    UserResponse,
    UserUpdate,
)
from app.infra.db import get_db

router = APIRouter(prefix="/example", tags=["Example CRUD"])


# ============ User Endpoints ============

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

    # Check email uniqueness if provided
    if user_data.email:
        existing_email = await UserCRUD.get_by_email(db, user_data.email)
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with email '{user_data.email}' already exists",
            )

    user = await UserCRUD.create(db, user_data)
    return user


@router.get(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Get user by ID",
)
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific user by their ID"""
    user = await UserCRUD.get_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )
    return user


@router.get(
    "/users",
    response_model=List[UserResponse],
    summary="Get all users",
)
async def get_users(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """Get all users with pagination"""
    users = await UserCRUD.get_all(db, skip=skip, limit=limit)
    return users


@router.patch(
    "/users/{user_id}",
    response_model=UserResponse,
    summary="Update user",
)
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a user's information. Only provided fields will be updated.
    """
    # Check email uniqueness if being updated
    if user_data.email:
        existing_email = await UserCRUD.get_by_email(db, user_data.email)
        if existing_email and existing_email.id != user_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with email '{user_data.email}' already exists",
            )

    user = await UserCRUD.update(db, user_id, user_data)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )
    return user


@router.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    summary="Delete user",
)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Delete a user by their ID"""
    success = await UserCRUD.delete(db, user_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id '{user_id}' not found",
        )
    return MessageResponse(
        message="User deleted successfully",
        detail=f"User '{user_id}' has been deleted",
    )


# ============ Room Endpoints ============

@router.post(
    "/rooms",
    response_model=RoomResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new room",
)
async def create_room(
    room_data: RoomCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new room with the following information:
    - **id**: Unique room identifier (required)
    - **title**: Room title (optional)
    - **created_by**: User ID of the room creator (required)
    - **status**: Room status (default: active)
    """
    # Check if room already exists
    existing_room = await RoomCRUD.get_by_id(db, room_data.id)
    if existing_room:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Room with id '{room_data.id}' already exists",
        )

    # Verify creator exists
    creator = await UserCRUD.get_by_id(db, room_data.created_by)
    if not creator:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Creator user with id '{room_data.created_by}' not found",
        )

    room = await RoomCRUD.create(db, room_data)
    return room


@router.get(
    "/rooms/{room_id}",
    response_model=RoomResponse,
    summary="Get room by ID",
)
async def get_room(
    room_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific room by its ID"""
    room = await RoomCRUD.get_by_id(db, room_id)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room with id '{room_id}' not found",
        )
    return room


@router.get(
    "/rooms",
    response_model=List[RoomResponse],
    summary="Get all rooms",
)
async def get_rooms(
    creator_id: str = Query(None, description="Filter by creator user ID"),
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """Get all rooms with optional filtering by creator and pagination"""
    if creator_id:
        rooms = await RoomCRUD.get_by_creator(db, creator_id, skip=skip, limit=limit)
    else:
        rooms = await RoomCRUD.get_all(db, skip=skip, limit=limit)
    return rooms


@router.patch(
    "/rooms/{room_id}",
    response_model=RoomResponse,
    summary="Update room",
)
async def update_room(
    room_id: str,
    room_data: RoomUpdate,
    db: AsyncSession = Depends(get_db),
):
    """
    Update a room's information. Only provided fields will be updated.
    """
    room = await RoomCRUD.update(db, room_id, room_data)
    if not room:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room with id '{room_id}' not found",
        )
    return room


@router.delete(
    "/rooms/{room_id}",
    response_model=MessageResponse,
    summary="Delete room (soft delete)",
)
async def delete_room(
    room_id: str,
    hard: bool = Query(False, description="Permanently delete the room"),
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a room by its ID.
    - By default, performs a soft delete (sets ended_at and status to 'ended')
    - Use ?hard=true for permanent deletion
    """
    if hard:
        success = await RoomCRUD.hard_delete(db, room_id)
    else:
        success = await RoomCRUD.delete(db, room_id)

    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Room with id '{room_id}' not found",
        )

    delete_type = "permanently deleted" if hard else "soft deleted (ended)"
    return MessageResponse(
        message="Room deleted successfully",
        detail=f"Room '{room_id}' has been {delete_type}",
    )
