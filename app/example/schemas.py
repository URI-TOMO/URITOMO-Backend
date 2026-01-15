"""
Pydantic schemas for example CRUD operations.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field


# ============ User Schemas ============

class UserBase(BaseModel):
    """Base user schema with common fields"""
    email: Optional[EmailStr] = None
    display_name: str = Field(..., min_length=1, max_length=128)
    locale: Optional[str] = Field(None, max_length=8)


class UserCreate(UserBase):
    """Schema for creating a new user"""
    id: str = Field(..., min_length=1, max_length=64)
    status: str = Field(default="active", max_length=16)


class UserUpdate(BaseModel):
    """Schema for updating a user"""
    email: Optional[EmailStr] = None
    display_name: Optional[str] = Field(None, min_length=1, max_length=128)
    locale: Optional[str] = Field(None, max_length=8)
    status: Optional[str] = Field(None, max_length=16)


class UserResponse(UserBase):
    """Schema for user response"""
    id: str
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============ Room Schemas ============

class RoomBase(BaseModel):
    """Base room schema with common fields"""
    title: Optional[str] = Field(None, max_length=255)


class RoomCreate(RoomBase):
    """Schema for creating a new room"""
    id: str = Field(..., min_length=1, max_length=64)
    created_by: str = Field(..., min_length=1, max_length=64)
    status: str = Field(default="active", max_length=16)


class RoomUpdate(BaseModel):
    """Schema for updating a room"""
    title: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, max_length=16)


class RoomResponse(RoomBase):
    """Schema for room response"""
    id: str
    created_by: str
    status: str
    created_at: datetime
    ended_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ============ Generic Response Schemas ============

class MessageResponse(BaseModel):
    """Generic message response"""
    message: str
    detail: Optional[str] = None
