"""
Simple Profile Endpoint
"""

from fastapi import APIRouter
from pydantic import BaseModel
from app.core.deps import get_db
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()

class ProfileRequest(BaseModel):
    """Profile request body"""
    name: str
    email: str
    age: int | None = None


@router.post("/profile")
async def create_profile(profile: ProfileRequest):
   

   if age = 0 : 
    return {
        "success": False,
        "message": "Age must be greater than 0"
    }
    

    return {
        "success": True
    }


