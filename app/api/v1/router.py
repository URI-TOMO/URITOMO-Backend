"""
API v1 Router
"""

from fastapi import APIRouter

from app.example.router import router as example_router
from app.user.google_login import router as google_login_router

api_router = APIRouter()

# Include example CRUD router
api_router.include_router(example_router)

# Include Google Login router
api_router.include_router(google_login_router)
