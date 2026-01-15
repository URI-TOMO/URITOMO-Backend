"""
API v1 Router
"""

from fastapi import APIRouter

from app.example.router import router as example_router

api_router = APIRouter()

# Include example CRUD router
api_router.include_router(example_router)
