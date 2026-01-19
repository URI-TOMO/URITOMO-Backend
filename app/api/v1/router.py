"""
API v1 Router
"""

from fastapi import APIRouter

from app.example.router import router as example_router
from app.example.token.router import router as example_token_router
from app.api.v1.user.main import router as main_router
from app.api.v1.user.setup_mock import router as setup_mock_router

api_router = APIRouter()

# Include example routers
api_router.include_router(example_router)
api_router.include_router(example_token_router)


# Include user routers
api_router.include_router(main_router)
api_router.include_router(setup_mock_router)
