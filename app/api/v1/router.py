"""
API Router configuration
"""

from fastapi import APIRouter

from app.api.v1.core import health
from app.api.v1.auth import (
    login,
    users,
    orgs,
)
from app.api.v1.rooms import router as rooms_router
from app.api.v1.ai import (
    segments,
    ws_realtime,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(login.router, prefix="/auth", tags=["auth"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(orgs.router, prefix="/orgs", tags=["orgs"])
api_router.include_router(rooms_router.router, prefix="/rooms", tags=["rooms"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])
api_router.include_router(ws_realtime.router, prefix="/ws", tags=["websocket"])
