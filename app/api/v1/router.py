"""
API Router configuration
"""

from fastapi import APIRouter

from app.api.v1 import (
    health,
    auth,
    orgs,
    meetings,
    segments,
    ws_realtime,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(orgs.router, prefix="/orgs", tags=["orgs"])
api_router.include_router(meetings.router, prefix="/meetings", tags=["meetings"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])
api_router.include_router(ws_realtime.router, prefix="/ws", tags=["websocket"])
