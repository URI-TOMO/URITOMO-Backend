"""
API Router configuration
"""

from fastapi import APIRouter

from app.api.v1.core import (
    health,
    auth,
    orgs,
    meetings,
)
from app.api.v1.ai import (
    segments,
    ws_realtime,
)
from app.api.v1.examples import (
    redis,
    mysql,
    qdrant,
    openai,
)

api_router = APIRouter()

api_router.include_router(health.router, tags=["health"])
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(orgs.router, prefix="/orgs", tags=["orgs"])
api_router.include_router(meetings.router, prefix="/meetings", tags=["meetings"])
api_router.include_router(segments.router, prefix="/segments", tags=["segments"])
api_router.include_router(ws_realtime.router, prefix="/ws", tags=["websocket"])

# Example APIs for Verification
api_router.include_router(redis.router, prefix="/examples/redis", tags=["examples"])
api_router.include_router(mysql.router, prefix="/examples/mysql", tags=["examples"])
api_router.include_router(qdrant.router, prefix="/examples/qdrant", tags=["examples"])
api_router.include_router(openai.router, prefix="/examples/openai", tags=["examples"])
