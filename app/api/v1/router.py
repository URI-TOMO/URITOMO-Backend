"""
API v1 Router
"""

from fastapi import APIRouter, Depends

from app.example.user.router import router as example_router
from app.debug.api import router as debug_router
from app.example.token.router import router as example_token_router
from app.api.v1.user.main import router as main_router
from app.user.login import router as auth_router

from app.meeting.api import router as meeting_router

from app.core.token import security_scheme

api_router = APIRouter()

# 1. Routes that DON'T need authentication (Public/Debug)
api_router.include_router(example_router) # Includes login-debug

api_router.include_router(auth_router)


# 2. Routes that DO need authentication (Protected)
api_router.include_router(example_token_router, dependencies=[Depends(security_scheme)])
api_router.include_router(main_router, dependencies=[Depends(security_scheme)])
api_router.include_router(meeting_router, dependencies=[Depends(security_scheme)])



