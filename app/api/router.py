"""
API v1 Router
"""

from fastapi import APIRouter, Depends

from app.example.user.router import router as example_router
from app.debug.api import router as debug_router
from app.example.token.router import router as example_token_router
from app.user.login import router as auth_router
from app.api.user.main import router as main_router

from app.meeting.api import router as meeting_router
from app.meeting.ws.ws_base import router as meeting_ws_router
from app.meeting.live_history import router as meeting_history_router

from app.core.token import security_scheme

api_router = APIRouter()

# 1. Routes that DON'T need authentication (Public/Debug)
api_router.include_router(auth_router)  # Includes real signup/login
api_router.include_router(example_router) # Includes login-debug
api_router.include_router(debug_router, prefix="/debug")

# 2. Routes that DO need authentication (Protected)
api_router.include_router(example_token_router, dependencies=[Depends(security_scheme)])
api_router.include_router(main_router, dependencies=[Depends(security_scheme)])
api_router.include_router(meeting_router, dependencies=[Depends(security_scheme)])
api_router.include_router(meeting_history_router, dependencies=[Depends(security_scheme)])
api_router.include_router(meeting_ws_router)

# 3. Summary Routes (Protected)
from app.api.summary.documents import router as summary_documents_router
from app.api.summary.main import router as summary_main_router
from app.api.summary.meeting_member import router as summary_member_router
from app.api.summary.translation_log import router as summary_translation_log_router
from app.api.summary.setup_mock import router as summary_setup_mock_router

api_router.include_router(summary_documents_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_main_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_member_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_translation_log_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_setup_mock_router, dependencies=[Depends(security_scheme)])




