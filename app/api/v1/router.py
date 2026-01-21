"""
API v1 Router
"""

from fastapi import APIRouter, Depends

from app.example.user.router import router as example_router
from app.example.token.router import router as example_token_router
from app.api.v1.user.main import router as main_router
from app.api.v1.user.setup_mock import router as setup_mock_router
from app.api.v1.login.main import router as login_router
from app.api.v1.login.setup_mock import router as login_setup_mock_router
from app.api.v1.summary.documents import router as summary_documents_router
from app.api.v1.summary.main import router as summary_main_router
from app.api.v1.summary.meeting_member import router as summary_member_router
from app.api.v1.summary.translation_log import router as summary_translation_log_router
from app.api.v1.summary.setup_mock import router as summary_setup_mock_router

from app.core.token import security_scheme

api_router = APIRouter()

# 1. Routes that DON'T need authentication (Public/Debug)
api_router.include_router(example_router) # Includes login-debug
api_router.include_router(login_router)
api_router.include_router(login_setup_mock_router)

# 2. Routes that DO need authentication (Protected)
api_router.include_router(example_token_router, dependencies=[Depends(security_scheme)])
api_router.include_router(main_router, dependencies=[Depends(security_scheme)])
api_router.include_router(setup_mock_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_documents_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_main_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_member_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_translation_log_router, dependencies=[Depends(security_scheme)])
api_router.include_router(summary_setup_mock_router, dependencies=[Depends(security_scheme)])



