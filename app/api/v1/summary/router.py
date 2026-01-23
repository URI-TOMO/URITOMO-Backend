from fastapi import APIRouter, Depends
from app.core.token import security_scheme

from app.api.v1.summary.documents import router as summary_documents_router
from app.api.v1.summary.main import router as summary_main_router
from app.api.v1.summary.meeting_member import router as summary_member_router
from app.api.v1.summary.translation_log import router as summary_translation_log_router
from app.api.v1.summary.setup_mock import router as summary_setup_mock_router

summary_router = APIRouter()

# 認証が必要なルート
summary_router.include_router(summary_documents_router, dependencies=[Depends(security_scheme)])
summary_router.include_router(summary_main_router, dependencies=[Depends(security_scheme)])
summary_router.include_router(summary_member_router, dependencies=[Depends(security_scheme)])
summary_router.include_router(summary_translation_log_router, dependencies=[Depends(security_scheme)])

# デバッグ用ルート
summary_router.include_router(summary_setup_mock_router, prefix="/debug")
