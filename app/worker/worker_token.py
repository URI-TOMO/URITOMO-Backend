from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from jose import jwt

from app.core.config import settings
from app.core.errors import AppError, AuthenticationError
from app.core.token import create_access_token
from app.core.logging import get_logger

router = APIRouter(prefix="/worker", tags=["worker"])
logger = get_logger(__name__)


class WorkerTokenRequest(BaseModel):
    room_id: str = Field(..., min_length=1, max_length=128)
    worker_id: str = Field(default="livekit_worker", min_length=1, max_length=128)
    name: Optional[str] = Field(default=None, max_length=128)
    ttl_seconds: int = Field(default=0, ge=0, le=86400)


class WorkerTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


def _require_worker_service_key() -> None:
    if not settings.worker_service_key:
        raise AppError(
            message="Worker service key is missing",
            status_code=500,
            code="WORKER_SERVICE_KEY_MISSING",
        )


@router.post("/token", response_model=WorkerTokenResponse)
def create_worker_token(
    data: WorkerTokenRequest,
    x_worker_key: Optional[str] = Header(default=None, alias="X-Worker-Key"),
):
    _require_worker_service_key()

    if not x_worker_key or x_worker_key != settings.worker_service_key:
        raise AuthenticationError("Invalid worker service key")

    payload = {
        "sub": f"worker:{data.worker_id}",
        "role": "worker",
        "room_id": data.room_id,
    }
    if data.name:
        payload["name"] = data.name

    if data.ttl_seconds == 0:
        payload["iat"] = datetime.utcnow()
        token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
        expires_in = 0
    else:
        token = create_access_token(
            data=payload,
            expires_delta=timedelta(seconds=data.ttl_seconds),
        )
        expires_in = data.ttl_seconds

    logger.info(
        "Issued worker token",
        worker_id=data.worker_id,
        room_id=data.room_id,
        expires_in=expires_in,
        token=token,
    )

    return WorkerTokenResponse(
        access_token=token,
        expires_in=expires_in,
    )
