from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from livekit import api as lk_api

from app.core.config import settings
from app.core.deps import SessionDep, RedisDep
from app.core.errors import AppError, NotFoundError, PermissionError
from app.core.logging import get_logger
from app.core.token import CurrentUserDep, decode_token
from app.meeting.livekit.events import publish_room_event
from app.models.room import Room, RoomMember
from app.models.user import User

router = APIRouter(prefix="/meeting/livekit", tags=["meetings"])
logger = get_logger(__name__)


class LiveKitTokenRequest(BaseModel):
    room_id: str = Field(..., min_length=1, max_length=128)


class LiveKitTokenResponse(BaseModel):
    url: str
    token: str


def _require_livekit_env() -> None:
    if not settings.livekit_url or not settings.livekit_api_key or not settings.livekit_api_secret:
        raise AppError(
            message="LiveKit environment variables are missing",
            status_code=500,
            code="LIVEKIT_CONFIG_MISSING",
        )


def _normalize_lang(locale: Optional[str]) -> Optional[str]:
    if not locale:
        return None
    lowered = locale.lower()
    if lowered in {"kr", "kor", "korea"}:
        return "ko"
    if lowered in {"jp", "jpn", "japan"}:
        return "ja"
    if lowered.startswith("ko"):
        return "ko"
    if lowered.startswith("ja"):
        return "ja"
    return None


@router.post("/token", response_model=LiveKitTokenResponse)
async def create_livekit_token(
    data: LiveKitTokenRequest,
    current_user_id: CurrentUserDep,
    session: SessionDep,
    request: Request,
    redis: RedisDep,
):
    _require_livekit_env()

    payload = None
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1]
        payload = decode_token(token)

    is_worker = payload is not None and payload.get("role") == "worker"

    room_result = await session.execute(select(Room).where(Room.id == data.room_id))
    room = room_result.scalar_one_or_none()
    if not room:
        raise NotFoundError("Room not found")

    if is_worker:
        token_room = payload.get("room_id")
        if token_room and token_room != data.room_id:
            raise PermissionError("Worker token not allowed for this room")

        display_name = payload.get("name") or "LiveKit Worker"

        attributes = {"role": "worker"}

        grants = lk_api.VideoGrants(
            room_join=True,
            room=data.room_id,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )

        token_builder = (
            lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(current_user_id)
            .with_grants(grants)
            .with_ttl(timedelta(seconds=3600))
            .with_name(display_name)
            .with_attributes(attributes)
        )

        jwt = token_builder.to_jwt()

        return LiveKitTokenResponse(url=settings.livekit_url, token=jwt)

    member_result = await session.execute(
        select(RoomMember).where(
            RoomMember.room_id == data.room_id,
            RoomMember.user_id == current_user_id,
        )
    )
    member = member_result.scalar_one_or_none()
    if not member:
        raise PermissionError("Not a member of this room")

    user_result = await session.execute(select(User).where(User.id == current_user_id))
    user = user_result.scalar_one_or_none()
    if not user:
        raise PermissionError("User not found")

    grants = lk_api.VideoGrants(
        room_join=True,
        room=data.room_id,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )

    lang = _normalize_lang(user.locale)
    attributes = {}
    if lang:
        attributes["lang"] = lang

    token_builder = (
        lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(current_user_id)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=3600))
        .with_name(member.display_name)
        .with_attributes(attributes)
    )

    jwt = token_builder.to_jwt()

    if not is_worker:
        try:
            await publish_room_event(
                redis,
                action="join",
                room_id=data.room_id,
                user_id=current_user_id,
            )
        except Exception as exc:
            logger.warning(
                "livekit.room_event.publish_failed",
                room_id=data.room_id,
                user_id=current_user_id,
                error=str(exc),
            )

    logger.info(
        "livekit.token.issued",
        room_id=data.room_id,
        user_id=current_user_id,
        is_worker=is_worker,
    )

    return LiveKitTokenResponse(url=settings.livekit_url, token=jwt)
