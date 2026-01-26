from datetime import timedelta
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from livekit import api as lk_api

from app.core.config import settings
from app.core.deps import SessionDep
from app.core.errors import AppError, NotFoundError, PermissionError
from app.core.token import CurrentUserDep
from app.models.room import Room, RoomMember
from app.models.user import User

router = APIRouter(prefix="/meeting/livekit", tags=["meetings"])


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


@router.post("/token", response_model=LiveKitTokenResponse)
async def create_livekit_token(
    data: LiveKitTokenRequest,
    current_user_id: CurrentUserDep,
    session: SessionDep,
):
    _require_livekit_env()

    room_result = await session.execute(select(Room).where(Room.id == data.room_id))
    room = room_result.scalar_one_or_none()
    if not room:
        raise NotFoundError("Room not found")

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

    token_builder = (
        lk_api.AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
        .with_identity(current_user_id)
        .with_grants(grants)
        .with_ttl(timedelta(seconds=3600))
        .with_name(member.display_name)
    )

    if user.locale in {"ko", "ja"}:
        token_builder = token_builder.with_attributes({"lang": user.locale})

    jwt = token_builder.to_jwt()

    return LiveKitTokenResponse(url=settings.livekit_url, token=jwt)
