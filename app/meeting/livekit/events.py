import json
from typing import Optional

from redis.asyncio import Redis

ROOM_EVENT_CHANNEL = "livekit:rooms"


async def publish_room_event(
    redis: Redis,
    *,
    action: str,
    room_id: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    payload = {
        "action": action,
        "room_id": room_id,
    }
    if session_id:
        payload["session_id"] = session_id
    if user_id:
        payload["user_id"] = user_id
    await redis.publish(ROOM_EVENT_CHANNEL, json.dumps(payload))
