import asyncio
import json
import os

from redis import asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger
from app.meeting.ws.manager import manager

logger = get_logger(__name__)

ALIEN_STAMP = "👽" * 20
STT_EVENTS_CHANNEL = os.getenv("LIVEKIT_STT_EVENTS_CHANNEL", "livekit:stt")


async def start_stt_event_listener() -> None:
    if not settings.enable_websocket:
        return

    redis = aioredis.from_url(
        settings.redis_url,
        db=settings.redis_db,
        encoding="utf-8",
        decode_responses=True,
    )
    pubsub = redis.pubsub()
    await pubsub.subscribe(STT_EVENTS_CHANNEL)
    logger.info(f"{ALIEN_STAMP} [STT WS] subscribed channel={STT_EVENTS_CHANNEL}")

    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                await asyncio.sleep(0)
                continue
            raw = message.get("data")
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                continue

            room_id = payload.get("room_id")
            ws_message = payload.get("message")
            if not room_id or not ws_message:
                continue

            await manager.broadcast(room_id, ws_message)

            data = ws_message.get("data") or {}
            logger.info(
                f"{ALIEN_STAMP} [STT WS] sent "
                f"room_id={room_id} seq={data.get('seq')} "
                f"text={data.get('text')!r} translated={data.get('translated_text')!r}"
            )
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await pubsub.unsubscribe(STT_EVENTS_CHANNEL)
            await pubsub.close()
        finally:
            await redis.close()
