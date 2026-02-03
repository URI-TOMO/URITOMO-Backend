import asyncio
import json
import os
from typing import Optional

from redis import asyncio as aioredis

from app.core.config import settings
from app.meeting.ws.manager import manager
from core.logging_system import emit_log, get_event_logger

STT_EVENTS_CHANNEL = os.getenv("LIVEKIT_STT_EVENTS_CHANNEL", "livekit:stt")
STT_LOGGER = get_event_logger("uritomo.stt")


def _log(
    level: str,
    *,
    domain: str,
    event: str,
    summary: str,
    payload: Optional[str] = None,
    **kv,
) -> None:
    emit_log(
        STT_LOGGER,
        level=level,
        domain=domain,
        event=event,
        summary=summary,
        kv=kv,
        payload=payload,
    )


async def start_stt_event_listener() -> None:
    if not settings.enable_websocket:
        _log(
            "WARN",
            domain="stt",
            event="stt.listener.disabled",
            summary="WebSocket disabled; STT listener not started",
        )
        return

    redis = aioredis.from_url(
        settings.redis_url,
        db=settings.redis_db,
        encoding="utf-8",
        decode_responses=True,
    )
    pubsub = redis.pubsub()
    await pubsub.subscribe(STT_EVENTS_CHANNEL)
    _log(
        "INFO",
        domain="redis",
        event="stt.redis.sub",
        summary="Subscribed to STT channel",
        channel=STT_EVENTS_CHANNEL,
    )

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
                _log(
                    "WARN",
                    domain="stt",
                    event="stt.redis.invalid_json",
                    summary="Invalid STT payload; ignored",
                    payload=str(raw),
                )
                continue

            room_id = payload.get("room_id")
            ws_message = payload.get("message")
            if not room_id or not ws_message:
                _log(
                    "WARN",
                    domain="stt",
                    event="stt.redis.missing_fields",
                    summary="Missing room_id or message; ignored",
                    payload=str(raw),
                )
                continue

            await manager.broadcast(room_id, ws_message)

            data = ws_message.get("data") or {}
            _log(
                "INFO",
                domain="broadcast",
                event="stt.broadcast",
                summary="STT broadcasted",
                room_id=room_id,
                session_id=data.get("session_id"),
                seq=data.get("seq"),
                payload=data.get("text") or "",
            )
    except asyncio.CancelledError:
        pass
    finally:
        try:
            await pubsub.unsubscribe(STT_EVENTS_CHANNEL)
            await pubsub.close()
            _log(
                "INFO",
                domain="redis",
                event="stt.redis.unsub",
                summary="Unsubscribed from STT channel",
                channel=STT_EVENTS_CHANNEL,
            )
        finally:
            await redis.close()
