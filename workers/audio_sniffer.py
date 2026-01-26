import argparse
import asyncio
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx
from livekit import rtc
from redis import asyncio as aioredis


@dataclass
class BackendTokenResponse:
    url: str
    token: str


@dataclass
class AuthState:
    backend: str
    service_auth: Optional[str]
    worker_key: Optional[str]
    worker_id: str
    worker_ttl: int
    force_relay: bool


@dataclass
class RoomState:
    room: rtc.Room
    tasks: set[asyncio.Task] = field(default_factory=set)


async def fetch_livekit_token(
    backend_base_url: str,
    room_id: str,
    service_auth_header_value: str,
    timeout_s: float = 10.0,
) -> BackendTokenResponse:
    endpoint = backend_base_url.rstrip("/") + "/meeting/livekit/token"
    headers = {
        "Content-Type": "application/json",
        "Authorization": service_auth_header_value,
    }
    payload = {"room_id": room_id}

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Token API failed: {response.status_code} {response.text}")
        data = response.json()
        if "url" not in data or "token" not in data:
            raise RuntimeError(f"Unexpected token response: {json.dumps(data, ensure_ascii=False)}")
        return BackendTokenResponse(url=data["url"], token=data["token"])


async def fetch_worker_auth(
    backend_base_url: str,
    room_id: str,
    worker_key: str,
    worker_id: str,
    ttl_seconds: int,
    timeout_s: float = 10.0,
) -> str:
    endpoint = backend_base_url.rstrip("/") + "/worker/token"
    headers = {
        "Content-Type": "application/json",
        "X-Worker-Key": worker_key,
    }
    payload = {
        "room_id": room_id,
        "worker_id": worker_id,
        "ttl_seconds": ttl_seconds,
    }

    async with httpx.AsyncClient(timeout=timeout_s) as client:
        response = await client.post(endpoint, headers=headers, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"Worker token API failed: {response.status_code} {response.text}")
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Unexpected worker token response: {json.dumps(data, ensure_ascii=False)}")
        return f"Bearer {token}"


def normalize_service_auth(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().strip('"').strip("'")
    value = " ".join(value.split())
    if not value.lower().startswith("bearer "):
        value = f"Bearer {value}"
    if value.lower() == "bearer":
        return None
    return value


def build_room_options(auto_subscribe: bool, force_relay: bool) -> rtc.RoomOptions:
    if not force_relay:
        return rtc.RoomOptions(auto_subscribe=auto_subscribe)

    rtc_config = None
    ice_transport = None
    if hasattr(rtc, "IceTransportType"):
        ice_transport = getattr(rtc.IceTransportType, "TRANSPORT_RELAY", None)
    if ice_transport is None and hasattr(rtc, "proto_room") and hasattr(rtc.proto_room, "IceTransportType"):
        ice_transport = getattr(rtc.proto_room.IceTransportType, "TRANSPORT_RELAY", None)

    if ice_transport is not None:
        for key in ("ice_transport_type", "ice_transport_policy"):
            try:
                rtc_config = rtc.RtcConfiguration(**{key: ice_transport})
                break
            except TypeError:
                rtc_config = None

    if rtc_config is None:
        rtc_config = rtc.RtcConfiguration()

    return rtc.RoomOptions(auto_subscribe=auto_subscribe, rtc_config=rtc_config)


async def ensure_service_auth(auth: AuthState, room_id: str) -> Optional[str]:
    if auth.service_auth:
        return auth.service_auth
    if auth.worker_key:
        auth.service_auth = await fetch_worker_auth(
            backend_base_url=auth.backend,
            room_id=room_id,
            worker_key=auth.worker_key,
            worker_id=auth.worker_id,
            ttl_seconds=auth.worker_ttl,
        )
    return auth.service_auth


async def fetch_livekit_token_with_retry(
    auth: AuthState,
    room_id: str,
    retry_seconds: float,
    max_attempts: int,
) -> BackendTokenResponse:
    attempt = 0
    refreshed = False
    while True:
        attempt += 1
        service_auth = await ensure_service_auth(auth, room_id)
        if not service_auth:
            raise RuntimeError("Missing auth. Provide SERVICE_AUTH or WORKER_SERVICE_KEY.")
        try:
            return await fetch_livekit_token(
                backend_base_url=auth.backend,
                room_id=room_id,
                service_auth_header_value=service_auth,
            )
        except Exception as exc:
            if not refreshed and auth.worker_key and ("401" in str(exc) or "403" in str(exc)):
                try:
                    auth.service_auth = await fetch_worker_auth(
                        backend_base_url=auth.backend,
                        room_id=room_id,
                        worker_key=auth.worker_key,
                        worker_id=auth.worker_id,
                        ttl_seconds=auth.worker_ttl,
                    )
                    refreshed = True
                    continue
                except Exception as refresh_exc:
                    print(f"[BOOT] worker token refresh failed: {refresh_exc!r}")
            print(f"[BOOT] token fetch failed (attempt={attempt}): {exc!r}")
            if max_attempts and attempt >= max_attempts:
                raise
            await asyncio.sleep(retry_seconds)


async def consume_audio(track: rtc.Track, *, label: str) -> None:
    try:
        stream = rtc.AudioStream.from_track(track=track, sample_rate=48000, num_channels=1)
    except Exception:
        # Fall back to constructor API for older/newer SDKs without from_track.
        try:
            stream = rtc.AudioStream(track=track, sample_rate=48000, num_channels=1)
        except TypeError:
            stream = rtc.AudioStream(track=track)

    frames = 0
    last_report = time.time()
    last_frame_ts: Optional[float] = None

    try:
        async for event in stream:
            frame = getattr(event, "frame", None)
            if frame is not None:
                frames += 1
                last_frame_ts = time.time()

            now = time.time()
            if now - last_report >= 1.0:
                age = (now - last_frame_ts) if last_frame_ts else None
                if age is None:
                    print(f"[AUDIO] {label} fps={frames}/s")
                else:
                    print(f"[AUDIO] {label} fps={frames}/s last_frame_age={age:.3f}s")
                frames = 0
                last_report = now
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[AUDIO] {label} stream error: {exc!r}")
    finally:
        await stream.aclose()


async def maybe_await(result) -> None:
    if inspect.iscoroutine(result):
        await result


async def connect_room(
    room_id: str,
    auth: AuthState,
    auto_subscribe: bool,
    rooms: dict[str, RoomState],
    retry_seconds: float,
    max_attempts: int,
) -> None:
    if room_id in rooms:
        return

    token_resp = await fetch_livekit_token_with_retry(
        auth=auth,
        room_id=room_id,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
    )
    print(f"[BOOT] got token. room_id={room_id} livekit_url={token_resp.url}")

    room = rtc.Room()
    state = RoomState(room=room)
    rooms[room_id] = state

    @room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        print(
            "[ROOM] participant_connected "
            f"room_id={room_id} identity={participant.identity} "
            f"name={participant.name} attrs={participant.attributes}"
        )

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"[ROOM] participant_disconnected room_id={room_id} identity={participant.identity}")

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        print(
            "[ROOM] track_subscribed "
            f"room_id={room_id} kind={track.kind} participant={participant.identity} "
            f"pub_sid={publication.sid} track_sid={track.sid}"
        )

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            label = f"room={room_id} from={participant.identity} track_sid={track.sid}"
            task = asyncio.create_task(consume_audio(track, label=label))
            state.tasks.add(task)
            task.add_done_callback(state.tasks.discard)

    opts = build_room_options(auto_subscribe=auto_subscribe, force_relay=auth.force_relay)
    print(
        f"[BOOT] connecting room_id={room_id} auto_subscribe={auto_subscribe} "
        f"force_relay={auth.force_relay}"
    )
    try:
        await room.connect(token_resp.url, token_resp.token, options=opts)
    except Exception as exc:
        rooms.pop(room_id, None)
        print(f"[BOOT] connect failed room_id={room_id}: {exc!r}")
        return
    print(f"[BOOT] connected. room_id={room_id} room={room.name}")


async def disconnect_room(room_id: str, rooms: dict[str, RoomState]) -> None:
    state = rooms.pop(room_id, None)
    if not state:
        return
    for task in list(state.tasks):
        task.cancel()
    await maybe_await(state.room.disconnect())
    print(f"[BOOT] disconnected room_id={room_id}")


async def listen_room_events(
    redis_url: str,
    channel: str,
    auth: AuthState,
    rooms: dict[str, RoomState],
    auto_subscribe: bool,
    retry_seconds: float,
    max_attempts: int,
) -> None:
    redis = aioredis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    pubsub = redis.pubsub()
    await pubsub.subscribe(channel)
    print(f"[BOOT] subscribed to {channel}")

    try:
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue
            try:
                data = json.loads(message.get("data") or "{}")
            except json.JSONDecodeError:
                continue
            action = data.get("action")
            room_id = data.get("room_id")
            if not room_id:
                continue
            if action == "join":
                await connect_room(
                    room_id=room_id,
                    auth=auth,
                    auto_subscribe=auto_subscribe,
                    rooms=rooms,
                    retry_seconds=retry_seconds,
                    max_attempts=max_attempts,
                )
            elif action == "leave":
                await disconnect_room(room_id, rooms)
    finally:
        await pubsub.close()
        await redis.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=False, help="예: http://localhost:8000")
    parser.add_argument("--room", required=False, help="room_id (선택)")
    parser.add_argument(
        "--auth",
        default=None,
        help='Authorization 헤더 값(예: "Bearer xxx"). 미지정 시 env SERVICE_AUTH 사용',
    )
    parser.add_argument("--auto-subscribe", default=None, choices=["true", "false"])
    args = parser.parse_args()

    backend = args.backend or os.getenv("BACKEND_URL")
    room_id = args.room or os.getenv("ROOM_ID")
    service_auth = normalize_service_auth(args.auth or os.getenv("SERVICE_AUTH"))
    worker_key = os.getenv("WORKER_SERVICE_KEY")
    worker_id = os.getenv("WORKER_ID", "livekit_worker")
    worker_ttl = int(os.getenv("WORKER_TOKEN_TTL_SECONDS", "0"))
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    channel = os.getenv("LIVEKIT_ROOM_EVENTS_CHANNEL", "livekit:rooms")
    force_relay_value = os.getenv("LIVEKIT_FORCE_RELAY", "false")
    force_relay = force_relay_value.lower() in {"1", "true", "yes", "y", "on"}

    if not backend:
        raise RuntimeError("Missing backend. Provide --backend or env BACKEND_URL")

    auto_subscribe_value = args.auto_subscribe or os.getenv("AUTO_SUBSCRIBE", "true")
    auto_subscribe = auto_subscribe_value.lower() == "true"

    auth = AuthState(
        backend=backend,
        service_auth=service_auth,
        worker_key=worker_key,
        worker_id=worker_id,
        worker_ttl=worker_ttl,
        force_relay=force_relay,
    )

    retry_seconds = float(os.getenv("TOKEN_FETCH_RETRY_SECONDS", "2"))
    max_attempts = int(os.getenv("TOKEN_FETCH_MAX_ATTEMPTS", "2"))

    rooms: dict[str, RoomState] = {}

    if room_id:
        await connect_room(
            room_id=room_id,
            auth=auth,
            auto_subscribe=auto_subscribe,
            rooms=rooms,
            retry_seconds=retry_seconds,
            max_attempts=max_attempts,
        )

    await listen_room_events(
        redis_url=redis_url,
        channel=channel,
        auth=auth,
        rooms=rooms,
        auto_subscribe=auto_subscribe,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
    )


if __name__ == "__main__":
    asyncio.run(main())
