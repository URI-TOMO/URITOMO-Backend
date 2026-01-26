import argparse
import asyncio
import inspect
import json
import math
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
    router: Optional["LangRouter"] = None
    ko_pub_sid: Optional[str] = None
    ja_pub_sid: Optional[str] = None


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


def make_sine_frame(
    *,
    freq_hz: float,
    sample_rate: int,
    num_channels: int,
    samples_per_channel: int,
    t0: float,
    amplitude: float = 0.15,
) -> rtc.AudioFrame:
    max_i16 = 32767
    data = bytearray(num_channels * samples_per_channel * 2)

    for i in range(samples_per_channel):
        t = t0 + (i / sample_rate)
        s = int(max_i16 * amplitude * math.sin(2.0 * math.pi * freq_hz * t))
        s = max(min(s, 32767), -32768)
        for ch in range(num_channels):
            idx = (i * num_channels + ch) * 2
            data[idx : idx + 2] = int(s).to_bytes(2, byteorder="little", signed=True)

    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=samples_per_channel,
    )


async def play_beep(
    source: rtc.AudioSource,
    *,
    freq_hz: float,
    label: str,
    on_ms: int = 600,
    off_ms: int = 1400,
    pattern: Optional[list[tuple[int, int]]] = None,
) -> None:
    frame_ms = 20
    sample_rate = source.sample_rate
    num_channels = source.num_channels
    samples = int(sample_rate * frame_ms / 1000)
    beep_pattern = pattern or [(on_ms, off_ms)]

    try:
        while True:
            for on_ms, off_ms in beep_pattern:
                start = time.time()
                for n in range(int(on_ms / frame_ms)):
                    frame = make_sine_frame(
                        freq_hz=freq_hz,
                        sample_rate=sample_rate,
                        num_channels=num_channels,
                        samples_per_channel=samples,
                        t0=start + (n * frame_ms / 1000.0),
                    )
                    await source.capture_frame(frame)
                    await asyncio.sleep(frame_ms / 1000.0)
                await asyncio.sleep(off_ms / 1000.0)
            print(f"[BEEP] {label} pattern done")
    except asyncio.CancelledError:
        raise


async def publish_output_track(
    room: rtc.Room,
    *,
    track_name: str,
) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication]:
    sample_rate = 48000
    num_channels = 1

    source = rtc.AudioSource(sample_rate=sample_rate, num_channels=num_channels)
    track = rtc.LocalAudioTrack.create_audio_track(track_name, source)

    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_MICROPHONE

    pub = await room.local_participant.publish_track(track, opts)
    print(f"[PUBLISH] track_name={track_name} sid={pub.sid}")
    return source, pub


class LangRouter:
    def __init__(
        self,
        room: rtc.Room,
        *,
        ko_sid: str,
        ja_sid: str,
        unknown_policy: str,
    ) -> None:
        self.room = room
        self.ko_sid = ko_sid
        self.ja_sid = ja_sid
        self.unknown_policy = unknown_policy
        self._lock = asyncio.Lock()
        self._pending: Optional[asyncio.Task] = None

    def schedule_recompute(self, reason: str) -> None:
        if self._pending and not self._pending.done():
            return
        self._pending = asyncio.create_task(self._debounced_recompute(reason))

    async def apply_now(self, reason: str) -> None:
        async with self._lock:
            self._apply_permissions(reason)

    async def _debounced_recompute(self, reason: str) -> None:
        await asyncio.sleep(0.15)
        async with self._lock:
            self._apply_permissions(reason)

    def _allowed_for_lang(self, lang: Optional[str]) -> list[str]:
        if lang == "ko":
            return [self.ko_sid]
        if lang == "ja":
            return [self.ja_sid]
        if self.unknown_policy == "ko":
            return [self.ko_sid]
        if self.unknown_policy == "none":
            return []
        return [self.ko_sid, self.ja_sid]

    def _resolve_permission_class(self):
        candidates = [
            "ParticipantTrackPermission",
            "participant.ParticipantTrackPermission",
            "proto_room.ParticipantTrackPermission",
        ]
        for path in candidates:
            obj = rtc
            ok = True
            for part in path.split("."):
                if not hasattr(obj, part):
                    ok = False
                    break
                obj = getattr(obj, part)
            if ok:
                return obj
        return None

    def _make_permission(self, identity: str, allowed: list[str]):
        cls = self._resolve_permission_class()
        if cls is None:
            return {
                "participant_identity": identity,
                "allowed_track_sids": allowed,
                "allow_all": False,
                "all_tracks_allowed": False,
            }
        try:
            return cls(
                participant_identity=identity,
                allow_all=False,
                allowed_track_sids=allowed,
            )
        except TypeError:
            return cls(
                participant_identity=identity,
                all_tracks_allowed=False,
                allowed_track_sids=allowed,
            )

    def _apply_permissions(self, reason: str) -> None:
        perms: list[rtc.ParticipantTrackPermission] = []
        for participant in self.room.remote_participants.values():
            lang = None
            try:
                lang = (participant.attributes or {}).get("lang")
            except Exception:
                lang = None
            allowed = self._allowed_for_lang(lang)
            perms.append(self._make_permission(participant.identity, allowed))

        try:
            self.room.local_participant.set_track_subscription_permissions(
                allow_all_participants=False,
                participant_permissions=perms,
            )
            print(f"[ROUTE] recompute ok reason={reason} participants={len(perms)}")
        except Exception as exc:
            print(f"[ROUTE] recompute failed reason={reason} error={exc!r}")


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
    ko_track: str,
    ja_track: str,
    ko_hz: float,
    ja_hz: float,
) -> None:
    if room_id in rooms:
        print(f"[ROOM] already connected room_id={room_id}")
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
        if state.router:
            state.router.schedule_recompute("participant_connected")

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"[ROOM] participant_disconnected room_id={room_id} identity={participant.identity}")
        if state.router:
            state.router.schedule_recompute("participant_disconnected")
        if len(room.remote_participants) == 0:
            print(f"[ROOM] no participants left, disconnecting room_id={room_id}")
            asyncio.create_task(disconnect_room(room_id, rooms))

    @room.on("participant_attributes_changed")
    def _on_participant_attributes_changed(
        changed_attributes: dict,
        participant: rtc.Participant,
    ):
        print(
            "[ROOM] participant_attributes_changed "
            f"room_id={room_id} identity={participant.identity} changed={changed_attributes}"
        )
        if state.router and ("lang" in changed_attributes):
            state.router.schedule_recompute("participant_attributes_changed")

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

    ko_source, ko_pub = await publish_output_track(room, track_name=ko_track)
    ja_source, ja_pub = await publish_output_track(room, track_name=ja_track)

    ko_pattern = [(800, 1200)]
    ja_pattern = [(200, 150), (200, 900)]
    ko_task = asyncio.create_task(
        play_beep(ko_source, freq_hz=ko_hz, label=f"KO({ko_hz}Hz)", pattern=ko_pattern)
    )
    ja_task = asyncio.create_task(
        play_beep(ja_source, freq_hz=ja_hz, label=f"JA({ja_hz}Hz)", pattern=ja_pattern)
    )
    unknown_policy = os.getenv("LIVEKIT_UNKNOWN_LANG_POLICY", "none").lower()
    if unknown_policy not in {"both", "ko", "none"}:
        unknown_policy = "none"
    state.router = LangRouter(
        room,
        ko_sid=ko_pub.sid,
        ja_sid=ja_pub.sid,
        unknown_policy=unknown_policy,
    )
    await state.router.apply_now("initial")
    state.tasks.update({ko_task, ja_task})
    for task in (ko_task, ja_task):
        task.add_done_callback(state.tasks.discard)


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
    ko_track: str,
    ja_track: str,
    ko_hz: float,
    ja_hz: float,
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
            print(f"[EVENT] action={action} room_id={room_id}")
            if action == "join":
                await connect_room(
                    room_id=room_id,
                    auth=auth,
                    auto_subscribe=auto_subscribe,
                    rooms=rooms,
                    retry_seconds=retry_seconds,
                    max_attempts=max_attempts,
                    ko_track=ko_track,
                    ja_track=ja_track,
                    ko_hz=ko_hz,
                    ja_hz=ja_hz,
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
    parser.add_argument("--ko-track", default=None)
    parser.add_argument("--ja-track", default=None)
    parser.add_argument("--ko-hz", type=float, default=None)
    parser.add_argument("--ja-hz", type=float, default=None)
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

    ko_track = args.ko_track or os.getenv("LIVEKIT_KO_TRACK", "lk.out.ko")
    ja_track = args.ja_track or os.getenv("LIVEKIT_JA_TRACK", "lk.out.ja")
    ko_hz = args.ko_hz or float(os.getenv("LIVEKIT_KO_HZ", "320.0"))
    ja_hz = args.ja_hz or float(os.getenv("LIVEKIT_JA_HZ", "880.0"))

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
            ko_track=ko_track,
            ja_track=ja_track,
            ko_hz=ko_hz,
            ja_hz=ja_hz,
        )

    await listen_room_events(
        redis_url=redis_url,
        channel=channel,
        auth=auth,
        rooms=rooms,
        auto_subscribe=auto_subscribe,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
        ko_track=ko_track,
        ja_track=ja_track,
        ko_hz=ko_hz,
        ja_hz=ja_hz,
    )


if __name__ == "__main__":
    asyncio.run(main())
