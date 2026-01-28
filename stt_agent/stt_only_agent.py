import argparse
import asyncio
import base64
import inspect
import json
import os
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

import audioop
import httpx
import websockets
from livekit import rtc
from redis import asyncio as aioredis
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.infra.db import AsyncSessionLocal
from app.models.message import ChatMessage
from app.models.room import RoomMember


REALTIME_SAMPLE_RATE = 24000
LIVEKIT_SAMPLE_RATE = 48000


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
    realtime_ko: Optional["STTSession"] = None
    realtime_ja: Optional["STTSession"] = None
    active_langs: set[str] = field(default_factory=set)
    empty_check_task: Optional[asyncio.Task] = None


def normalize_lang(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    value = value.strip().lower()
    if value in {"kr", "kor", "korean"}:
        return "ko"
    if value in {"jp", "jpn", "japanese"}:
        return "ja"
    if value.startswith("ko"):
        return "ko"
    if value.startswith("ja"):
        return "ja"
    return None


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


def pcm16_resample(data: bytes, *, from_rate: int, to_rate: int, state):
    if from_rate == to_rate:
        return data, state
    converted, next_state = audioop.ratecv(data, 2, 1, from_rate, to_rate, state)
    return converted, next_state


class STTSession:
    """STT dedicated session using OpenAI Realtime API (text-only modality)"""
    def __init__(
        self,
        *,
        lang: str,
        room_id: str,
        api_key: str,
        model: str,
        base_url: str,
        transcribe_model: str,
        vad_threshold: float,
        vad_prefix_ms: int,
        vad_silence_ms: int,
        save_stt: bool,
    ) -> None:
        self.lang = lang
        self.room_id = room_id
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.transcribe_model = transcribe_model
        self.vad_threshold = vad_threshold
        self.vad_prefix_ms = vad_prefix_ms
        self.vad_silence_ms = vad_silence_ms
        self._save_stt = save_stt

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._send_lock = asyncio.Lock()

        self._last_speaker_identity: Optional[str] = None
        self._last_speaker_name: Optional[str] = None
        self._last_speaker_lang: Optional[str] = None
        self._last_speaker_ts = 0.0
        self._member_cache: dict[str, Optional[str]] = {}

    def note_speaker(self, identity: str, name: Optional[str], lang: Optional[str]) -> None:
        self._last_speaker_identity = identity
        self._last_speaker_name = name
        self._last_speaker_lang = lang
        self._last_speaker_ts = time.time()

    def _speaker_tag(self) -> str:
        if not self._last_speaker_identity:
            return "unknown"
        name = f"{self._last_speaker_name}" if self._last_speaker_name else "anon"
        lang = self._last_speaker_lang or "unknown"
        return f"name={name} id={self._last_speaker_identity} user_lang={lang}"

    def _format_stt_block(self, text: str) -> str:
        speaker = self._speaker_tag()
        return (
            "------------ STT ------------\n"
            f"[Speaker] {speaker}\n"
            f"[SessionLang] {self.lang}\n"
            f"[Data] {text}\n"
            "-----------------------------"
        )

    def _session_update_payload(self) -> dict:
        pcm_format = {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE}
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": "You are an STT engine. Transcribe audio to text accurately.",
                "output_modalities": ["text"], # Text only for STT
                "audio": {
                    "input": {
                        "format": pcm_format,
                        "transcription": {
                            "model": self.transcribe_model,
                            "language": self.lang,
                        },
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": self.vad_threshold,
                            "prefix_padding_ms": self.vad_prefix_ms,
                            "silence_duration_ms": self.vad_silence_ms,
                            "create_response": False, # Do not auto-respond
                            "interrupt_response": False,
                        },
                    },
                },
            },
        }

    async def start(self) -> None:
        url = f"{self.base_url}?model={self.model}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        self._ws = await websockets.connect(url, extra_headers=headers)
        payload = self._session_update_payload()
        await self._send_json(payload)
        print(f"[STT] session.update sent lang={self.lang}")
        self._ready.set()
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        print(f"[STT] connected lang={self.lang}")

    async def close(self) -> None:
        self._closed = True
        if self._send_task:
            self._send_task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        print(f"[STT] closed lang={self.lang}")

    def send_audio(self, pcm16_24k: bytes) -> None:
        if self._closed or not self._ready.is_set():
            return
        if pcm16_24k:
            self._send_queue.put_nowait(pcm16_24k)

    async def _send_json(self, payload: dict) -> None:
        if not self._ws:
            return
        async with self._send_lock:
            await self._ws.send(json.dumps(payload))

    async def _send_loop(self) -> None:
        assert self._ws is not None
        try:
            while True:
                chunk = await self._send_queue.get()
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
                await self._send_json(payload)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[STT] send_loop error lang={self.lang} err={exc!r}")

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("type")
                
                if event_type in {
                    "conversation.item.input_audio_transcription.completed",
                    "input_audio_transcription.completed",
                }:
                    transcript = data.get("transcript") or data.get("text") or ""
                    if transcript:
                        print(self._format_stt_block(transcript))
                        asyncio.create_task(self._save_transcript(transcript))
                elif event_type in {
                    "conversation.item.input_audio_transcription.delta",
                    "input_audio_transcription.delta",
                }:
                    delta_text = data.get("delta") or data.get("text") or ""
                    if delta_text:
                        # Debug-ish log for partial results
                        pass
                elif event_type == "session.updated":
                    print(f"[STT] session.updated lang={self.lang}")
                elif event_type == "error":
                    print(f"[STT] error lang={self.lang} data={data}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[STT] recv_loop error lang={self.lang} err={exc!r}")

    async def _save_transcript(self, transcript: str) -> None:
        if not self._save_stt:
            return
        if not transcript:
            return
        speaker_id = self._last_speaker_identity
        if not speaker_id:
            return
        member_id = self._member_cache.get(speaker_id)
        try:
            async with AsyncSessionLocal() as session:
                if member_id is None and speaker_id not in self._member_cache:
                    result = await session.execute(
                        select(RoomMember).where(
                            RoomMember.room_id == self.room_id,
                            RoomMember.user_id == speaker_id,
                        )
                    )
                    member = result.scalar_one_or_none()
                    member_id = member.id if member else None
                    self._member_cache[speaker_id] = member_id

                if not member_id:
                    return

                for _ in range(3):
                    seq_result = await session.execute(
                        select(func.max(ChatMessage.seq)).where(ChatMessage.room_id == self.room_id)
                    )
                    max_seq = seq_result.scalar() or 0
                    next_seq = max_seq + 1
                    message_id = f"stt_{uuid.uuid4().hex[:16]}"
                    new_message = ChatMessage(
                        id=message_id,
                        room_id=self.room_id,
                        seq=next_seq,
                        sender_type="human",
                        sender_member_id=member_id,
                        message_type="stt",
                        text=transcript,
                        lang=self._last_speaker_lang,
                        meta={
                            "speaker_identity": speaker_id,
                            "speaker_name": self._last_speaker_name,
                            "session_lang": self.lang,
                        },
                        created_at=datetime.utcnow(),
                    )
                    session.add(new_message)
                    try:
                        await session.commit()
                        print(
                            "🧾 [STT] saved "
                            f"room_id={self.room_id} seq={next_seq} "
                            f"member_id={member_id} lang={self._last_speaker_lang}"
                        )
                        return
                    except IntegrityError:
                        await session.rollback()
                        continue
        except Exception as exc:
            print(f"[STT] save failed room_id={self.room_id} err={exc!r}")


async def maybe_await(result) -> None:
    if inspect.iscoroutine(result):
        await result


def compute_active_langs(room: rtc.Room, unknown_policy: str) -> set[str]:
    langs: set[str] = set()
    for participant in room.remote_participants.values():
        lang = normalize_lang((participant.attributes or {}).get("lang"))
        if lang:
            langs.add(lang)
            continue
        if unknown_policy == "ko":
            langs.add("ko")
        elif unknown_policy == "ja":
            langs.add("ja")
        elif unknown_policy == "both":
            langs.update({"ko", "ja"})
    return langs


async def consume_audio(
    track: rtc.Track,
    *,
    state: RoomState,
    unknown_policy: str,
    label: str,
    participant_identity: str,
    participant_name: Optional[str],
    participant_lang: Optional[str],
) -> None:
    frames = 0
    last_report = time.time()
    try:
        stream = rtc.AudioStream.from_track(track=track, sample_rate=LIVEKIT_SAMPLE_RATE, num_channels=1)
    except Exception:
        try:
            stream = rtc.AudioStream(track=track, sample_rate=LIVEKIT_SAMPLE_RATE, num_channels=1)
        except TypeError:
            stream = rtc.AudioStream(track=track)

    resample_state = None
    try:
        async for event in stream:
            frame = getattr(event, "frame", None)
            if frame is None:
                continue
            data = frame.data
            channels = frame.num_channels
            if channels > 1:
                data = audioop.tomono(data, 2, 0.5, 0.5)
                channels = 1
            data, resample_state = pcm16_resample(
                data,
                from_rate=frame.sample_rate,
                to_rate=REALTIME_SAMPLE_RATE,
                state=resample_state,
            )

            active_langs = compute_active_langs(state.room, unknown_policy)
            state.active_langs = active_langs

            if "ko" in active_langs and state.realtime_ko:
                state.realtime_ko.note_speaker(participant_identity, participant_name, participant_lang)
                state.realtime_ko.send_audio(data)
            if "ja" in active_langs and state.realtime_ja:
                state.realtime_ja.note_speaker(participant_identity, participant_name, participant_lang)
                state.realtime_ja.send_audio(data)

            frames += 1
            now = time.time()
            if now - last_report >= 10.0:
                fps = frames / (now - last_report)
                print(f"[AUDIO] {label} consuming...")
                frames = 0
                last_report = now
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[AUDIO] {label} stream error: {exc!r}")
    finally:
        await stream.aclose()


async def connect_room(
    room_id: str,
    auth: AuthState,
    rooms: dict[str, RoomState],
    retry_seconds: float,
    max_attempts: int,
    unknown_policy: str,
    realtime_model: str,
    realtime_url: str,
    realtime_key: str,
    transcribe_model: str,
    vad_threshold: float,
    vad_prefix_ms: int,
    vad_silence_ms: int,
    save_stt: bool,
) -> None:
    if room_id in rooms:
        return

    token_resp = await fetch_livekit_token_with_retry(
        auth=auth,
        room_id=room_id,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
    )
    print(f"[BOOT] got token. room_id={room_id}")

    room = rtc.Room()
    state = RoomState(room=room)
    rooms[room_id] = state

    @room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        print(f"🟢👤 [ROOM] participant_connected room_id={room_id} identity={participant.identity}")
        if state.empty_check_task and not state.empty_check_task.done():
            state.empty_check_task.cancel()
        state.empty_check_task = None

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"🔴👤 [ROOM] participant_disconnected room_id={room_id} identity={participant.identity}")
        if state.empty_check_task and not state.empty_check_task.done():
            state.empty_check_task.cancel()
        state.empty_check_task = asyncio.create_task(_disconnect_if_empty(room_id, rooms))

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            lang = normalize_lang((participant.attributes or {}).get("lang")) or "unknown"
            label = f"room={room_id} from={participant.identity} track_sid={track.sid}"
            task = asyncio.create_task(
                consume_audio(
                    track,
                    state=state,
                    unknown_policy=unknown_policy,
                    label=label,
                    participant_identity=participant.identity,
                    participant_name=participant.name,
                    participant_lang=lang,
                )
            )
            state.tasks.add(task)
            task.add_done_callback(state.tasks.discard)

    print(f"[BOOT] connecting room_id={room_id}")
    try:
        await room.connect(token_resp.url, token_resp.token, options=rtc.RoomOptions(auto_subscribe=True))
    except Exception as exc:
        rooms.pop(room_id, None)
        print(f"[BOOT] connect failed room_id={room_id}: {exc!r}")
        return
    print(f"🤖🚪 [STT-AGENT] joined room_id={room_id}")

    state.realtime_ko = STTSession(
        lang="ko",
        room_id=room_id,
        api_key=realtime_key,
        model=realtime_model,
        base_url=realtime_url,
        transcribe_model=transcribe_model,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        save_stt=save_stt,
    )
    state.realtime_ja = STTSession(
        lang="ja",
        room_id=room_id,
        api_key=realtime_key,
        model=realtime_model,
        base_url=realtime_url,
        transcribe_model=transcribe_model,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        save_stt=save_stt,
    )

    await asyncio.gather(state.realtime_ko.start(), state.realtime_ja.start())
    print(f"🚀🚀🚀 STT-ONLY AGENT READY! room_id={room_id} 🤖🤖🤖")


async def _disconnect_if_empty(room_id: str, rooms: dict[str, RoomState]) -> None:
    await asyncio.sleep(5.0) # Grace period
    state = rooms.get(room_id)
    if not state:
        return
    if state.room.remote_participants:
        return
    print(f"[ROOM] no participants left, disconnecting room_id={room_id}")
    await disconnect_room(room_id, rooms)


async def disconnect_room(room_id: str, rooms: dict[str, RoomState]) -> None:
    state = rooms.pop(room_id, None)
    if not state:
        return
    if state.empty_check_task and not state.empty_check_task.done():
        state.empty_check_task.cancel()
    for task in list(state.tasks):
        task.cancel()
    if state.realtime_ko:
        await state.realtime_ko.close()
    if state.realtime_ja:
        await state.realtime_ja.close()
    await maybe_await(state.room.disconnect())
    print(f"[BOOT] disconnected room_id={room_id}")


async def listen_room_events(
    redis_url: str,
    channel: str,
    auth: AuthState,
    rooms: dict[str, RoomState],
    retry_seconds: float,
    max_attempts: int,
    unknown_policy: str,
    realtime_model: str,
    realtime_url: str,
    realtime_key: str,
    transcribe_model: str,
    vad_threshold: float,
    vad_prefix_ms: int,
    vad_silence_ms: int,
    save_stt: bool,
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
                print(f"📥🟢 [EVENT] action=join room_id={room_id}")
                try:
                    await connect_room(
                        room_id=room_id,
                        auth=auth,
                        rooms=rooms,
                        retry_seconds=retry_seconds,
                        max_attempts=max_attempts,
                        unknown_policy=unknown_policy,
                        realtime_model=realtime_model,
                        realtime_url=realtime_url,
                        realtime_key=realtime_key,
                        transcribe_model=transcribe_model,
                        vad_threshold=vad_threshold,
                        vad_prefix_ms=vad_prefix_ms,
                        vad_silence_ms=vad_silence_ms,
                        save_stt=save_stt,
                    )
                except Exception as exc:
                    print(f"[EVENT] join failed room_id={room_id} error={exc!r}")
            elif action == "leave":
                await disconnect_room(room_id, rooms)
    finally:
        await pubsub.close()
        await redis.close()


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=False, help="Backend URL")
    parser.add_argument("--room", required=False, help="room_id (optional)")
    args = parser.parse_args()

    backend = args.backend or os.getenv("BACKEND_URL")
    room_id = args.room or os.getenv("ROOM_ID")
    service_auth = normalize_service_auth(os.getenv("SERVICE_AUTH"))
    worker_key = os.getenv("WORKER_SERVICE_KEY")
    worker_id = os.getenv("WORKER_ID", "stt_only_worker")
    worker_ttl = int(os.getenv("WORKER_TOKEN_TTL_SECONDS", "0"))
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    channel = os.getenv("LIVEKIT_ROOM_EVENTS_CHANNEL", "livekit:rooms")
    force_relay_value = os.getenv("LIVEKIT_FORCE_RELAY", "false")
    force_relay = force_relay_value.lower() in {"1", "true", "yes", "y", "on"}

    unknown_policy = os.getenv("LIVEKIT_UNKNOWN_LANG_POLICY", "both").lower()
    realtime_key = os.getenv("OPENAI_API_KEY")
    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    realtime_url = os.getenv("OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime")
    transcribe_model = os.getenv("OPENAI_REALTIME_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    
    vad_threshold = float(os.getenv("OPENAI_REALTIME_VAD_THRESHOLD", "0.5"))
    vad_prefix_ms = int(os.getenv("OPENAI_REALTIME_VAD_PREFIX_MS", "300"))
    vad_silence_ms = int(os.getenv("OPENAI_REALTIME_VAD_SILENCE_MS", "500"))

    if not backend:
        raise RuntimeError("Missing backend. Provide --backend or env BACKEND_URL")
    if not realtime_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    
    save_stt_value = os.getenv("OPENAI_STT_SAVE", "true")
    save_stt = save_stt_value.lower() in {"1", "true", "yes", "y", "on"}

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
            rooms=rooms,
            retry_seconds=retry_seconds,
            max_attempts=max_attempts,
            unknown_policy=unknown_policy,
            realtime_model=realtime_model,
            realtime_url=realtime_url,
            realtime_key=realtime_key,
            transcribe_model=transcribe_model,
            vad_threshold=vad_threshold,
            vad_prefix_ms=vad_prefix_ms,
            vad_silence_ms=vad_silence_ms,
            save_stt=save_stt,
        )

    await listen_room_events(
        redis_url=redis_url,
        channel=channel,
        auth=auth,
        rooms=rooms,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
        unknown_policy=unknown_policy,
        realtime_model=realtime_model,
        realtime_url=realtime_url,
        realtime_key=realtime_key,
        transcribe_model=transcribe_model,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        save_stt=save_stt,
    )


if __name__ == "__main__":
    asyncio.run(main())
