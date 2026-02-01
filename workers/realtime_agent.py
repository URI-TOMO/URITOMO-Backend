import argparse
import asyncio
import base64
import inspect
import json
import os
import re
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

from app.core.time import to_jst_iso
from app.infra.db import AsyncSessionLocal
from app.models.room import RoomMember, RoomLiveSession
from app.models.stt import RoomAiResponse, RoomSttResult
from app.translation.deepl_service import deepl_service
from app.translation.openai_service import openai_service


REALTIME_SAMPLE_RATE = 24000
LIVEKIT_SAMPLE_RATE = 48000
ALIEN_STAMP = "👽" * 20
MOCK_TRANSLATION_PREFIXES = ("[KO]", "[JA]", "[TRANS]", "[Korean]", "[Japanese]")
REALTIME_MIN_COMMIT_MS = 100


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
    realtime_ko: Optional["RealtimeSession"] = None
    realtime_ja: Optional["RealtimeSession"] = None
    active_langs: set[str] = field(default_factory=set)
    ko_pub_sid: Optional[str] = None
    ja_pub_sid: Optional[str] = None
    session_id: Optional[str] = None
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


def lang_code_to_name(value: Optional[str]) -> Optional[str]:
    code = normalize_lang(value)
    if code == "ko":
        return "Korean"
    if code == "ja":
        return "Japanese"
    return None


def opposite_lang_code(value: Optional[str]) -> Optional[str]:
    code = normalize_lang(value)
    if code == "ko":
        return "ja"
    if code == "ja":
        return "ko"
    return None

def looks_like_mock_translation(text: Optional[str]) -> bool:
    if not text:
        return True
    return text.startswith(MOCK_TRANSLATION_PREFIXES)


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


class RealtimeSession:
    def __init__(
        self,
        *,
        lang: str,
        room_id: str,
        session_id: Optional[str],
        api_key: str,
        model: str,
        base_url: str,
        transcribe_model: str,
        trigger_phrases: list[str],
        wake_cooldown_s: float,
        output_source: rtc.AudioSource,
        vad_threshold: float,
        vad_prefix_ms: int,
        vad_silence_ms: int,
        voice: str,
        output_modalities: list[str],
        always_respond: bool,
        history_max_turns: int,
        summary_max_chars: int,
        save_stt: bool,
        trigger_debug: bool,
        redis_url: str,
        stt_channel: str,
        force_commit_ms: int,
    ) -> None:
        self.lang = lang
        self.room_id = room_id
        self._session_id = session_id
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.transcribe_model = transcribe_model
        self.output_source = output_source
        self.voice = voice
        self.output_modalities = output_modalities
        self.vad_threshold = vad_threshold
        self.vad_prefix_ms = vad_prefix_ms
        self.vad_silence_ms = vad_silence_ms
        self._always_respond = always_respond
        self._trigger_debug = trigger_debug
        self._trigger_phrases = [p for p in (phrase.strip() for phrase in trigger_phrases) if p]
        self._trigger_norm = [self._normalize_text(p) for p in self._trigger_phrases]
        self._trigger_prompt = ", ".join(self._trigger_phrases)
        self._wake_cooldown_s = wake_cooldown_s
        self._last_wake_ts = 0.0
        self._history_max_turns = history_max_turns
        self._summary_max_chars = summary_max_chars
        self._history: list[dict[str, str]] = []
        self._assistant_partial = ""
        self._response_in_flight = False
        self._pending_transcript: Optional[str] = None
        self._pending_force = False
        self._pending_log_label: Optional[str] = None
        self._save_stt = save_stt
        self._last_stt_seq: Optional[int] = None
        self._last_stt_text: Optional[str] = None
        self._redis_url = redis_url
        self._stt_channel = stt_channel
        self._force_commit_ms = max(force_commit_ms, 0)
        self._force_commit_s = self._force_commit_ms / 1000.0 if self._force_commit_ms else 0.0

        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._send_task: Optional[asyncio.Task] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._ready = asyncio.Event()
        self._closed = False
        self._send_lock = asyncio.Lock()
        self._buffered_ms = 0.0
        self._last_commit_ts = 0.0

        self._out_buffer = bytearray()
        self._out_state = None
        self._audio_bytes = 0
        self._last_audio_log = 0.0
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

    def _instructions(self) -> str:
        if self.lang == "ko":
            return "You are a voice assistant. Respond only in Korean."
        return "You are a voice assistant. Respond only in Japanese."

    def _session_update_payload(self) -> dict:
        pcm_format = {"type": "audio/pcm", "rate": REALTIME_SAMPLE_RATE}
        return {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": self._instructions(),
                "output_modalities": self.output_modalities,
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
                            "create_response": False,
                            "interrupt_response": True,
                        },
                    },
                    "output": {
                        "format": pcm_format,
                        "voice": self.voice,
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
        print(f"[REALTIME] session.update sent lang={self.lang} keys={list((payload.get('session') or {}).keys())}")
        self._ready.set()
        self._send_task = asyncio.create_task(self._send_loop())
        self._recv_task = asyncio.create_task(self._recv_loop())
        print(f"[REALTIME] connected lang={self.lang}")

    async def close(self) -> None:
        self._closed = True
        if self._send_task:
            self._send_task.cancel()
        if self._recv_task:
            self._recv_task.cancel()
        if self._ws:
            await self._ws.close()
        print(f"[REALTIME] closed lang={self.lang}")

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
            if self._last_commit_ts <= 0:
                self._last_commit_ts = time.monotonic()
            while True:
                chunk = await self._send_queue.get()
                chunk_ms = (len(chunk) / (REALTIME_SAMPLE_RATE * 2)) * 1000.0
                if chunk_ms <= 0:
                    continue
                self._buffered_ms += chunk_ms
                payload = {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(chunk).decode("ascii"),
                }
                await self._send_json(payload)
                if self._force_commit_s:
                    now = time.monotonic()
                    if (
                        now - self._last_commit_ts >= self._force_commit_s
                        and self._buffered_ms >= REALTIME_MIN_COMMIT_MS
                    ):
                        await self._send_json({"type": "input_audio_buffer.commit"})
                        print(
                            f"[REALTIME] buffer.commit sent lang={self.lang} "
                            f"reason=timer interval_ms={self._force_commit_ms} "
                            f"buffer_ms={self._buffered_ms:.1f}"
                        )
                        self._last_commit_ts = now
                        self._buffered_ms = 0.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[REALTIME] send_loop error lang={self.lang} err={exc!r}")

    async def _recv_loop(self) -> None:
        assert self._ws is not None
        try:
            async for message in self._ws:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                event_type = data.get("type")
                if event_type in {"response.output_audio.delta", "response.audio.delta"}:
                    delta = data.get("delta")
                    if delta:
                        audio_bytes = base64.b64decode(delta)
                        self._audio_bytes += len(audio_bytes)
                        now = time.time()
                        if now - self._last_audio_log >= 1.0:
                            print(f"[REALTIME] audio.delta lang={self.lang} bytes={self._audio_bytes}")
                            self._audio_bytes = 0
                            self._last_audio_log = now
                        await self._push_audio(audio_bytes)
                elif event_type in {
                    "conversation.item.input_audio_transcription.completed",
                    "input_audio_transcription.completed",
                }:
                    transcript = data.get("transcript") or data.get("text") or ""
                    if transcript:
                        print(self._format_stt_block(transcript))
                        asyncio.create_task(self._save_transcript(transcript))
                    await self._handle_transcript(transcript)
                elif event_type in {
                    "conversation.item.input_audio_transcription.delta",
                    "input_audio_transcription.delta",
                }:
                    delta_text = data.get("delta") or data.get("text") or ""
                    if delta_text:
                        print(
                            f"✨✍️✨ [STT] speaker=({self._speaker_tag()}) "
                            f"session_lang={self.lang} delta={delta_text!r} ✨✍️✨"
                        )
                elif event_type in {
                    "conversation.item.input_audio_transcription.segment",
                    "input_audio_transcription.segment",
                }:
                    segment_text = data.get("text") or ""
                    if segment_text:
                        print(
                            f"✨🧩✨ [STT] speaker=({self._speaker_tag()}) "
                            f"session_lang={self.lang} segment={segment_text!r} ✨🧩✨"
                        )
                elif event_type == "input_audio_buffer.speech_started":
                    print(f"[REALTIME] vad.started lang={self.lang}")
                elif event_type == "input_audio_buffer.speech_stopped":
                    print(f"[REALTIME] vad.stopped lang={self.lang}")
                elif event_type == "input_audio_buffer.committed":
                    print(f"[REALTIME] buffer.committed lang={self.lang}")
                    self._buffered_ms = 0.0
                    self._last_commit_ts = time.monotonic()
                elif event_type == "input_audio_buffer.cleared":
                    print(f"[REALTIME] buffer.cleared lang={self.lang}")
                    self._buffered_ms = 0.0
                    self._last_commit_ts = time.monotonic()
                elif event_type == "input_audio_buffer.timeout_triggered":
                    print(f"[REALTIME] buffer.timeout lang={self.lang}")
                elif event_type == "response.created":
                    self._response_in_flight = True
                    print(f"[REALTIME] response.created lang={self.lang}")
                elif event_type == "response.done":
                    status = (data.get("response") or {}).get("status")
                    self._response_in_flight = False
                    assistant_text = self._assistant_partial.strip()
                    if assistant_text:
                        self._append_history("assistant", assistant_text)
                        print(f"🤖 [AI] response.text lang={self.lang} text={assistant_text!r}")
                        asyncio.create_task(self._save_ai_response(assistant_text))
                        self._assistant_partial = ""
                    print(f"[REALTIME] response.done lang={self.lang} status={status}")
                    if self._pending_transcript:
                        asyncio.create_task(self._flush_pending_response())
                elif event_type == "response.output_audio.done":
                    pass
                elif event_type in {"response.output_text.delta", "response.text.delta"}:
                    delta_text = data.get("delta") or ""
                    if delta_text:
                        self._assistant_partial += delta_text
                elif event_type in {"response.output_text.done", "response.text.done"}:
                    text_out = data.get("text") or ""
                    if text_out:
                        self._assistant_partial += text_out
                elif event_type == "session.updated":
                    print(f"[REALTIME] session.updated lang={self.lang}")
                elif event_type == "error":
                    print(f"[REALTIME] error lang={self.lang} data={data}")
                    err = data.get("error") or {}
                    if err.get("code") == "input_audio_buffer_commit_empty":
                        self._buffered_ms = 0.0
                        self._last_commit_ts = time.monotonic()
                        self._force_commit_s = 0.0
                        self._force_commit_ms = 0
                        print(
                            f"[REALTIME] force_commit disabled lang={self.lang} "
                            "reason=commit_empty"
                        )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"[REALTIME] recv_loop error lang={self.lang} err={exc!r}")

    async def _send_response(self, transcript: str, *, log_label: str, force: bool) -> None:
        now = time.monotonic()
        if not force and now - self._last_wake_ts < self._wake_cooldown_s:
            return
        self._last_wake_ts = now
        summary = self._build_history_summary()
        system_text = self._build_system_prompt(summary)
        await self._send_json(
            {
                "type": "response.create",
                "response": {
                    "conversation": "auto",
                    "input": [
                        {
                            "type": "message",
                            "role": "system",
                            "content": [{"type": "input_text", "text": system_text}],
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": transcript}],
                        }
                    ],
                },
            }
        )
        print(
            f"[REALTIME] {log_label} lang={self.lang} "
            f"summary_chars={len(summary)} transcript={transcript!r}"
        )

    def _set_pending_response(self, transcript: str, log_label: str) -> None:
        self._pending_transcript = transcript
        self._pending_force = True
        self._pending_log_label = log_label
        print(
            f"[REALTIME] defer response lang={self.lang} "
            f"reason=in_flight transcript={transcript!r}"
        )

    async def _flush_pending_response(self) -> None:
        transcript = self._pending_transcript
        if not transcript:
            return
        log_label = self._pending_log_label or "deferred response"
        force = self._pending_force
        self._pending_transcript = None
        self._pending_log_label = None
        self._pending_force = False
        await self._send_response(transcript, log_label=log_label, force=force)

    async def _handle_transcript(self, transcript: str) -> None:
        if not transcript:
            return
        if self._trigger_prompt and transcript.strip() == self._trigger_prompt:
            print(
                f"[REALTIME] ignore transcript matches trigger prompt lang={self.lang} "
                f"transcript={transcript!r}"
            )
            return
        self._append_history("user", transcript)
        triggered = self._always_respond or self._contains_trigger_phrase(transcript)
        if self._trigger_debug and not triggered and not self._always_respond:
            print(
                f"[REALTIME] trigger miss lang={self.lang} "
                f"transcript={transcript!r} normalized={self._normalize_text(transcript)!r} "
                f"triggers={self._trigger_phrases}"
            )
        if self._response_in_flight:
            if triggered:
                log_label = "trigger detected (deferred)" if not self._always_respond else "auto response (deferred)"
                self._set_pending_response(transcript, log_label)
            return
        if not triggered:
            return
        log_label = "trigger detected" if not self._always_respond else "auto response"
        await self._send_response(transcript, log_label=log_label, force=False)

    def _normalize_text(self, text: str) -> str:
        cleaned = text.lower()
        # Convert Katakana to Hiragana (U+30A1–U+30F6 -> U+3041–U+3096)
        converted = []
        for ch in cleaned:
            code = ord(ch)
            if 0x30A1 <= code <= 0x30F6:
                converted.append(chr(code - 0x60))
            else:
                converted.append(ch)
        cleaned = "".join(converted)
        for ch in [
            " ", "\t", "\n", "\r",
            ".", ",", "!", "?", "。", "、", "！", "？",
            "…", "‥", "・", "ー", "－", "—", "〜", "～",
        ]:
            cleaned = cleaned.replace(ch, "")
        return cleaned

    def _contains_trigger_phrase(self, text: str) -> bool:
        normalized = self._normalize_text(text)
        for phrase, norm in zip(self._trigger_phrases, self._trigger_norm, strict=False):
            if not norm:
                continue
            # Raw contains OR normalized contains (both directions to tolerate shorter utterances)
            if phrase in text or norm in normalized or normalized in norm:
                return True
        return False

    def _append_history(self, role: str, text: str) -> None:
        if not text:
            return
        self._history.append({"role": role, "text": text})
        if self._history_max_turns <= 0:
            return
        if len(self._history) > self._history_max_turns:
            self._history = self._history[-self._history_max_turns :]

    def _build_history_messages(self) -> list[dict[str, str]]:
        if not self._history:
            return []
        return list(self._history)

    def _build_history_summary(self) -> str:
        if not self._history:
            return ""
        history = self._history
        if history and history[-1].get("role") == "user":
            history = history[:-1]
        if not history:
            return ""

        limit = self._summary_max_chars
        if limit <= 0:
            limit = 800

        parts: list[str] = []
        total = 0
        for item in reversed(history):
            role = item.get("role")
            text = (item.get("text") or "").replace("\n", " ").strip()
            if not text:
                continue
            if len(text) > 160:
                text = text[:160].rstrip() + "..."
            if role == "user":
                prefix = "사용자" if self.lang == "ko" else "ユーザー"
            elif role == "assistant":
                prefix = "어시스턴트" if self.lang == "ko" else "アシスタント"
            else:
                prefix = "시스템" if self.lang == "ko" else "システム"
            segment = f"{prefix}: {text}"
            sep = " / " if parts else ""
            if total + len(sep) + len(segment) > limit:
                break
            parts.append(segment)
            total += len(sep) + len(segment)

        parts.reverse()
        return " / ".join(parts)

    def _build_system_prompt(self, summary: str) -> str:
        if self.lang == "ko":
            base = (
                "너는 음성 비서다. 아래 대화 요약을 참고해 사용자의 최신 발화를 간단히 정리하고 "
                "실용적인 조언을 제공하라. 반드시 한국어로만 답하라."
            )
            if not summary:
                return base + " 대화 요약: (없음)"
            return base + f" 대화 요약: {summary}"
        base = (
            "あなたは音声アシスタントです。以下の会話要約を参考に、ユーザーの最新発話を簡潔に整理し、"
            "実用的な助言を提供してください。日本語のみで回答してください。"
        )
        if not summary:
            return base + " 会話要約: (なし)"
        return base + f" 会話要約: {summary}"

    def set_session_id(self, session_id: Optional[str]) -> None:
        if session_id:
            self._session_id = session_id

    async def _resolve_live_session_id(self, session) -> Optional[str]:
        if self._session_id:
            return self._session_id
        result = await session.execute(
            select(RoomLiveSession)
            .where(
                RoomLiveSession.room_id == self.room_id,
                RoomLiveSession.status == "active",
            )
            .order_by(RoomLiveSession.started_at.desc())
        )
        live_session = result.scalars().first()
        if not live_session:
            return None
        self._session_id = live_session.id
        return live_session.id

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
                session_id = await self._resolve_live_session_id(session)
                if not session_id:
                    print(f"[STT] save skipped room_id={self.room_id} reason=no_active_session")
                    return
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
                    new_member = RoomMember(
                        id=f"member_{uuid.uuid4().hex[:16]}",
                        room_id=self.room_id,
                        user_id=speaker_id,
                        display_name=self._last_speaker_name or "Guest",
                        role="member",
                        joined_at=datetime.utcnow(),
                        client_meta={
                            "source": "realtime_agent",
                            "speaker_identity": speaker_id,
                        },
                    )
                    session.add(new_member)
                    try:
                        await session.commit()
                        member_id = new_member.id
                        self._member_cache[speaker_id] = member_id
                        print(
                            "🧾 [STT] created room_member "
                            f"room_id={self.room_id} member_id={member_id} speaker_id={speaker_id}"
                        )
                    except IntegrityError:
                        await session.rollback()
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
                        print(
                            f"[STT] save skipped room_id={self.room_id} "
                            f"reason=no_member speaker_id={speaker_id}"
                        )
                        return

                source_lang_code = normalize_lang(self._last_speaker_lang) or normalize_lang(self.lang)
                target_lang_code = opposite_lang_code(source_lang_code)
                translated_text = None
                translated_lang = None
                if source_lang_code and target_lang_code:
                    source_lang_name = lang_code_to_name(source_lang_code)
                    target_lang_name = lang_code_to_name(target_lang_code)
                    if source_lang_name and target_lang_name:
                        try:
                            if deepl_service.enabled:
                                translated_text = deepl_service.translate_text(
                                    text=transcript,
                                    source_lang=source_lang_name,
                                    target_lang=target_lang_name,
                                )
                            if looks_like_mock_translation(translated_text):
                                translated_text = await openai_service.translate_text(
                                    text=transcript,
                                    source_lang=source_lang_name,
                                    target_lang=target_lang_name,
                                )
                            if translated_text:
                                translated_lang = target_lang_code
                        except Exception as exc:
                            print(f"[STT] translate failed room_id={self.room_id} err={exc!r}")

                for _ in range(3):
                    seq_result = await session.execute(
                        select(func.max(RoomSttResult.seq)).where(RoomSttResult.session_id == session_id)
                    )
                    max_seq = seq_result.scalar() or 0
                    next_seq = max_seq + 1
                    stt_id = f"stt_{uuid.uuid4().hex[:16]}"
                    stt_result = RoomSttResult(
                        id=stt_id,
                        room_id=self.room_id,
                        session_id=session_id,
                        member_id=member_id,
                        user_lang=self._last_speaker_lang or self.lang,
                        stt_text=transcript,
                        translated_text=translated_text,
                        translated_lang=translated_lang,
                        seq=next_seq,
                        meta={
                            "speaker_identity": speaker_id,
                            "speaker_name": self._last_speaker_name,
                            "speaker_lang": self._last_speaker_lang,
                            "session_lang": self.lang,
                        },
                        created_at=datetime.utcnow(),
                    )
                    session.add(stt_result)
                    try:
                        await session.commit()
                        self._last_stt_seq = next_seq
                        self._last_stt_text = transcript
                        print(
                            f"{ALIEN_STAMP} 🧾 [STT] saved room_stt_results "
                            f"room_id={self.room_id} session_id={session_id} "
                            f"seq={next_seq} member_id={member_id} lang={self.lang} "
                            f"text={transcript!r} translated={translated_text!r}"
                        )
                        await self._publish_stt_event(
                            room_id=self.room_id,
                            message={
                                "type": "stt",
                                "data": {
                                    "id": stt_id,
                                    "room_id": self.room_id,
                                    "session_id": session_id,
                                    "seq": next_seq,
                                    "user_id": speaker_id,
                                    "display_name": self._last_speaker_name or "Guest",
                                    "text": transcript,
                                    "lang": self._last_speaker_lang or self.lang,
                                    "translated_text": translated_text,
                                    "translated_lang": translated_lang,
                                    "is_final": True,
                                    "created_at": to_jst_iso(datetime.utcnow()),
                                },
                            },
                        )
                        return
                    except IntegrityError:
                        await session.rollback()
                        continue
        except Exception as exc:
            print(f"[STT] save failed room_id={self.room_id} err={exc!r}")

    async def _publish_stt_event(self, room_id: str, message: dict) -> None:
        payload = {
            "room_id": room_id,
            "message": message,
        }
        redis = None
        try:
            redis = aioredis.from_url(self._redis_url, encoding="utf-8", decode_responses=True)
            await redis.publish(self._stt_channel, json.dumps(payload, ensure_ascii=False))
        except Exception as exc:
            print(f"[STT] publish failed room_id={room_id} err={exc!r}")
        finally:
            if redis:
                try:
                    await redis.close()
                except Exception:
                    pass

    async def _save_ai_response(self, text: str) -> None:
        if not text:
            return
        try:
            async with AsyncSessionLocal() as session:
                session_id = await self._resolve_live_session_id(session)
                if not session_id:
                    print(f"[AI] save skipped room_id={self.room_id} reason=no_active_session")
                    return

                stt_seq_end = self._last_stt_seq
                stt_text = self._last_stt_text
                if stt_seq_end is None or not stt_text:
                    last_result = await session.execute(
                        select(RoomSttResult)
                        .where(RoomSttResult.session_id == session_id)
                        .order_by(RoomSttResult.seq.desc())
                        .limit(1)
                    )
                    last_stt = last_result.scalars().first()
                    if last_stt:
                        stt_seq_end = last_stt.seq
                        stt_text = last_stt.stt_text
                if stt_seq_end is None or not stt_text:
                    print(f"[AI] save skipped room_id={self.room_id} reason=no_stt_anchor")
                    return

                response_id = f"air_{uuid.uuid4().hex[:16]}"
                ai_response = RoomAiResponse(
                    id=response_id,
                    room_id=self.room_id,
                    session_id=session_id,
                    lang=self.lang,
                    stt_text=stt_text,
                    stt_seq_end=stt_seq_end,
                    answer_text=text,
                    meta={
                        "session_lang": self.lang,
                        "reply_to_speaker_identity": self._last_speaker_identity,
                        "reply_to_speaker_name": self._last_speaker_name,
                        "reply_to_speaker_lang": self._last_speaker_lang,
                    },
                    created_at=datetime.utcnow(),
                )
                session.add(ai_response)
                await session.commit()
                print(
                    "🧾 [AI] saved room_ai_responses "
                    f"room_id={self.room_id} session_id={session_id} "
                    f"stt_seq_end={stt_seq_end} lang={self.lang}"
                )
        except Exception as exc:
            print(f"[AI] save failed room_id={self.room_id} err={exc!r}")

    async def _push_audio(self, pcm16_24k: bytes) -> None:
        if not pcm16_24k:
            return
        if self.output_source.sample_rate != REALTIME_SAMPLE_RATE:
            pcm16_24k, self._out_state = pcm16_resample(
                pcm16_24k,
                from_rate=REALTIME_SAMPLE_RATE,
                to_rate=self.output_source.sample_rate,
                state=self._out_state,
            )
        self._out_buffer.extend(pcm16_24k)
        await self._flush_output()

    async def _flush_output(self) -> None:
        frame_ms = 20
        samples = int(self.output_source.sample_rate * frame_ms / 1000)
        frame_bytes = samples * self.output_source.num_channels * 2
        while len(self._out_buffer) >= frame_bytes:
            chunk = bytes(self._out_buffer[:frame_bytes])
            del self._out_buffer[:frame_bytes]
            frame = rtc.AudioFrame(
                data=chunk,
                sample_rate=self.output_source.sample_rate,
                num_channels=self.output_source.num_channels,
                samples_per_channel=samples,
            )
            await self.output_source.capture_frame(frame)


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
        await asyncio.sleep(0.2)
        async with self._lock:
            self._apply_permissions(reason)

    def _allowed_for_lang(self, lang: Optional[str]) -> list[str]:
        if lang == "ko":
            return [self.ko_sid]
        if lang == "ja":
            return [self.ja_sid]
        if self.unknown_policy == "ko":
            return [self.ko_sid]
        if self.unknown_policy == "ja":
            return [self.ja_sid]
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
        perms = []
        for participant in self.room.remote_participants.values():
            lang = None
            try:
                lang = normalize_lang((participant.attributes or {}).get("lang"))
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


def resolve_target_langs(participant_lang: Optional[str], unknown_policy: str) -> set[str]:
    lang = normalize_lang(participant_lang)
    if lang in {"ko", "ja"}:
        return {lang}
    return set()


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
    last_empty_log = 0.0
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

            target_langs = resolve_target_langs(participant_lang, unknown_policy)
            state.active_langs = target_langs
            if not target_langs:
                now = time.time()
                if now - last_empty_log >= 5.0:
                    print(f"[AUDIO] {label} no active_langs (unknown_policy={unknown_policy})")
                    last_empty_log = now

            if "ko" in target_langs and state.realtime_ko:
                state.realtime_ko.note_speaker(participant_identity, participant_name, participant_lang)
                state.realtime_ko.send_audio(data)
            if "ja" in target_langs and state.realtime_ja:
                state.realtime_ja.note_speaker(participant_identity, participant_name, participant_lang)
                state.realtime_ja.send_audio(data)

            frames += 1
            now = time.time()
            if now - last_report >= 5.0:
                fps = frames / (now - last_report)
                print(f"[AUDIO] {label} fps={fps:.1f} active_langs={sorted(target_langs)}")
                frames = 0
                last_report = now
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"[AUDIO] {label} stream error: {exc!r}")
    finally:
        await stream.aclose()


async def publish_output_track(
    room: rtc.Room,
    *,
    track_name: str,
) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication]:
    source = rtc.AudioSource(sample_rate=LIVEKIT_SAMPLE_RATE, num_channels=1)
    track = rtc.LocalAudioTrack.create_audio_track(track_name, source)

    opts = rtc.TrackPublishOptions()
    opts.source = rtc.TrackSource.SOURCE_MICROPHONE

    pub = await room.local_participant.publish_track(track, opts)
    print(f"[PUBLISH] track_name={track_name} sid={pub.sid}")
    return source, pub


async def publish_output_track_with_retry(
    room: rtc.Room,
    *,
    track_name: str,
    retry_seconds: float,
    max_attempts: int,
) -> tuple[rtc.AudioSource, rtc.LocalTrackPublication]:
    attempt = 0
    last_exc: Optional[Exception] = None
    while True:
        attempt += 1
        try:
            return await publish_output_track(room, track_name=track_name)
        except Exception as exc:
            last_exc = exc
            print(
                f"[PUBLISH] failed track_name={track_name} "
                f"attempt={attempt} error={exc!r}"
            )
            if max_attempts and attempt >= max_attempts:
                break
            await asyncio.sleep(retry_seconds)
    raise RuntimeError(f"publish_output_track failed after {attempt} attempts") from last_exc


async def connect_room(
    room_id: str,
    session_id: Optional[str],
    auth: AuthState,
    auto_subscribe: bool,
    rooms: dict[str, RoomState],
    retry_seconds: float,
    max_attempts: int,
    ko_track: str,
    ja_track: str,
    unknown_policy: str,
    realtime_model: str,
    realtime_url: str,
    realtime_key: str,
    voice_ko: str,
    voice_ja: str,
    transcribe_model: str,
    output_modalities: list[str],
    trigger_phrases_ko: list[str],
    trigger_phrases_ja: list[str],
    wake_cooldown_s: float,
    vad_threshold: float,
    vad_prefix_ms: int,
    vad_silence_ms: int,
    always_respond: bool,
    history_max_turns: int,
    summary_max_chars: int,
    save_stt: bool,
    trigger_debug: bool,
    redis_url: str,
    stt_channel: str,
    force_commit_ms: int,
) -> None:
    if room_id in rooms:
        state = rooms.get(room_id)
        if state and session_id:
            state.session_id = session_id
            if state.realtime_ko:
                state.realtime_ko.set_session_id(session_id)
            if state.realtime_ja:
                state.realtime_ja.set_session_id(session_id)
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
    state.session_id = session_id
    rooms[room_id] = state

    @room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        lang = normalize_lang((participant.attributes or {}).get("lang")) or "unknown"
        print(
            "🟢👤 [ROOM] participant_connected "
            f"room_id={room_id} identity={participant.identity} "
            f"name={participant.name} lang={lang} attrs={participant.attributes}"
        )
        if state.router:
            state.router.schedule_recompute("participant_connected")
            asyncio.create_task(_delayed_recompute(state.router, "participant_connected_delayed"))
        if state.empty_check_task and not state.empty_check_task.done():
            state.empty_check_task.cancel()
        state.empty_check_task = None

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"🔴👤 [ROOM] participant_disconnected room_id={room_id} identity={participant.identity}")
        if state.router:
            state.router.schedule_recompute("participant_disconnected")
        if state.empty_check_task and not state.empty_check_task.done():
            state.empty_check_task.cancel()
        state.empty_check_task = asyncio.create_task(_disconnect_if_empty(room_id, rooms))

    @room.on("participant_attributes_changed")
    def _on_participant_attributes_changed(changed: dict, participant: rtc.Participant):
        print(
            "[ROOM] participant_attributes_changed "
            f"room_id={room_id} identity={participant.identity} changed={changed}"
        )
        if state.router and "lang" in changed:
            state.router.schedule_recompute("participant_attributes_changed")

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        lang = normalize_lang((participant.attributes or {}).get("lang")) or "unknown"
        print(
            "📡🎧 [ROOM] track_subscribed "
            f"room_id={room_id} kind={track.kind} participant={participant.identity} "
            f"lang={lang} pub_sid={publication.sid} track_sid={track.sid}"
        )
        if track.kind == rtc.TrackKind.KIND_AUDIO:
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
    print(f"🤖🚪 [AGENT] joined room_id={room_id} room={room.name}")

    publish_retry_seconds = float(os.getenv("LIVEKIT_PUBLISH_RETRY_SECONDS", "1.0"))
    publish_max_attempts = int(os.getenv("LIVEKIT_PUBLISH_MAX_ATTEMPTS", "3"))
    try:
        ko_source, ko_pub = await publish_output_track_with_retry(
            room,
            track_name=ko_track,
            retry_seconds=publish_retry_seconds,
            max_attempts=publish_max_attempts,
        )
        ja_source, ja_pub = await publish_output_track_with_retry(
            room,
            track_name=ja_track,
            retry_seconds=publish_retry_seconds,
            max_attempts=publish_max_attempts,
        )
    except Exception as exc:
        print(f"[PUBLISH] abort room_id={room_id} error={exc!r}")
        await disconnect_room(room_id, rooms)
        return

    state.ko_pub_sid = ko_pub.sid
    state.ja_pub_sid = ja_pub.sid
    state.router = LangRouter(
        room,
        ko_sid=ko_pub.sid,
        ja_sid=ja_pub.sid,
        unknown_policy=unknown_policy,
    )
    await state.router.apply_now("initial")

    state.realtime_ko = RealtimeSession(
        lang="ko",
        room_id=room_id,
        session_id=session_id,
        api_key=realtime_key,
        model=realtime_model,
        base_url=realtime_url,
        transcribe_model=transcribe_model,
        trigger_phrases=trigger_phrases_ko,
        wake_cooldown_s=wake_cooldown_s,
        output_source=ko_source,
        voice=voice_ko,
        output_modalities=output_modalities,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        always_respond=always_respond,
        history_max_turns=history_max_turns,
        summary_max_chars=summary_max_chars,
        save_stt=save_stt,
        trigger_debug=trigger_debug,
        redis_url=redis_url,
        stt_channel=stt_channel,
        force_commit_ms=force_commit_ms,
    )
    state.realtime_ja = RealtimeSession(
        lang="ja",
        room_id=room_id,
        session_id=session_id,
        api_key=realtime_key,
        model=realtime_model,
        base_url=realtime_url,
        transcribe_model=transcribe_model,
        trigger_phrases=trigger_phrases_ja,
        wake_cooldown_s=wake_cooldown_s,
        output_source=ja_source,
        voice=voice_ja,
        output_modalities=output_modalities,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        always_respond=always_respond,
        history_max_turns=history_max_turns,
        summary_max_chars=summary_max_chars,
        save_stt=save_stt,
        trigger_debug=trigger_debug,
        redis_url=redis_url,
        stt_channel=stt_channel,
        force_commit_ms=force_commit_ms,
    )

    await asyncio.gather(state.realtime_ko.start(), state.realtime_ja.start())
    print(f"🤖🇰🇷 [AGENT] ready lang=ko room_id={room_id} track={ko_track}")
    print(f"🤖🇯🇵 [AGENT] ready lang=ja room_id={room_id} track={ja_track}")
    print("🚀🚀🚀 OPENAI 시작! 🤖🤖🤖")


async def _delayed_recompute(router: LangRouter, reason: str) -> None:
    await asyncio.sleep(0.8)
    router.schedule_recompute(reason)


async def _disconnect_if_empty(room_id: str, rooms: dict[str, RoomState]) -> None:
    await asyncio.sleep(0.6)
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
    stt_channel: str,
    auth: AuthState,
    rooms: dict[str, RoomState],
    auto_subscribe: bool,
    retry_seconds: float,
    max_attempts: int,
    ko_track: str,
    ja_track: str,
    unknown_policy: str,
    realtime_model: str,
    realtime_url: str,
    realtime_key: str,
    voice_ko: str,
    voice_ja: str,
    transcribe_model: str,
    output_modalities: list[str],
    trigger_phrases_ko: list[str],
    trigger_phrases_ja: list[str],
    wake_cooldown_s: float,
    vad_threshold: float,
    vad_prefix_ms: int,
    vad_silence_ms: int,
    force_commit_ms: int,
    always_respond: bool,
    history_max_turns: int,
    summary_max_chars: int,
    save_stt: bool,
    trigger_debug: bool,
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
            session_id = data.get("session_id")
            if not room_id:
                continue
            if action == "join":
                print(f"📥🟢 [EVENT] action=join room_id={room_id}")
                try:
                    await connect_room(
                        room_id=room_id,
                        session_id=session_id,
                        auth=auth,
                        auto_subscribe=auto_subscribe,
                        rooms=rooms,
                        retry_seconds=retry_seconds,
                        max_attempts=max_attempts,
                        ko_track=ko_track,
                        ja_track=ja_track,
                        unknown_policy=unknown_policy,
                        realtime_model=realtime_model,
                        realtime_url=realtime_url,
                        realtime_key=realtime_key,
                        voice_ko=voice_ko,
                        voice_ja=voice_ja,
                        transcribe_model=transcribe_model,
                        output_modalities=output_modalities,
                        trigger_phrases_ko=trigger_phrases_ko,
                        trigger_phrases_ja=trigger_phrases_ja,
                        wake_cooldown_s=wake_cooldown_s,
                        vad_threshold=vad_threshold,
                        vad_prefix_ms=vad_prefix_ms,
                        vad_silence_ms=vad_silence_ms,
                        force_commit_ms=force_commit_ms,
                        always_respond=always_respond,
                        history_max_turns=history_max_turns,
                        summary_max_chars=summary_max_chars,
                        save_stt=save_stt,
                        trigger_debug=trigger_debug,
                        redis_url=redis_url,
                        stt_channel=stt_channel,
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
    parser.add_argument("--backend", required=False, help="예: http://localhost:8000")
    parser.add_argument("--room", required=False, help="room_id (선택)")
    parser.add_argument("--auto-subscribe", default=None, choices=["true", "false"])
    args = parser.parse_args()

    backend = args.backend or os.getenv("BACKEND_URL")
    room_id = args.room or os.getenv("ROOM_ID")
    service_auth = normalize_service_auth(os.getenv("SERVICE_AUTH"))
    worker_key = os.getenv("WORKER_SERVICE_KEY")
    worker_id = os.getenv("WORKER_ID", "livekit_worker")
    worker_ttl = int(os.getenv("WORKER_TOKEN_TTL_SECONDS", "0"))
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    channel = os.getenv("LIVEKIT_ROOM_EVENTS_CHANNEL", "livekit:rooms")
    stt_channel = os.getenv("LIVEKIT_STT_EVENTS_CHANNEL", "livekit:stt")
    force_relay_value = os.getenv("LIVEKIT_FORCE_RELAY", "false")
    force_relay = force_relay_value.lower() in {"1", "true", "yes", "y", "on"}

    ko_track = os.getenv("LIVEKIT_KO_TRACK", "lk.out.ko")
    ja_track = os.getenv("LIVEKIT_JA_TRACK", "lk.out.ja")
    unknown_policy = os.getenv("LIVEKIT_UNKNOWN_LANG_POLICY", "both").lower()
    realtime_key = os.getenv("OPENAI_API_KEY")
    realtime_model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime")
    realtime_url = os.getenv("OPENAI_REALTIME_URL", "wss://api.openai.com/v1/realtime")
    default_voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    voice_ko = os.getenv("OPENAI_REALTIME_VOICE_KO", default_voice)
    voice_ja = os.getenv("OPENAI_REALTIME_VOICE_JA", default_voice)
    transcribe_model = os.getenv("OPENAI_REALTIME_TRANSCRIBE_MODEL") or "gpt-4o-mini-transcribe"
    output_modalities_raw = os.getenv("OPENAI_REALTIME_OUTPUT_MODALITIES", "audio")
    output_modalities = [
        part.strip().lower()
        for part in output_modalities_raw.split(",")
        if part.strip()
    ]
    output_modalities = [m for m in output_modalities if m in {"audio", "text"}]
    if not output_modalities:
        output_modalities = ["audio"]
    if "audio" in output_modalities and "text" in output_modalities:
        allow_both_value = os.getenv("OPENAI_REALTIME_ALLOW_BOTH_MODALITIES", "false")
        allow_both = allow_both_value.lower() in {"1", "true", "yes", "y", "on"}
        if not allow_both:
            print(
                "[REALTIME] output_modalities includes both audio+text; "
                "fallback to audio only to avoid API error "
                "(set OPENAI_REALTIME_ALLOW_BOTH_MODALITIES=true to allow both)"
            )
            output_modalities = ["audio"]
    fallback_trigger_raw = os.getenv(
        "OPENAI_TRIGGER_PHRASES",
        "우리토모는 어떻게 생각해?,ウリトモはどう思ってる？",
    )
    trigger_ko_raw = os.getenv("OPENAI_TRIGGER_PHRASES_KO", fallback_trigger_raw)
    trigger_ja_raw = os.getenv("OPENAI_TRIGGER_PHRASES_JA", fallback_trigger_raw)
    trigger_phrases_ko = [
        part.strip()
        for part in re.split(r"[,\n、，]+", trigger_ko_raw)
        if part.strip()
    ]
    trigger_phrases_ja = [
        part.strip()
        for part in re.split(r"[,\n、，]+", trigger_ja_raw)
        if part.strip()
    ]
    wake_cooldown_raw = os.getenv("OPENAI_WAKE_COOLDOWN_SECONDS")
    wake_cooldown_s = float(wake_cooldown_raw or "2.0")
    always_respond_value = os.getenv("OPENAI_ALWAYS_RESPOND", "false")
    always_respond = always_respond_value.lower() in {"1", "true", "yes", "y", "on"}
    if always_respond and wake_cooldown_raw is None:
        wake_cooldown_s = 0.0
    vad_threshold = float(os.getenv("OPENAI_REALTIME_VAD_THRESHOLD", "0.5"))
    vad_prefix_ms = int(os.getenv("OPENAI_REALTIME_VAD_PREFIX_MS", "300"))
    vad_silence_ms = int(os.getenv("OPENAI_REALTIME_VAD_SILENCE_MS", "500"))
    force_commit_ms = int(os.getenv("OPENAI_REALTIME_FORCE_COMMIT_MS", "0"))

    if not backend:
        raise RuntimeError("Missing backend. Provide --backend or env BACKEND_URL")
    if not realtime_key:
        raise RuntimeError("Missing OPENAI_API_KEY")
    history_max_turns = int(os.getenv("OPENAI_HISTORY_MAX_TURNS", "0"))
    summary_max_chars = int(os.getenv("OPENAI_HISTORY_SUMMARY_MAX_CHARS", "800"))
    save_stt_value = os.getenv("OPENAI_STT_SAVE", "true")
    save_stt = save_stt_value.lower() in {"1", "true", "yes", "y", "on"}
    print(
        f"[BOOT] stt_config save_stt={save_stt} "
        f"transcribe_model={transcribe_model} output_modalities={output_modalities} "
        f"vad_threshold={vad_threshold} force_commit_ms={force_commit_ms}"
    )
    trigger_debug_value = os.getenv("OPENAI_TRIGGER_DEBUG", "false")
    trigger_debug = trigger_debug_value.lower() in {"1", "true", "yes", "y", "on"}

    if trigger_debug:
        print(
            "[REALTIME] trigger phrases loaded "
            f"ko={trigger_phrases_ko} ja={trigger_phrases_ja}"
        )

    if not always_respond and not (trigger_phrases_ko or trigger_phrases_ja):
        raise RuntimeError("OPENAI_TRIGGER_PHRASES_KO/JA are empty")

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
            session_id=None,
            auth=auth,
            auto_subscribe=auto_subscribe,
            rooms=rooms,
            retry_seconds=retry_seconds,
            max_attempts=max_attempts,
            ko_track=ko_track,
            ja_track=ja_track,
            unknown_policy=unknown_policy,
            realtime_model=realtime_model,
            realtime_url=realtime_url,
            realtime_key=realtime_key,
            voice_ko=voice_ko,
            voice_ja=voice_ja,
            transcribe_model=transcribe_model,
            output_modalities=output_modalities,
            trigger_phrases_ko=trigger_phrases_ko,
            trigger_phrases_ja=trigger_phrases_ja,
            wake_cooldown_s=wake_cooldown_s,
            vad_threshold=vad_threshold,
            vad_prefix_ms=vad_prefix_ms,
            vad_silence_ms=vad_silence_ms,
            force_commit_ms=force_commit_ms,
            always_respond=always_respond,
            history_max_turns=history_max_turns,
            summary_max_chars=summary_max_chars,
            save_stt=save_stt,
            trigger_debug=trigger_debug,
            redis_url=redis_url,
            stt_channel=stt_channel,
        )

    await listen_room_events(
        redis_url=redis_url,
        channel=channel,
        stt_channel=stt_channel,
        auth=auth,
        rooms=rooms,
        auto_subscribe=auto_subscribe,
        retry_seconds=retry_seconds,
        max_attempts=max_attempts,
        ko_track=ko_track,
        ja_track=ja_track,
        unknown_policy=unknown_policy,
        realtime_model=realtime_model,
        realtime_url=realtime_url,
        realtime_key=realtime_key,
        voice_ko=voice_ko,
        voice_ja=voice_ja,
        transcribe_model=transcribe_model,
        output_modalities=output_modalities,
        trigger_phrases_ko=trigger_phrases_ko,
        trigger_phrases_ja=trigger_phrases_ja,
        wake_cooldown_s=wake_cooldown_s,
        vad_threshold=vad_threshold,
        vad_prefix_ms=vad_prefix_ms,
        vad_silence_ms=vad_silence_ms,
        force_commit_ms=force_commit_ms,
        always_respond=always_respond,
        history_max_turns=history_max_turns,
        summary_max_chars=summary_max_chars,
        save_stt=save_stt,
        trigger_debug=trigger_debug,
    )


if __name__ == "__main__":
    asyncio.run(main())
