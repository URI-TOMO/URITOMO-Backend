"""
Emoji Logging System (Project-agnostic)
======================================

Project Common Logging Guide
----------------------------
- One-line summary only: every log line is a compact, human-readable single line.
- Format (fixed, required):
  {time} {sev_emoji} {LEVEL:<5} {domain_emoji} {event:<18} {kv_pairs} | {summary}
- time: default "HH:MM:SS" (ISO8601 optional).
- event: dotted hierarchy {domain}.{action}[.{detail}] (examples: ws.connected, trans.fallback).
- kv_pairs: space-separated key=value (room_id/user_id/session_id/seq/latency_ms/provider/trace_id).
- summary: short sentence; include text only as preview (max 40 chars + "...").
- Emojis: 1-2 per line only (severity + domain).
- Payloads: never dump full payload at INFO; DEBUG only or previewed.

Event Naming Rules
-----------------
- {domain}.{action}[.{detail}] (dotted hierarchy).
- Examples:
  ws.connected, ws.disconnected, ws.invalid_json
  chat.received, chat.saved, chat.broadcast
  trans.start, trans.fallback, trans.ok, trans.fail
  stt.redis.recv, stt.broadcast
  db.query, db.commit, db.rollback
  app.startup, app.shutdown

Log Level Policy
----------------
- DEBUG: full payloads, verbose diagnostic data.
- INFO : state changes + key events (connect, start/ok/fail, broadcast summary).
- WARN : fallbacks/retries/abnormal but service continues.
- ERROR: exceptions with correlation keys only (no payload/body).
- CRITICAL: service cannot continue.

Correlation Keys (include when available)
-----------------------------------------
- request_id / trace_id / span_id (HTTP = strongly recommended)
- room_id or session_id or user_id (realtime/WS = strongly recommended)
- seq (message order)
- latency_ms (external calls/translation/DB/AI)
- provider (deepl/openai/etc)

Domain Emoji Map (25+)
----------------------
- websocket/connectivity: 🔌
- broadcast/send: 📣
- inbound/receive: 📥
- outbound/publish: 📤
- stt/audio: 🎧
- microphone/audio capture: 🎙️
- translation/i18n: 🌐
- ai/model: 🤖
- inference/agent: 🧠
- redis/cache: 🧰
- db/sql: 🧱
- migration/schema: 🧾
- search/query: 🔍
- latency/timing: ⏱️
- worker/background: 🧵
- queue/job: 🧲
- routing/api: 🧭
- auth/security: 🔐
- test/verification: 🧪
- cleanup/shutdown: 🧹
- exception/rollback: 🧯
- rate-limit/guard: 🚦
- file/storage: 🗂️
- external api/http: 🛰️
- deploy/build: 📦

Noise-Reduction Rules (5)
-------------------------
1) INFO never prints raw payloads or long JSON; use preview only.
2) Preview max 40 chars for any text; suffix with "..." if truncated.
3) Drop kv pairs with None/empty values; keep only keys that help triage.
4) Avoid duplicate logs for the same event; log once per state change.
5) Stack traces only in ERROR/CRITICAL (or DEBUG); INFO/WARN are one-line.

Abstract Design (Library-agnostic)
----------------------------------
Emitter -> Event -> Formatter -> Sink
1) Emitter creates a LogEvent with level/domain/event/summary/kv/payload.
2) Formatter enforces the fixed format and payload policy.
3) Sink is any logger (logging/structlog/loguru) or stdout/stderr.

Minimal-Change Refactor Strategy
--------------------------------
- Wrap the existing logger with a small helper (log_event).
- Standardize all modules to pass event + domain + kv + summary.
- Add a single formatter/handler configuration in app startup.
- Gradually replace ad-hoc logging with log_event calls.

Examples (10+)
--------------
12:00:01 ℹ️ INFO  🔌 ws.connected      room_id=room_01 user_id=u_123 | WebSocket connected
12:00:02 ℹ️ INFO  📥 chat.received     room_id=room_01 user_id=u_123 seq=18 | Incoming chat "Hello worl..."
12:00:02 ℹ️ INFO  🧱 db.commit         room_id=room_01 seq=18 | Chat message persisted
12:00:03 ℹ️ INFO  🌐 trans.start       room_id=room_01 seq=18 provider=deepl | Translation started
12:00:03 ⚠️ WARN  🌐 trans.fallback    room_id=room_01 seq=18 provider=deepl | Fallback to openai
12:00:04 ✅ INFO  🌐 trans.ok          room_id=room_01 seq=18 provider=openai latency_ms=412 | Translation ok
12:00:04 ℹ️ INFO  📣 chat.broadcast    room_id=room_01 seq=18 targets=6 | Broadcasted message
12:00:05 ℹ️ INFO  🎧 stt.redis.recv    room_id=room_01 session_id=s_99 seq=77 | STT event received
12:00:05 ℹ️ INFO  📣 stt.broadcast     room_id=room_01 session_id=s_99 seq=77 | STT final broadcast
12:00:06 ❌ ERROR 🧯 ws.invalid_json    room_id=room_01 user_id=u_123 | Invalid JSON from client
12:00:07 ℹ️ INFO  🧵 worker.start      worker_id=livekit_1 | Worker started
12:00:08 💥 CRITICAL 🧱 db.rollback    room_id=room_01 | DB unavailable, shutting down
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

LEVEL_EMOJI = {
    "DEBUG": "🐛",
    "INFO": "ℹ️",
    "WARN": "⚠️",
    "WARNING": "⚠️",
    "ERROR": "❌",
    "CRITICAL": "💥",
}

DOMAIN_EMOJI = {
    "ws": "🔌",
    "broadcast": "📣",
    "inbound": "📥",
    "outbound": "📤",
    "stt": "🎧",
    "mic": "🎙️",
    "trans": "🌐",
    "ai": "🤖",
    "agent": "🧠",
    "redis": "🧰",
    "db": "🧱",
    "migration": "🧾",
    "search": "🔍",
    "latency": "⏱️",
    "worker": "🧵",
    "queue": "🧲",
    "api": "🧭",
    "auth": "🔐",
    "test": "🧪",
    "cleanup": "🧹",
    "exception": "🧯",
    "rate": "🚦",
    "file": "🗂️",
    "external": "🛰️",
    "deploy": "📦",
}

DEFAULT_TIME_FORMAT = "%H:%M:%S"


def preview(text: Optional[str], max_len: int = 40) -> str:
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_time(ts: Optional[datetime], use_iso: bool) -> str:
    ts = ts or datetime.now(timezone.utc)
    if use_iso:
        return ts.isoformat()
    return ts.astimezone().strftime(DEFAULT_TIME_FORMAT)


def _format_kv_pairs(kv: Mapping[str, Any]) -> str:
    parts = []
    for key, value in kv.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    return " ".join(parts) if parts else "-"


@dataclass(frozen=True)
class LogEvent:
    level: str
    domain: str
    event: str
    summary: str
    kv: Mapping[str, Any] = field(default_factory=dict)
    payload: Optional[str] = None
    success: bool = False
    ts: Optional[datetime] = None
    use_iso_time: bool = False


def format_log_line(evt: LogEvent) -> str:
    level = evt.level.upper()
    sev_emoji = "✅" if level == "INFO" and evt.success else LEVEL_EMOJI.get(level, "ℹ️")
    domain_emoji = DOMAIN_EMOJI.get(evt.domain, "🧭")

    summary = evt.summary
    if evt.payload:
        if level == "DEBUG":
            summary = f"{summary} payload={evt.payload}"
        else:
            summary = f"{summary} payload={preview(evt.payload)}"

    time_str = _format_time(evt.ts, evt.use_iso_time)
    kv_pairs = _format_kv_pairs(evt.kv)
    return f"{time_str} {sev_emoji} {level:<5} {domain_emoji} {evt.event:<18} {kv_pairs} | {summary}"

