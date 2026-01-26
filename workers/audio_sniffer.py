import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import httpx
from livekit import rtc


@dataclass
class BackendTokenResponse:
    url: str
    token: str


async def fetch_livekit_token(
    backend_base_url: str,
    room_id: str,
    service_auth_header_value: str,
    timeout_s: float = 10.0,
) -> BackendTokenResponse:
    endpoint = backend_base_url.rstrip("/") + "/livekit/token"
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


async def consume_audio(track: rtc.Track, *, label: str) -> None:
    stream = rtc.AudioStream.from_track(track=track, sample_rate=48000, num_channels=1)

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


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", required=True, help="예: http://localhost:8000")
    parser.add_argument("--room", required=True, help="room_id")
    parser.add_argument(
        "--auth",
        default=None,
        help='Authorization 헤더 값(예: "Bearer xxx"). 미지정 시 env SERVICE_AUTH 사용',
    )
    parser.add_argument("--auto-subscribe", default="true", choices=["true", "false"])
    args = parser.parse_args()

    service_auth = args.auth or os.getenv("SERVICE_AUTH")
    if not service_auth:
        raise RuntimeError('Missing auth. Provide --auth or env SERVICE_AUTH (e.g. "Bearer ...")')

    token_resp = await fetch_livekit_token(
        backend_base_url=args.backend,
        room_id=args.room,
        service_auth_header_value=service_auth,
    )
    print(f"[BOOT] got token. livekit_url={token_resp.url}")

    room = rtc.Room()

    @room.on("participant_connected")
    def _on_participant_connected(participant: rtc.RemoteParticipant):
        print(
            "[ROOM] participant_connected "
            f"identity={participant.identity} name={participant.name} attrs={participant.attributes}"
        )

    @room.on("participant_disconnected")
    def _on_participant_disconnected(participant: rtc.RemoteParticipant):
        print(f"[ROOM] participant_disconnected identity={participant.identity}")

    @room.on("track_subscribed")
    def _on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        print(
            "[ROOM] track_subscribed "
            f"kind={track.kind} participant={participant.identity} "
            f"pub_sid={publication.sid} track_sid={track.sid}"
        )

        if track.kind == rtc.TrackKind.KIND_AUDIO:
            label = f"from={participant.identity} track_sid={track.sid}"
            asyncio.create_task(consume_audio(track, label=label))

    auto_subscribe = args.auto_subscribe.lower() == "true"
    opts = rtc.RoomOptions(auto_subscribe=auto_subscribe)

    print(f"[BOOT] connecting room_id={args.room} auto_subscribe={auto_subscribe}")
    await room.connect(token_resp.url, token_resp.token, options=opts)
    print(f"[BOOT] connected. room={room.name}")

    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())
