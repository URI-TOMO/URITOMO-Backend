import uuid
from datetime import datetime
from typing import Optional

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from starlette.websockets import WebSocketState
from sqlalchemy import select

from app.core.token import verify_token
from app.models.room import Room
from app.meeting.ws.manager import manager
from app.meeting.ws.ws_message import handle_chat_message, handle_stt_message

from app.infra.db import AsyncSessionLocal

router = APIRouter(prefix="/meeting", tags=["websocket"])
logger = logging.getLogger("uritomo.ws")


@router.websocket("/{room_id}")
async def meeting_websocket(
    websocket: WebSocket,
    room_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for a live meeting session.
    Path: /meeting/{room_id}
    """
    # 1. Authenticate (optional but recommended for chat)
    user_id = None
    if token:
        user_id = verify_token(token)
    if token and user_id:
        logger.info(f"💬 CHAT WS Attempt | Room: {room_id} | User: {user_id}")
        print(f"💬 CHAT WS Attempt | Room: {room_id} | User: {user_id}", flush=True)
    elif token and not user_id:
        logger.info(f"⚠️ WS Auth Failed | Room: {room_id} | Token: provided")
        print(f"⚠️ WS Auth Failed | Room: {room_id} | Token: provided", flush=True)
    else:
        logger.info(f"🔌 WS Attempt | Room: {room_id} | User: None")
        print(f"🔌 WS Attempt | Room: {room_id} | User: None", flush=True)
    
    # 2. Check if room exists
    async with AsyncSessionLocal() as db_session:
        result = await db_session.execute(
            select(Room).where(Room.id == room_id)
        )
        room = result.scalar_one_or_none()

        if not room:
            logger.info(f"🚫 WS Room Not Found | Room: {room_id} | User: {user_id}")
            # Accept first so we can send a concrete error reason.
            await websocket.accept()
            await websocket.send_json({
                "type": "error",
                "code": "ROOM_NOT_FOUND",
                "message": "ルームが存在しません。room_idを確認してください。"
            })
            await websocket.close(code=1008) # Policy Violation
            return

    # 3. Handle connection via manager
    await manager.connect(room_id, websocket, user_id)
    logger.info(f"✅ WS Connected | Room: {room_id} | User: {user_id}")
    print(f"✅ WS Connected | Room: {room_id} | User: {user_id}", flush=True)
    
    try:
        # Send initial success message
        await websocket.send_json({
            "type": "room_connected",
            "data": {
                "room_id": room_id,
                "user_id": user_id
            }
        })
        logger.info(f"📤 WS Sent room_connected | Room: {room_id} | User: {user_id}")

        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json({
                        "type": "error",
                        "code": "INVALID_JSON",
                        "message": "無効なJSON形式です。送信内容を確認してください。"
                    })
                continue
            
            msg_type = data.get("type")
            logger.info(f"📥 WS Received | Room: {room_id} | User: {user_id} | Type: {msg_type} | Data: {data}")
            print(f"📥 WS Received | Room: {room_id} | User: {user_id} | Type: {msg_type}", flush=True)
            
            if msg_type == "chat":
                if not user_id:
                    logger.warning(f"⚠️ WS Chat Auth Required | Room: {room_id}")
                    print(f"⚠️ WS Chat Auth Required | Room: {room_id}", flush=True)
                    await websocket.send_json({
                        "type": "error",
                        "code": "AUTH_REQUIRED",
                        "message": "チャット送信には認証トークンが必要です。"
                    })
                    continue
                
                logger.info(f"💬 WS Chat Processing | Room: {room_id} | User: {user_id} | Text: {data.get('text', '')[:50]}")
                await handle_chat_message(room_id, user_id, data)
            
            elif msg_type == "stt":
                if not user_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Authentication required for STT"
                    })
                    continue
                await handle_stt_message(session_id, user_id, data)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                # Default echo or unknown type
                await websocket.send_json({
                    "type": "unknown_type",
                    "received": data
                })

    except WebSocketDisconnect:
        logger.info(f"🔌 WS Disconnected | Room: {room_id} | User: {user_id}")
        manager.disconnect(room_id, websocket, user_id)
    except Exception as e:
        logger.error(f"❌ WS Error | Room: {room_id} | User: {user_id} | Error: {e}")
        manager.disconnect(room_id, websocket, user_id)
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json({
                    "type": "error",
                    "code": "INTERNAL_ERROR",
                    "message": "サーバー内部エラーが発生しました。しばらくしてから再接続してください。"
                })
                await websocket.close(code=1011)
        except:
            pass
