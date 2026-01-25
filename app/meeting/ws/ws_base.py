import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from app.core.token import verify_token
from app.models.room import RoomLiveSession
from app.meeting.ws.manager import manager
from app.meeting.ws.ws_message import handle_chat_message

from app.infra.db import AsyncSessionLocal

router = APIRouter(prefix="/meeting", tags=["websocket"])

@router.get("/ws-info", include_in_schema=True)
async def websocket_docs():
    """
    Returns documentation on how to connect to the meeting WebSocket.
    """
    return {
        "websocket_url": "/api/v1/meeting/{session_id}",
        "auth": "pass token as query parameter ?token=...",
        "message_formats": {
            "incoming": {
                "chat": {
                    "type": "chat",
                    "text": "Hello world",
                    "lang": "ja"
                }
            },
            "outgoing": {
                "session_connected": "Sent on successful connection",
                "chat": "Broadcasting chat message to all members",
                "error": "Error details"
            }
        }
    }

@router.websocket("/{session_id}")
async def meeting_websocket(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(None)
):
    """
    WebSocket endpoint for a live meeting session.
    Path: /meeting/{session_id}
    """
    # 1. Authenticate (optional but recommended for chat)
    user_id = None
    if token:
        user_id = verify_token(token)
    
    # 2. Check if session exists
    async with AsyncSessionLocal() as db_session:
        result = await db_session.execute(
            select(RoomLiveSession).where(RoomLiveSession.id == session_id)
        )
        live_session = result.scalar_one_or_none()

        if not live_session:
            await websocket.close(code=1008) # Policy Violation
            return

    # 3. Handle connection via manager
    await manager.connect(session_id, websocket, user_id)
    
    try:
        # Send initial success message
        await websocket.send_json({
            "type": "session_connected",
            "data": {
                "session_id": session_id,
                "user_id": user_id
            }
        })

        while True:
            # Receive message from client
            try:
                data = await websocket.receive_json()
            except Exception:
                # Invalid JSON
                break
            
            msg_type = data.get("type")
            
            if msg_type == "chat":
                if not user_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Authentication required for chat"
                    })
                    continue
                
                await handle_chat_message(session_id, user_id, data)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            else:
                # Default echo or unknown type
                await websocket.send_json({
                    "type": "unknown_type",
                    "received": data
                })

    except WebSocketDisconnect:
        manager.disconnect(session_id, websocket, user_id)
    except Exception as e:
        print(f"WebSocket error in {session_id}: {e}")
        manager.disconnect(session_id, websocket, user_id)
        try:
            await websocket.close(code=1011)
        except:
            pass
