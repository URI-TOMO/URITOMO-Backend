import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select

from app.core.token import verify_token
from app.models.room import RoomLiveSession
from app.infra.db import AsyncSessionLocal

router = APIRouter(prefix="/meeting", tags=["meeting_ws"])

@router.get("/ws-info", include_in_schema=True)
async def websocket_docs():
    """
    Returns documentation on how to connect to the meeting WebSocket.
    """
    return {
        "websocket_url": "/api/v1/meeting/{session_id}",
        "auth": "pass token as query parameter ?token=...",
        "messages": {
            "receive": "JSON format",
            "send": "JSON format with echo"
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
    # 1. Authenticate (optional but recommended)
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
            # If session doesn't exist, we close the connection
            await websocket.close(code=1008) # Policy Violation
            return

    # 3. Accept connection
    await websocket.accept()
    
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
            # Expecting JSON data
            data = await websocket.receive_json()
            
            # Simple Echo for now
            # In a real app, this would relay to other members in the session
            await websocket.send_json({
                "type": "echo",
                "content": data
            })

    except WebSocketDisconnect:
        # Handle disconnection
        pass
    except Exception as e:
        # Log error?
        print(f"WebSocket error in {session_id}: {e}")
        try:
            await websocket.close(code=1011) # Internal Error
        except:
            pass
