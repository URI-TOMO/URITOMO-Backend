from typing import Optional
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from sqlalchemy import select
from app.core.token import verify_token
from app.models.dm import DmThread, DmParticipant
from app.meeting.ws.manager import manager
from app.infra.db import AsyncSessionLocal

router = APIRouter(tags=["dm-ws"])
logger = logging.getLogger("uritomo.dm")

@router.websocket("/dm/ws/{thread_id}")
async def dm_websocket(
    websocket: WebSocket,
    thread_id: str,
    token: Optional[str] = Query(None)
):
    print(f"🔌 [DM WS] Connection attempt | thread_id={thread_id} | token={'present' if token else 'missing'}", flush=True)
    
    # 1. Auth required for DM
    user_id = None
    if token:
        try:
            # Manually decode to log errors
            from jose import jwt
            from app.core.config import settings
            payload = jwt.decode(
                token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
            )
            user_id = payload.get("sub")
        except Exception as e:
            print(f"❌ [DM WS] Auth error | thread_id={thread_id} | error={e}", flush=True)
            user_id = None
    
    if not user_id:
        print(f"❌ [DM WS] Auth failed | thread_id={thread_id}", flush=True)
        # Unauthorized
        await websocket.close(code=4003)
        return

    print(f"✅ [DM WS] Auth success | user_id={user_id}", flush=True)

    # 2. Check Thread and Participation
    async with AsyncSessionLocal() as db_session:
        stmt_part = select(DmParticipant).where(
            DmParticipant.thread_id == thread_id, 
            DmParticipant.user_id == user_id
        )
        res_part = await db_session.execute(stmt_part)
        participant = res_part.scalar_one_or_none()
        
        if not participant:
            print(f"❌ [DM WS] Participant not found | thread_id={thread_id} | user_id={user_id}", flush=True)
            # Not allowed or thread not found
            await websocket.close(code=4003)
            return

    print(f"✅ [DM WS] Participant verified | thread_id={thread_id}", flush=True)

    # 3. Connect
    await manager.connect(thread_id, websocket, user_id)
    print(f"✅ [DM WS] Connected | thread_id={thread_id} | user_id={user_id}", flush=True)
    logger.info(f"✅ DM WS Connected | Thread: {thread_id} | User: {user_id}")
    
    try:
        # Send initial connected message
        await websocket.send_json({
            "type": "room_connected",
            "data": {
                "room_id": thread_id,
                "user_id": user_id
            }
        })
        print(f"📤 [DM WS] Sent room_connected | thread_id={thread_id}", flush=True)
        
        while True:
            # Keep alive & listen for potential client messages (e.g. typing indicators)
            data = await websocket.receive_json()
            print(f"📥 [DM WS] Received data | thread_id={thread_id} | data={data}", flush=True)
            # Currently we don't handle messages sent via WS for DM, preferring REST API
            # But we could add handlers here later.
            
    except WebSocketDisconnect:
        manager.disconnect(thread_id, websocket, user_id)
        print(f"🔌 [DM WS] Disconnected | thread_id={thread_id} | user_id={user_id}", flush=True)
        logger.info(f"🔌 DM WS Disconnected | Thread: {thread_id} | User: {user_id}")
    except Exception as e:
        print(f"❌ [DM WS] Error | thread_id={thread_id} | error={e}", flush=True)
        logger.error(f"WS Error: {e}")
        manager.disconnect(thread_id, websocket, user_id)
