"""
WebSocket Realtime Endpoint
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.security import verify_token
from app.core.logging import get_logger
from app.services.connection_manager import manager
from app.services.segment_service import SegmentService
from app.services.translation_service import TranslationService
from app.services.explanation_service import ExplanationService
from app.services.rag_service import RagService
from app.schemas.segment import SegmentIngest
from app.infra.qdrant import client as qdrant_client

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/realtime")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    meeting_id: int = Query(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Realtime WebSocket endpoint for translation
    """
    # 1. Authenticate (Before accepting if possible, but FastAPI handles Query params after handshake)
    user_id = verify_token(token)
    if not user_id:
        await websocket.accept() # Must accept to send close code
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    # 2. Connect
    await manager.connect(websocket, meeting_id)
    logger.info(f"WS Connected: User {user_id}, Meeting {meeting_id}")

    try:
        while True:
            # 3. Receive message
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "segment.ingest":
                await handle_segment_ingest(websocket, meeting_id, data, db)
            
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket, meeting_id)
        logger.info(f"WS Disconnected: User {user_id}")
    except Exception as e:
        logger.error(f"WS Error for User {user_id}: {e}")
        manager.disconnect(websocket, meeting_id)
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass


async def handle_segment_ingest(websocket: WebSocket, meeting_id: int, data: dict, db: AsyncSession):
    """Handle incoming transcript segment"""
    ingest_data = data.get("data", {})
    text = ingest_data.get("text")
    lang = ingest_data.get("lang", "ja")
    
    if not text:
        return

    # 1. Ack immediately to the sender
    await websocket.send_json({
        "type": "segment.ack",
        "status": "received",
        "data": {"segment_ts": ingest_data.get("ts")}
    })
    
    # 2. Ingest to DB
    seg_service = SegmentService(db)
    try:
        segment = await seg_service.ingest_segment(SegmentIngest(**ingest_data))
    except Exception as e:
        logger.error(f"Segment ingest failed: {e}")
        await websocket.send_json({"type": "error", "message": "Failed to save segment"})
        return

    # 3. Translation Pipeline
    trans_service = TranslationService()
    target_lang = "ko" # Default for MVP
    
    translated_text, _ = await trans_service.translate(
        text, source_lang=lang, target_lang=target_lang
    )
    
    # Broadcast final translation to ALL participants in the meeting
    await manager.broadcast({
        "type": "translation.final",
        "data": {
            "segment_id": segment.id,
            "translated_text": translated_text,
            "original_text": text,
            "speaker": ingest_data.get("speaker")
        }
    }, meeting_id)
    
    # 4. Explanation (RAG) Pipeline
    if qdrant_client:
        rag_service = RagService(qdrant_client)
        exp_service = ExplanationService(rag_service)
        
        decision = await exp_service.check_explanation_needed(text)
        
        if decision.should_explain:
            explanation = await exp_service.generate_explanation(
                text, translated_text, decision.matched_cards
            )
            
            # Broadcast explanation to ALL participants
            await manager.broadcast({
                "type": "explanation.available",
                "data": {
                    "segment_id": segment.id,
                    "explanation_text": explanation,
                    "matched_cards": [c.model_dump() for c in decision.matched_cards]
                }
            }, meeting_id)

