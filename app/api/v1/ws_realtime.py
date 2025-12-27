"""
WebSocket Realtime Endpoint
"""

from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db
from app.core.security import verify_token
from app.core.logging import get_logger

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
    # 1. Connect
    await websocket.accept()
    
    # 2. Authenticate
    user_id = verify_token(token)
    if not user_id:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        logger.warning("WS connection rejected: Invalid token")
        return

    logger.info(f"WS Connected: User {user_id}, Meeting {meeting_id}")

    try:
        while True:
            # 3. Receive message
            data = await websocket.receive_json()
            
            # TODO: Handle messages
            # For now, just echo ack
            message_type = data.get("type")
            
            if message_type == "segment.ingest":
                # 1. Parse Data
                ingest_data = data.get("data", {})
                meeting_id_in = ingest_data.get("meeting_id")
                text = ingest_data.get("text")
                lang = ingest_data.get("lang")
                
                if not text:
                    continue

                # Ack immediately
                await websocket.send_json({
                    "type": "segment.ack",
                    "status": "received",
                    "data": {"segment_ts": ingest_data.get("ts")}
                })
                
                # 2. Ingest Segment (Async)
                # In real app, might want to do this via queue/background if blocking
                # For now, simplistic inline
                from app.services.segment_service import SegmentService
                from app.schemas.segment import SegmentIngest
                
                seg_service = SegmentService(db)
                try:
                    segment = await seg_service.ingest_segment(SegmentIngest(**ingest_data))
                except Exception as e:
                    logger.error(f"Ingest failed: {e}")
                    await websocket.send_json({"type": "error", "message": str(e)})
                    continue

                # 3. Translation & Explanation Pipeline
                # Initialize services (could be dependencies)
                from app.services.translation_service import TranslationService
                from app.services.explanation_service import ExplanationService
                from app.services.rag_service import RagService
                from app.infra.qdrant import get_qdrant
                
                # We need Qdrant client for Explanation
                # This is a bit heavy inside loop, ideally services are initialized outside
                # or via dependency injection properly
                
                # For MVP simplicity, we init here (efficient if services are lightweight/cached)
                trans_service = TranslationService()
                
                # TODO: use dependency injected qdrant client from function args if possible
                # But we have it in `db`? No, we need `get_qdrant` result.
                # Since `socket` is long-lived, we should get deps at start.
                # For now, let's just do translation first
                
                # Target lang (hardcoded or from meeting settings in real app)
                target_lang = "ko" 
                
                translated_text, latency = await trans_service.translate(
                    text, source_lang=lang, target_lang=target_lang
                )
                
                # Push partial/final translation
                await websocket.send_json({
                    "type": "translation.final",
                    "data": {
                        "segment_id": segment.id,
                        "translated_text": translated_text,
                        "original_text": text
                    }
                })
                
                # 4. Explanation Check
                # Need Qdrant client
                # Using a quick hack to get client instance for now
                # In production, pass `qdrant` as dependency to websocket_endpoint
                from app.infra.qdrant import client as qdrant_client
                
                if qdrant_client:
                    rag_service = RagService(qdrant_client)
                    exp_service = ExplanationService(rag_service)
                    
                    decision = await exp_service.check_explanation_needed(text)
                    
                    if decision.should_explain:
                        explanation = await exp_service.generate_explanation(
                            text, translated_text, decision.matched_cards
                        )
                        
                        # Push explanation
                        await websocket.send_json({
                            "type": "explanation.available",
                            "data": {
                                "segment_id": segment.id,
                                "explanation_text": explanation,
                                "matched_cards": [c.model_dump() for c in decision.matched_cards]
                            }
                        })

    except WebSocketDisconnect:
        logger.info(f"WS Disconnected: User {user_id}")
    except Exception as e:
        logger.error(f"WS Error: {e}")
        try:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except:
            pass
