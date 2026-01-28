"""
Translation API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.translation.schemas import TranslationRequest, TranslationResponse
from app.translation.deepl_service import deepl_service
from app.models.ai import AIEvent
from app.models.room import Room
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("/translate", response_model=TranslationResponse)
async def translate_message(
    request: TranslationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Translate a message and store the event.
    """
    try:
        # Determine target language based on source language
        # Requirement: Korean -> Japanese, Japanese -> Korean
        source_lang = request.Language
        target_lang = "Japanese" if "Korean" in source_lang else "Korean"
        
        # Perform translation
        translated_text = deepl_service.translate_text(
            text=request.Original,
            source_lang=source_lang,
            target_lang=target_lang
        )
        
        # Store in DB (AIEvent)
        # We need to ensure the room exists first to avoid FK constraint errors, 
        # but for performance we might assume it exists or handle the error.
        # Let's try to verify room existence if needed, but usually we just insert.
        # However, we need to convert sequence to int if it's stored as string in request 
        # but BigInteger in DB. The Requirement says sequence is "0" (string).
        # AIEvent.seq is BigInteger.
        
        try:
            seq_int = int(request.sequence)
        except ValueError:
            seq_int = 0 # Default or handle error
            
        ai_event = AIEvent(
            id=f"trans_{request.room_id}_{request.sequence}_{request.participant_id}"[:64], # Generate a unique ID or use UUID
            room_id=request.room_id,
            seq=seq_int,
            event_type="translation",
            original_text=request.Original,
            original_lang=source_lang,
            translated_text=translated_text,
            translated_lang=target_lang,
            meta={
                "participant_id": request.participant_id,
                "participant_name": request.participant_name,
                "timestamp": request.timestamp
            }
        )

        try:
             # Check if we need to generate ID manually. 
             # Usually models use default UUID generation if not provided, 
             # but here I assigned one to be safe/deterministic based on request.
             # However, it might clash if logic isn't perfect. 
             # Let's check AIEvent model definition again. 
             # It is String(64) primary key.
             # Better to use a UUID to avoid collision.
             import uuid
             ai_event.id = str(uuid.uuid4())
             
             db.add(ai_event)
             await db.commit()
        except Exception as e:
            logger.error(f"Failed to save translation event: {e}")
            # We continue even if DB save fails? 
            # Requirements didn't specify, but usually we want to return translation anyway.
            # But let's log it.
        
        return TranslationResponse(
            room_id=request.room_id,
            participant_id=request.participant_id,
            participant_name=request.participant_name,
            Original=request.Original,
            translated=translated_text,
            timestamp=request.timestamp,
            sequence=request.sequence
        )

    except Exception as e:
        logger.error(f"Translation endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
