"""
STT (Speech-to-Text) Translation API Endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.translation.schemas import STTTranslationRequest, STTTranslationResponse, DescriptionResponse
from app.translation.deepl_service import deepl_service
from app.translation.openai_service import openai_service
from app.models.room import RoomLiveSession
from app.models.stt import RoomSttResult
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()

@router.post("/stt", response_model=STTTranslationResponse)
async def translate_stt(
    request: STTTranslationRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Translate STT transcription and store the event.
    Targeted specifically for real-time speech-to-text data.
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
        
        # Store in DB (RoomSttResult) only if it's a final transcript
        # For non-final (partial) results, we might skip DB storage to save space,
        # but the user didn't specify. Usually only final results are persisted.
        if request.is_final:
            try:
                try:
                    seq_int = int(request.sequence)
                except ValueError:
                    seq_int = 0

                session_result = await db.execute(
                    select(RoomLiveSession)
                    .where(
                        RoomLiveSession.room_id == request.room_id,
                        RoomLiveSession.status == "active",
                    )
                    .order_by(RoomLiveSession.started_at.desc())
                )
                live_session = session_result.scalars().first()
                if not live_session:
                    raise ValueError("Active live session not found")

                import uuid
                stt_result = RoomSttResult(
                    id=str(uuid.uuid4()),
                    room_id=request.room_id,
                    session_id=live_session.id,
                    member_id=request.participant_id,
                    user_lang=source_lang,
                    stt_text=request.Original,
                    translated_text=translated_text,
                    translated_lang=target_lang,
                    seq=seq_int,
                    meta={
                        "participant_name": request.participant_name,
                        "timestamp": request.timestamp,
                        "is_speaking": request.is_speaking,
                    },
                )

                db.add(stt_result)
                await db.commit()
            except Exception as e:
                logger.error(f"Failed to save STT translation event: {e}")
        
        return STTTranslationResponse(
            room_id=request.room_id,
            participant_id=request.participant_id,
            participant_name=request.participant_name,
            Original=request.Original,
            translated=translated_text,
            timestamp=request.timestamp,
            sequence=request.sequence,
            is_final=request.is_final
        )

    except Exception as e:
        logger.error(f"STT Translation endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/description/{room_id}", response_model=DescriptionResponse)
async def get_term_descriptions(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed explanations for difficult terms used in the meeting so far.
    Fetches all STT data, aggregates it, and uses OpenAI to identify/explain terms.
    """
    try:
        session_result = await db.execute(
            select(RoomLiveSession)
            .where(
                RoomLiveSession.room_id == room_id,
                RoomLiveSession.status == "active",
            )
            .order_by(RoomLiveSession.started_at.desc())
        )
        live_session = session_result.scalars().first()
        if not live_session:
            return DescriptionResponse(room_id=room_id, terms=[])

        # 1. Fetch all STT data for the active session
        stmt = (
            select(RoomSttResult)
            .where(
                RoomSttResult.room_id == room_id,
                RoomSttResult.session_id == live_session.id,
            )
            .order_by(RoomSttResult.seq.asc())
        )

        result = await db.execute(stmt)
        results = result.scalars().all()

        if not results:
            return DescriptionResponse(room_id=room_id, terms=[])
            
        # 2. Aggregate text
        full_text = " ".join([r.stt_text for r in results])
        
        # 3. Get descriptions from OpenAI
        terms = await openai_service.get_description_for_terms(full_text)
        
        return DescriptionResponse(
            room_id=room_id,
            terms=terms
        )
        
    except Exception as e:
        logger.error(f"Error in description endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
