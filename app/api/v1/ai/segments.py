"""
Segment Endpoints
"""

from typing import List, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, CurrentUserDep
from app.schemas.segment import SegmentCreate, SegmentResponse, SegmentIngest
from app.services.segment_service import SegmentService

router = APIRouter()


@router.post("/", response_model=SegmentResponse, status_code=status.HTTP_201_CREATED)
async def ingest_segment(
    segment_in: SegmentIngest,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Ingest a transcript segment (used by HTTP clients)
    """
    segment_service = SegmentService(db)
    try:
        segment = await segment_service.ingest_segment(segment_in)
        return segment
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/meetings/{meeting_id}", response_model=List[SegmentResponse])
async def list_segments(
    meeting_id: int,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get all segments for a meeting
    """
    segment_service = SegmentService(db)
    segments = await segment_service.get_meeting_segments(meeting_id)
    return segments
