"""
Meeting Endpoints
"""

from typing import List, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, CurrentUserDep, QueueDep
from app.schemas.meeting import MeetingCreate, MeetingResponse, MeetingUpdate
from app.schemas.summary import SummaryResponse
from app.services.meeting_service import MeetingService
from app.infra.queue import JobQueue

router = APIRouter()


@router.post("/", response_model=MeetingResponse, status_code=status.HTTP_201_CREATED)
async def create_meeting(
    meeting_in: MeetingCreate,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Create a new meeting
    """
    meeting_service = MeetingService(db)
    try:
        meeting = await meeting_service.create_meeting(meeting_in, int(current_user_id))
        return meeting
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{meeting_id}", response_model=MeetingResponse)
async def read_meeting(
    meeting_id: int,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get meeting details
    """
    meeting_service = MeetingService(db)
    meeting = await meeting_service.get_meeting(meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail="Meeting not found")
    return meeting


@router.get("/", response_model=List[MeetingResponse])
async def list_meetings(
    org_id: int,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    List meetings for an organization
    """
    meeting_service = MeetingService(db)
    # Validate org access (TODO)
    meetings = await meeting_service.get_org_meetings(org_id)
    return meetings


@router.post("/{meeting_id}/summary", status_code=202)
async def trigger_meeting_summary(
    meeting_id: int,
    current_user_id: CurrentUserDep,
    queue: JobQueue = Depends(QueueDep),
) -> Any:
    """
    Trigger summary generation background job
    """
    # Simply enqueue job
    queue.enqueue(
        "app.workers.jobs.summarize_meeting.summarize_meeting",
        args=(meeting_id,),
    )
    return {"status": "enqueued", "meeting_id": meeting_id}


@router.get("/{meeting_id}/summary", response_model=List[SummaryResponse])
async def get_meeting_summary(
    meeting_id: int,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get summaries for a meeting
    """
    # Simple direct query for now or add to MeetingService
    # For MVP, assuming checking `meeting.summaries` relationship via get_meeting
    # But lazy loading might be an issue if session closed.
    # Better to have dedicated method.
    pass 
    # Placeholder: Implemented below correctly
    from app.models.summary import Summary
    from sqlalchemy import select
    
    result = await db.execute(select(Summary).where(Summary.meeting_id == meeting_id))
    return list(result.scalars().all())
