"""
Segment Service

Handles transcript segments logic.
"""

from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.segment import TranscriptSegment
from app.models.meeting import Meeting
from app.schemas.segment import SegmentIngest


class SegmentService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def ingest_segment(self, segment_in: SegmentIngest) -> TranscriptSegment:
        # Validate meeting
        meeting = await self.session.get(Meeting, segment_in.meeting_id)
        if not meeting:
            raise ValueError("Meeting not found")

        segment = TranscriptSegment(
            meeting_id=segment_in.meeting_id,
            timestamp=segment_in.ts,
            speaker_name=segment_in.speaker,
            language=segment_in.lang,
            text=segment_in.text,
            is_final=segment_in.is_final,
        )
        self.session.add(segment)
        await self.session.commit()
        await self.session.refresh(segment)
        return segment

    async def get_meeting_segments(self, meeting_id: int) -> List[TranscriptSegment]:
        stmt = (
            select(TranscriptSegment)
            .options(selectinload(TranscriptSegment.translations))
            .where(TranscriptSegment.meeting_id == meeting_id)
            .order_by(TranscriptSegment.timestamp)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
