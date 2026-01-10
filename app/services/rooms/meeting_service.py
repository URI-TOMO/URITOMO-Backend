"""
Meeting Service
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.meeting import Meeting, MeetingParticipant, MeetingSetting
from app.models.org import Organization
from app.schemas.meeting import MeetingCreate, MeetingUpdate, MeetingSettingBase


class MeetingService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_meeting(self, meeting_in: MeetingCreate, host_id: int) -> Meeting:
        # Check org exists
        org = await self.session.get(Organization, meeting_in.org_id)
        if not org:
            raise ValueError("Organization not found")

        # Create meeting
        meeting = Meeting(
            title=meeting_in.title,
            description=meeting_in.description,
            start_time=meeting_in.start_time,
            end_time=meeting_in.end_time,
            status=meeting_in.status,
            org_id=meeting_in.org_id,
            host_id=host_id,
        )
        self.session.add(meeting)
        await self.session.flush()

        # Add settings (use default if none)
        settings_in = meeting_in.settings or MeetingSettingBase()
        settings = MeetingSetting(
            meeting_id=meeting.id,
            source_lang=settings_in.source_lang,
            target_lang=settings_in.target_lang,
            enable_translation=settings_in.enable_translation,
            enable_explanation=settings_in.enable_explanation,
            style_profile=settings_in.style_profile,
            explanation_level=settings_in.explanation_level,
        )
        self.session.add(settings)

        # Add host as participant
        host_participant = MeetingParticipant(
            meeting_id=meeting.id,
            user_id=host_id,
            role="host"
        )
        self.session.add(host_participant)

        await self.session.commit()
        await self.session.refresh(meeting)
        return meeting

    async def get_meeting(self, meeting_id: int) -> Optional[Meeting]:
        stmt = (
            select(Meeting)
            .options(
                selectinload(Meeting.settings),
                selectinload(Meeting.participants)
            )
            .where(Meeting.id == meeting_id)
        )
        result = await self.session.execute(stmt)
        return result.scalars().first()

    async def get_org_meetings(self, org_id: int) -> List[Meeting]:
        stmt = select(Meeting).where(Meeting.org_id == org_id)
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
