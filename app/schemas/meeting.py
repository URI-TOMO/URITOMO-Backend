"""
Meeting Schemas
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# Settings -----------------------------------------------------------
class MeetingSettingBase(BaseModel):
    source_lang: str = "ja"
    target_lang: str = "ko"
    enable_translation: bool = True
    enable_explanation: bool = True
    style_profile: str = "business"
    explanation_level: str = "normal"


class MeetingSettingUpdate(MeetingSettingBase):
    pass


class MeetingSettingResponse(MeetingSettingBase):
    id: int
    
    class Config:
        from_attributes = True


# Participant --------------------------------------------------------
class ParticipantBase(BaseModel):
    user_id: Optional[int] = None
    guest_name: Optional[str] = None
    role: str = "attendee"


class ParticipantResponse(ParticipantBase):
    id: int
    joined_at: datetime
    
    class Config:
        from_attributes = True


# Meeting ------------------------------------------------------------
class MeetingBase(BaseModel):
    title: str
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: str = "scheduled"


class MeetingCreate(MeetingBase):
    org_id: int
    settings: Optional[MeetingSettingBase] = None


class MeetingUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: Optional[str] = None


class MeetingResponse(MeetingBase):
    id: int
    org_id: int
    host_id: int
    created_at: datetime
    updated_at: datetime
    settings: Optional[MeetingSettingResponse] = None
    
    class Config:
        from_attributes = True
