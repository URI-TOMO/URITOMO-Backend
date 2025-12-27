"""
Segment Schemas
"""

from typing import List, Optional

from pydantic import BaseModel
from .translation import TranslationResponse


class SegmentBase(BaseModel):
    meeting_id: int
    timestamp: float
    duration: float = 0.0
    speaker_name: str
    language: str
    text: str
    is_final: bool = True


class SegmentCreate(SegmentBase):
    pass


class SegmentResponse(SegmentBase):
    id: int
    translations: List[TranslationResponse] = []
    
    class Config:
        from_attributes = True


# For WebSocket ingest
class SegmentIngest(BaseModel):
    meeting_id: int
    speaker: str
    lang: str
    text: str
    ts: float  # Timestamp
    is_final: bool = True
