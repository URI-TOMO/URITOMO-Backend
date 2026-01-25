"""
STT Translation Schemas
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class STTTranslationRequest(BaseModel):
    """
    Request model for STT (Speech-to-Text) translation
    """
    room_id: str = Field(..., description="ID of the room")
    participant_id: str = Field(..., description="ID of the participant")
    participant_name: str = Field(..., description="Name of the participant")
    is_speaking: bool = Field(..., description="Whether the participant is currently speaking")
    is_final: bool = Field(False, description="Whether this is a final transcript")
    Original: str = Field(..., description="Original transcribed text")
    timestamp: str = Field(..., description="ISO timestamp")
    sequence: str = Field(..., description="Order sequence of the transcription")
    Language: str = Field(..., description="Source language of the transcription")


class STTTranslationResponse(BaseModel):
    """
    Response model for STT (Speech-to-Text) translation
    """
    room_id: str
    participant_id: str
    participant_name: str
    Original: str
    translated: str
    timestamp: str
    sequence: str
    is_final: bool


class TermDescription(BaseModel):
    term: str
    explanation_ko: str
    explanation_ja: str


class DescriptionResponse(BaseModel):
    room_id: str
    terms: List[TermDescription]
