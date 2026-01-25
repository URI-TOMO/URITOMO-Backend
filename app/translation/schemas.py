"""
Translation Schemas
"""

from typing import Optional
from pydantic import BaseModel, Field


class TranslationRequest(BaseModel):
    """
    Request model for translation
    """
    room_id: str
    participant_id: str
    participant_name: str
    is_speaking: bool
    Original: str  # Note: Capitalized as per requirements
    timestamp: str
    sequence: str
    Language: str  # Note: Capitalized as per requirements


class TranslationResponse(BaseModel):
    """
    Response model for translation
    """
    room_id: str
    participant_id: str
    participant_name: str
    Original: str
    translated: str
    timestamp: str
    sequence: str
