"""
Translation Schemas
"""

from typing import Optional, Dict, Any

from pydantic import BaseModel


class TranslationBase(BaseModel):
    target_lang: str
    translated_text: str
    has_explanation: bool = False
    explanation_text: Optional[str] = None


class TranslationCreate(TranslationBase):
    original_segment_id: int
    provider: str = "openai"
    model: Optional[str] = None
    latency_ms: Optional[float] = None
    rag_context: Optional[Dict[str, Any]] = None


class TranslationResponse(TranslationBase):
    id: int
    original_segment_id: int
    
    class Config:
        from_attributes = True
