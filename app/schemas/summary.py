"""
Summary Schemas
"""

from typing import List, Optional, Any

from pydantic import BaseModel


class SummaryBase(BaseModel):
    content_text: str
    decisions: Optional[List[Any]] = None
    action_items: Optional[List[Any]] = None
    risks: Optional[List[Any]] = None
    model_used: Optional[str] = None


class SummaryCreate(SummaryBase):
    meeting_id: int


class SummaryResponse(SummaryBase):
    id: int
    meeting_id: int
    created_at: Any
    
    class Config:
        from_attributes = True
