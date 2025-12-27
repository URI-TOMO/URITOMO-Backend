"""
Glossary Schemas
"""

from typing import Optional

from pydantic import BaseModel


class GlossaryEntryBase(BaseModel):
    term: str
    definition: str
    context_notes: Optional[str] = None


class GlossaryEntryCreate(GlossaryEntryBase):
    pass


class GlossaryEntryUpdate(BaseModel):
    term: Optional[str] = None
    definition: Optional[str] = None
    context_notes: Optional[str] = None


class GlossaryEntryResponse(GlossaryEntryBase):
    id: int
    org_id: int
    
    class Config:
        from_attributes = True
