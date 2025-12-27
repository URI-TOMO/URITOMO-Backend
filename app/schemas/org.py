"""
Organization Schemas
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class OrgBase(BaseModel):
    name: str
    slug: str
    default_source_lang: str = "ja"
    default_target_lang: str = "ko"


class OrgCreate(OrgBase):
    pass


class OrgUpdate(BaseModel):
    name: Optional[str] = None
    default_source_lang: Optional[str] = None
    default_target_lang: Optional[str] = None


class OrgResponse(OrgBase):
    id: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class OrgMemberBase(BaseModel):
    user_id: int
    role: str = "member"


class OrgMemberCreate(OrgMemberBase):
    pass
