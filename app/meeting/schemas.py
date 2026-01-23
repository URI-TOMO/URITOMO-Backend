from datetime import datetime
from typing import Optional
from pydantic import BaseModel

class LiveSessionSchema(BaseModel):
    id: str
    room_id: str
    title: str
    status: str
    started_by: str
    started_at: datetime
    ended_at: Optional[datetime]

class LiveSessionMemberSchema(BaseModel):
    id: str
    session_id: str
    member_id: str
    user_id: Optional[str]
    display_name: str
    role: str
    joined_at: datetime
    left_at: Optional[datetime]

class SuccessResponse(BaseModel):
    status: str
    data: dict
