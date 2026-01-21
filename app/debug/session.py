from datetime import datetime
from uuid import uuid4
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.infra.db import get_db
from app.core.token import CurrentUserDep
from app.models import User, Room, RoomMember, RoomLiveSession

router = APIRouter(tags=["debug"])

from app.core.token import create_access_token
import random

@router.post("/all-in-one", status_code=status.HTTP_201_CREATED)
async def all_in_one_debug_setup(
    db: AsyncSession = Depends(get_db)
):
    """
    **All-In-One Debug Setup (No Auth Required)**
    
    1. Creates a random user.
    2. Issues an Access Token.
    3. Creates a Room & Active Live Session.
    4. Returns everything needed to test WebSocket immediately (including a Copy-Paste PowerShell script).
    """
    # 1. Create Random User
    rand_suffix = uuid4().hex[:6]
    user_id = f"debug_user_{rand_suffix}"
    
    user = User(
        id=user_id,
        display_name=f"Tester_{rand_suffix}",
        email=f"tester_{rand_suffix}@example.com",
        locale="ko",
        status="active",
        created_at=datetime.utcnow()
    )
    db.add(user)
    
    # 2. Setup Room
    room_id = f"room_{rand_suffix}"
    room = Room(
        id=room_id,
        title=f"Debug Room {rand_suffix}",
        created_by=user_id,
        status="active",
        created_at=datetime.utcnow()
    )
    db.add(room)
    
    # Add Member
    member = RoomMember(
        id=str(uuid4()),
        room_id=room_id,
        user_id=user_id,
        display_name=user.display_name,
        role="owner",
        joined_at=datetime.utcnow()
    )
    db.add(member)
    
    # 3. Create Active Session
    session_id = f"ls_{rand_suffix}"
    live_session = RoomLiveSession(
        id=session_id,
        room_id=room_id,
        title=f"Live Session {rand_suffix}",
        status="active",
        started_by=user_id,
        started_at=datetime.utcnow()
    )
    db.add(live_session)
    
    await db.commit()
    
    # 4. Generate Token
    access_token = create_access_token(data={"sub": user_id})
    
    # 5. Prepare Output
    from app.core.config import settings
    # If host is 0.0.0.0, we probably want to show localhost or specific IP for testing
    # But for now let's respect the settings or fallback to a sensible default 
    # For Docker local dev, 10.0.255.80 seems to be the user's specific IP, 
    # but we can try to use what's in settings if possible, or keep it generic.
    
    # Ideally, we should detect the request host, but here we are in a background task context potentially.
    # Let's use the explicit request from user if available, or settings.
    # Since we can't easily get request obj here without adding it to params, let's use the env settings.
    
    host = settings.api_host if settings.api_host != "0.0.0.0" else "10.0.255.80"
    ws_url = f"ws://{host}:{settings.api_port}/meeting/{session_id}?token={access_token}"
    
    # PowerShell Script for instant testing
    ps_script = f"""
$url = "{ws_url}"
$ws = New-Object System.Net.WebSockets.ClientWebSocket
$ct = New-Object System.Threading.CancellationToken
$ws.ConnectAsync($url, $ct).Wait()
Write-Host "✅ Connected to {session_id}" -ForegroundColor Green
$buffer = New-Object byte[] 4096
while ($ws.State -eq 'Open') {{
    $res = $ws.ReceiveAsync(new-object System.ArraySegment[byte]($buffer), $ct)
    $res.Wait()
    $msg = [System.Text.Encoding]::UTF8.GetString($buffer, 0, $res.Result.Count)
    Write-Host "📩 $msg" -ForegroundColor Cyan
}}
"""

    return {
        "message": "All-in-one setup complete!",
        "user_id": user_id,
        "session_id": session_id,
        "token": access_token,
        "ws_url": ws_url,
        "powershell_script": ps_script
    }
