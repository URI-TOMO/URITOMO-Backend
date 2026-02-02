from datetime import datetime
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import and_, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.token import CurrentUserDep
from app.infra.db import get_db
from app.models.dm import DmParticipant, DmThread
from app.models.friend import UserFriend

router = APIRouter(tags=["dm"])

class DmThreadResponse(BaseModel):
    id: str
    created_at: datetime
    friend_id: str
    friend_name: Optional[str] = None

@router.post("/dm/start", response_model=DmThreadResponse)
async def start_dm(
    friend_id: str,
    current_user: CurrentUserDep,
    session: AsyncSession = Depends(get_db)
):
    """
    Start or get existing DM thread with a friend.
    """
    try:
        # current_user is a string (user_id), not a User object
        user_id = current_user
        print(f"🔗 [DM] start_dm called | friend_id={friend_id} | current_user={user_id}", flush=True)
        
        # 1. Check friendship
        stmt = select(UserFriend).where(
            or_(
                and_(UserFriend.requester_id == user_id, UserFriend.addressee_id == friend_id),
                and_(UserFriend.requester_id == friend_id, UserFriend.addressee_id == user_id)
            ),
            UserFriend.status == "accepted"
        ).options(selectinload(UserFriend.dm_thread))
        
        result = await session.execute(stmt)
        friendship = result.scalars().first()

        if not friendship:
            print(f"❌ [DM] Friendship not found | friend_id={friend_id}", flush=True)
            raise HTTPException(status_code=404, detail="Friends relation not found")

        print(f"✅ [DM] Friendship found | friendship_id={friendship.id}", flush=True)
        thread = friendship.dm_thread
        
        if not thread:
            print(f"📝 [DM] Creating new thread for friendship={friendship.id}", flush=True)
            # Create new thread
            thread = DmThread(
                id=f"dm_{uuid.uuid4().hex[:16]}",
                user_friend_id=friendship.id,
                created_at=datetime.utcnow(),
                status="active"
            )
            session.add(thread)
            await session.flush() # get ID
            print(f"✅ [DM] Thread created | thread_id={thread.id}", flush=True)

            # Add participants
            p1 = DmParticipant(
                id=f"dmp_{uuid.uuid4().hex[:16]}",
                thread_id=thread.id,
                user_id=user_id,
                joined_at=datetime.utcnow()
            )
            p2 = DmParticipant(
                id=f"dmp_{uuid.uuid4().hex[:16]}",
                thread_id=thread.id,
                user_id=friend_id,
                joined_at=datetime.utcnow()
            )
            session.add_all([p1, p2])
            await session.commit()
            print(f"✅ [DM] Participants added", flush=True)
        else:
            print(f"✅ [DM] Existing thread found | thread_id={thread.id}", flush=True)
            
            # Ensure participants exist (Self-healing for inconsistent state)
            # 1. Check current user
            p1_stmt = select(DmParticipant).where(
                DmParticipant.thread_id == thread.id, 
                DmParticipant.user_id == user_id
            )
            p1_res = await session.execute(p1_stmt)
            if not p1_res.scalars().first():
                print(f"⚠️ [DM] Repairing participant for user={user_id}", flush=True)
                new_p1 = DmParticipant(
                    id=f"dmp_{uuid.uuid4().hex[:16]}",
                    thread_id=thread.id,
                    user_id=user_id,
                    joined_at=datetime.utcnow()
                )
                session.add(new_p1)
                
            # 2. Check friend
            p2_stmt = select(DmParticipant).where(
                DmParticipant.thread_id == thread.id, 
                DmParticipant.user_id == friend_id
            )
            p2_res = await session.execute(p2_stmt)
            if not p2_res.scalars().first():
                print(f"⚠️ [DM] Repairing participant for friend={friend_id}", flush=True)
                new_p2 = DmParticipant(
                     id=f"dmp_{uuid.uuid4().hex[:16]}",
                    thread_id=thread.id,
                    user_id=friend_id,
                    joined_at=datetime.utcnow()
                )
                session.add(new_p2)
            
            await session.commit()
        
        return {
            "id": thread.id, 
            "created_at": thread.created_at,
            "friend_id": friend_id,
            "friend_name": friendship.friend_name
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"❌ [DM] Error in start_dm: {str(e)}", flush=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"DM start failed: {str(e)}")
