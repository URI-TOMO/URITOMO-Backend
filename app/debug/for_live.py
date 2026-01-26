from datetime import datetime, timedelta
import random
from uuid import uuid4
from typing import List

from fastapi import APIRouter, Depends, status, Body, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.infra.db import get_db
from app.core.token import verify_token
# from app.core.token import CurrentUserDep # Removed for body-token usage
from app.models import (
    User, Room, RoomMember, ChatMessage, Live, AIEvent,
    UserFriend, DmThread, DmParticipant, DmMessage,
    RoomLiveSession, RoomLiveSessionMember
)

router = APIRouter(tags=["debug"])

@router.post("/for-live", status_code=status.HTTP_201_CREATED)
async def generate_dense_live_data(
    token: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db)
):
    """
    **Dense Mock Data Generator (For Live Demo)**
    
    Populates the database with a high-density set of mock data for the user associated with the provided token.
    The token should be passed in the request body.
    """
    
    # 1. Resolve User from token
    current_user_id = verify_token(token)
    if not current_user_id:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    result = await db.execute(select(User).where(User.id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        return {"error": "User not found"}

    stats = {
        "friends": 0,
        "dm_messages": 0,
        "rooms": 0,
        "chat_messages": 0,
        "live_sessions": 0,
        "live_events": 0
    }

    # Data Pools
    locales = ["ko", "ja", "en"]
    names = {
        "ko": ["지훈", "서연", "도윤", "하은", "준우", "민지", "현우", "지수"],
        "ja": ["Hiroshi", "Sakura", "Kenji", "Yui", "Takumi", "Aoi", "Daisuke", "Nana"],
        "en": ["Oliver", "Sophie", "Leo", "Mia", "Jack", "Chloe", "Noah", "Lily"]
    }
    msgs_pool = [
        "How are you doing today?",
        "Did you see the latest update?",
        "I'm really excited about the live session.",
        "Can we schedule a meeting for tomorrow?",
        "That sounds like a great plan!",
        "Wait, I didn't quite catch that. Could you repeat?",
        "The translation quality is actually quite good.",
        "Let's focus on the cultural context part.",
        "I'll share the summary after the call.",
        "Does anyone have questions?"
    ]
    
    created_friends: List[User] = []

    # 2. Create 4 Friends & DM Threads
    for i in range(4):
        loc = random.choice(locales)
        name = random.choice(names[loc]) + f"_{uuid4().hex[:3]}"
        f_id = f"f_live_{uuid4().hex[:8]}"
        
        friend = User(
            id=f_id,
            display_name=name,
            email=f"{f_id}@uritomo.com",
            locale=loc,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=60)
        )
        db.add(friend)
        created_friends.append(friend)
        
        # Friendship
        friendship = UserFriend(
            id=str(uuid4()),
            requester_id=current_user_id,
            addressee_id=f_id,
            status="accepted",
            requested_at=datetime.utcnow() - timedelta(days=59),
            responded_at=datetime.utcnow() - timedelta(days=58)
        )
        db.add(friendship)
        
        # DM Thread
        thread = DmThread(
            id=str(uuid4()),
            user_friend_id=friendship.id,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=58)
        )
        db.add(thread)
        
        # Participants
        db.add(DmParticipant(id=str(uuid4()), thread_id=thread.id, user_id=current_user_id, joined_at=thread.created_at))
        db.add(DmParticipant(id=str(uuid4()), thread_id=thread.id, user_id=f_id, joined_at=thread.created_at))
        
        # 10 DM messages
        base_time = thread.created_at + timedelta(minutes=30)
        for k in range(10):
            sender_id = current_user_id if k % 2 == 0 else f_id
            sender_locale = user.locale if sender_id == current_user_id else friend.locale
            
            msg = DmMessage(
                id=str(uuid4()),
                thread_id=thread.id,
                seq=k+1,
                sender_type="human",
                sender_user_id=sender_id,
                message_type="text",
                text=random.choice(msgs_pool) + f" (Seq: {k+1})",
                lang=sender_locale,
                created_at=base_time + timedelta(hours=k)
            )
            db.add(msg)
            stats["dm_messages"] += 1
        
        stats["friends"] += 1
        if i % 4 == 0:
            await db.flush()

    # 3. Create 2 Rooms
    room_titles = [
        "Global Engineering Sync",
        "Culture & Language Hub",
    ]
    
    for i, title in enumerate(room_titles):
        room_id = str(uuid4())
        room = Room(
            id=room_id,
            title=title,
            created_by=current_user_id,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=30)
        )
        db.add(room)
        
        # Add Owner (Main User)
        owner_member = RoomMember(
            id=str(uuid4()),
            room_id=room_id,
            user_id=current_user_id,
            display_name=user.display_name,
            role="owner",
            joined_at=room.created_at
        )
        db.add(owner_member)
        
        # Add 1 random friend as member (2 members per room including owner)
        room_members = [owner_member]
        selected_friends = random.sample(created_friends, k=1)
        id_to_locale = {user.id: user.locale}
        
        for f in selected_friends:
            mem = RoomMember(
                id=str(uuid4()),
                room_id=room_id,
                user_id=f.id,
                display_name=f.display_name,
                role="member",
                joined_at=room.created_at + timedelta(minutes=random.randint(5, 500))
            )
            db.add(mem)
            room_members.append(mem)
            id_to_locale[f.id] = f.locale
            
        await db.flush()
            
        # 12 room chat messages
        base_time = room.created_at + timedelta(days=1)
        for k in range(12):
            sender_mem = random.choice(room_members)
            sender_locale = id_to_locale.get(sender_mem.user_id, "en")
            
            chat = ChatMessage(
                id=str(uuid4()),
                room_id=room_id,
                seq=k+1,
                sender_type="human",
                sender_member_id=sender_mem.id,
                message_type="text",
                text=f"Room Chat #{k+1}: {random.choice(msgs_pool)}",
                lang=sender_locale,
                created_at=base_time + timedelta(hours=k*2)
            )
            db.add(chat)
            stats["chat_messages"] += 1
            
        # 4. 1 Live Session per room
        live_seq = 0
        for s in range(1):
            session_id = str(uuid4())
            s_start = room.created_at + timedelta(days=s*5 + 2, hours=10)
            session = RoomLiveSession(
                id=session_id,
                room_id=room_id,
                title=f"{title} - Review Session {s+1}",
                status="ended",
                started_by=current_user_id,
                started_at=s_start,
                ended_at=s_start + timedelta(minutes=45)
            )
            db.add(session)
            
            # Session Members
            for m in room_members:
                db.add(RoomLiveSessionMember(
                    id=str(uuid4()),
                    session_id=session_id,
                    room_id=room_id,
                    member_id=m.id,
                    user_id=m.user_id,
                    display_name=m.display_name,
                    role=m.role,
                    joined_at=s_start + timedelta(seconds=random.randint(0, 5)),
                    left_at=s_start + timedelta(minutes=random.randint(30, 45))
                ))
            
            # 60 Utterances per session
            for u in range(60):
                speaker = random.choice(room_members)
                u_id = str(uuid4())
                u_start = s_start + timedelta(seconds=u*30)
                
                live_text = f"Utterance #{u+1}: Speaking about important details of {title}..."
                live_seq += 1
                
                live_item = Live(
                    id=u_id,
                    room_id=room_id,
                    member_id=speaker.id,
                    seq=live_seq,
                    text=live_text,
                    lang=id_to_locale.get(speaker.user_id, "en"),
                    start_ms=u*30000,
                    end_ms=u*30000 + 4000,
                    created_at=u_start
                )
                db.add(live_item)
                
                target_lang = "ko" if live_item.lang != "ko" else "ja"
                db.add(AIEvent(
                    id=str(uuid4()),
                    room_id=room_id,
                    seq=live_seq,
                    event_type="translation",
                    source_live_id=u_id,
                    original_text=live_text,
                    original_lang=live_item.lang,
                    translated_text=f"[AI Translated to {target_lang}] {live_text}",
                    translated_lang=target_lang,
                    created_at=u_start + timedelta(milliseconds=600)
                ))
                stats["live_events"] += 1
            
            stats["live_sessions"] += 1
        
        stats["rooms"] += 1
        await db.flush()

    await db.commit()

    return {
        "status": "success",
        "user_name": user.display_name,
        "seed_summary": {
            "total_friends_created": stats["friends"],
            "total_dm_messages_created": stats["dm_messages"],
            "total_rooms_created": stats["rooms"],
            "total_chat_messages_created": stats["chat_messages"],
            "total_live_sessions_created": stats["live_sessions"],
            "total_live_events_created": stats["live_events"]
        }
    }
