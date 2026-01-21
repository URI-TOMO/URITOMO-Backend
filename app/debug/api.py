from datetime import datetime, timedelta
import random
import hashlib
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Depends, status, Query
from sqlalchemy import text, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.infra.db import get_db
from app.models import (
    User, Room, RoomMember, ChatMessage, Live, AIEvent,
    UserFriend, DmThread, DmParticipant, DmMessage,
    RoomLiveSession, RoomLiveSessionMember, AuthToken
)
from app.core.token import CurrentUserDep

from app.debug.signin import router as signin_router

router = APIRouter(tags=["debug"])
router.include_router(signin_router)

FIXED_USERS = [
    {"id": "1", "display_name": "Jin", "email": "jin@example.com", "locale": "ko"},
    {"id": "2", "display_name": "Daiki", "email": "daiki@example.com", "locale": "ja"},
    {"id": "3", "display_name": "Jeahyun", "email": "jeahyun@example.com", "locale": "ko"},
    {"id": "4", "display_name": "Kashihara", "email": "kashihara@example.com", "locale": "ja"},
    {"id": "5", "display_name": "Sarah", "email": "sarah@example.com", "locale": "en"},
    {"id": "6", "display_name": "Mike", "email": "mike@example.com", "locale": "en"},
]

@router.post("/seed", status_code=status.HTTP_201_CREATED)
async def seed_large_mock_data(db: AsyncSession = Depends(get_db)):
    """
    Seed database with deterministic Large Mock Data.
    Creates users, full mesh friendships, dense DMs, multiple rooms with dense chats, and live sessions.
    """
    
    # 1. Create Users
    users = []
    for u_data in FIXED_USERS:
        # Check if user exists to avoid Primary Key Error if run multiple times without clear
        existing = await db.get(User, u_data["id"])
        if not existing:
            user = User(
                id=u_data["id"],
                email=u_data["email"],
                display_name=u_data["display_name"],
                locale=u_data["locale"],
                status="active",
                created_at=datetime.utcnow() - timedelta(days=60)
            )
            db.add(user)
            users.append(user)
            
            # Create a dummy token for easy testing (Token = "token_" + user_id)
            # Hash logic: we just simulate a hash. In real app, standard hashing applies.
            raw_token = f"token_{user.id}"
            token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
            
            auth_token = AuthToken(
                id=str(uuid4()),
                user_id=user.id,
                token_hash=token_hash, # This probably fails meaningful validation but populates the table
                expires_at=datetime.utcnow() + timedelta(days=365),
                issued_at=datetime.utcnow(),
                session_id=str(uuid4())
            )
            db.add(auth_token)
        else:
            users.append(existing)
    
    await db.flush()

    # 2. Friendships (Complete Graph - Everyone is friends with everyone)
    # Also create DMs for every friendship immediately
    user_count = len(users)
    for i in range(user_count):
        for j in range(i + 1, user_count):
            u1, u2 = users[i], users[j]
            
            # Check exist
            exists = await db.execute(
                select(UserFriend).where(
                    ((UserFriend.requester_id == u1.id) & (UserFriend.addressee_id == u2.id)) |
                    ((UserFriend.requester_id == u2.id) & (UserFriend.addressee_id == u1.id))
                )
            )
            if exists.scalars().first():
                continue

            # Random requester
            req, addr = (u1, u2) if random.random() > 0.5 else (u2, u1)
            
            friendship = UserFriend(
                id=str(uuid4()),
                requester_id=req.id,
                addressee_id=addr.id,
                status="accepted",
                requested_at=datetime.utcnow() - timedelta(days=50),
                responded_at=datetime.utcnow() - timedelta(days=49)
            )
            db.add(friendship)
            
            # 3. DM Thread
            dm_thread = DmThread(
                id=str(uuid4()),
                user_friend_id=friendship.id,
                status="active",
                created_at=datetime.utcnow() - timedelta(days=49)
            )
            db.add(dm_thread)
            
            # DM Participants
            db.add(DmParticipant(id=str(uuid4()), thread_id=dm_thread.id, user_id=u1.id, joined_at=datetime.utcnow() - timedelta(days=49)))
            db.add(DmParticipant(id=str(uuid4()), thread_id=dm_thread.id, user_id=u2.id, joined_at=datetime.utcnow() - timedelta(days=49)))
            
            # 4. Dense DM Messages (200 messages per pair)
            base_time = datetime.utcnow() - timedelta(days=48)
            time_increment = timedelta(days=48) / 205
            
            for k in range(200):
                sender = u1 if k % 2 == 0 else u2
                msg_text = f"Hey {u2.display_name if sender == u1 else u1.display_name}, this is DM #{k+1}. How are you?"
                
                msg = DmMessage(
                    id=str(uuid4()),
                    thread_id=dm_thread.id,
                    seq=k+1,
                    sender_type="human",
                    sender_user_id=sender.id,
                    message_type="text",
                    text=msg_text,
                    lang=sender.locale,
                    created_at=base_time + (time_increment * k)
                )
                db.add(msg)

    await db.flush()

    # 5. Create Rooms (Dense Activity)
    room_configs = [
        ("Global Team Sync", users),  # All users
        ("Project Alpha", [users[0], users[1], users[2]]),
        ("Lunch Buddies", [users[2], users[3], users[4]]),
        ("Gaming Club", [users[0], users[3], users[5]]),
        ("Serious Business", [users[1], users[5]]),
    ]
    
    created_rooms = []
    # Increase density: Create 300 messages per room
    for title, participants in room_configs:
        room = Room(
            id=str(uuid4()),
            title=title,
            created_by=participants[0].id,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=40)
        )
        db.add(room)
        created_rooms.append(room)
        
        member_map = {}
        for p in participants:
            member = RoomMember(
                id=str(uuid4()),
                room_id=room.id,
                user_id=p.id,
                display_name=p.display_name,
                role="owner" if p.id == room.created_by else "member",
                joined_at=room.created_at
            )
            db.add(member)
            member_map[p.id] = member
            
        await db.flush()
        
        # 6. Room Chat Messages (300 messages)
        base_time = room.created_at + timedelta(minutes=10)
        time_increment = timedelta(days=20) / 305
        
        for k in range(300):
            sender_user = participants[k % len(participants)]
            sender_member = member_map[sender_user.id]
            
            chat = ChatMessage(
                id=str(uuid4()),
                room_id=room.id,
                seq=k+1,
                sender_type="human",
                sender_member_id=sender_member.id,
                message_type="text",
                text=f"[{title}] This is message #{k+1}. Hope everyone is doing great! @{random.choice(participants).display_name}",
                lang=sender_user.locale,
                created_at=base_time + (time_increment * k)
            )
            db.add(chat)
            
        # 7. Room Live Sessions (5 sessions per room)
        for s in range(5):
            # Stagger sessions
            session_start = room.created_at + timedelta(days=s*5, hours=14)
            session = RoomLiveSession(
                id=str(uuid4()),
                room_id=room.id,
                title=f"{title} - Live Discussion {s+1}",
                status="ended",
                started_by=participants[0].id,
                started_at=session_start,
                ended_at=session_start + timedelta(minutes=45)
            )
            db.add(session)
            
            # Session Members
            for p in participants:
                # Randomly some join, some don't? No, user requested "dense", let's make most join
                if random.random() > 0.1: # 90% join rate
                    sm = RoomLiveSessionMember(
                        id=str(uuid4()),
                        session_id=session.id,
                        room_id=room.id,
                        member_id=member_map[p.id].id,
                        user_id=p.id,
                        display_name=p.display_name,
                        role=member_map[p.id].role,
                        joined_at=session_start + timedelta(minutes=random.randint(0, 5)),
                        left_at=session_start + timedelta(minutes=random.randint(30, 45))
                    )
                    db.add(sm)
            
            # 8. Live Events (Utterances) - Dense: 100 per session
            for u in range(100):
                speaker_user = participants[u % len(participants)]
                speaker_member = member_map[speaker_user.id]
                utterance_start = session_start + timedelta(seconds=u*15)
                
                live_text = f"Speaking about important topic point #{u+1}..."
                
                global_seq = (s * 100) + u + 1 # Ensure uniqueness per room
                
                live = Live(
                    id=str(uuid4()),
                    room_id=room.id,
                    member_id=speaker_member.id,
                    seq=global_seq,
                    text=live_text,
                    lang=speaker_user.locale,
                    start_ms=u*15000,
                    end_ms=(u*15000)+5000,
                    created_at=utterance_start
                )
                db.add(live)
                
                # AI Translation / Summary Event
                if u % 2 == 0: # Translate every other line
                    ai_event = AIEvent(
                        id=str(uuid4()),
                        room_id=room.id,
                        seq=global_seq,
                        event_type="translation",
                        source_live_id=live.id,
                        original_text=live_text,
                        original_lang=speaker_user.locale,
                        translated_text=f"(Translated) {live_text}",
                        translated_lang="en", 
                        created_at=utterance_start + timedelta(milliseconds=200)
                    )
                    db.add(ai_event)

    await db.commit()
    return {
        "message": "Heavy dense mock data seeded!",
        "created_room_ids": [r.id for r in created_rooms],
        "stats": {
            "users": len(users),
            "friendships": user_count * (user_count - 1) // 2,
            "dms_per_thread": 200,
            "rooms": len(room_configs),
            "msg_per_room": 300,
            "sessions_per_room": 5,
            "live_utterances_per_session": 100
        }
    }



@router.post("/seed/user", status_code=status.HTTP_201_CREATED)
async def seed_user_personal_data(
    user_id: str = Query(..., description="Target User ID to populate personal data for"),
    db: AsyncSession = Depends(get_db)
):
    """
    **Personal Data Seeder**
    
    Sets up a specific user environment for testing:
    - Creates the user if missing.
    - Adds 2 friends (Accepted status).
    - Creates 2 active rooms where this user is a member/owner.
    
    Useful for testing "My Page", "Friends List", or specific user scenarios.
    """
    # 1. Ensure main user exists
    user_stmt = select(User).where(User.id == user_id)
    user_result = await db.execute(user_stmt)
    user = user_result.scalar_one_or_none()
    
    if not user:
        user = User(
            id=user_id,
            display_name=f"User_{user_id[:4]}",
            email=f"{user_id}@example.com",
            locale="ja",
            status="active",
            created_at=datetime.utcnow()
        )
        db.add(user)
    
    # 2. Add friends
    for i in range(1, 3):
        friend_id = f"friend_{i}_{uuid4().hex[:6]}"
        
        # Create friend user
        friend_user = User(
            id=friend_id,
            display_name=f"Friend {i}",
            email=f"friend_{i}_{friend_id}@example.com",
            locale="en",
            status="active",
            created_at=datetime.utcnow()
        )
        db.add(friend_user)
        
        # Create accepted friendship
        friendship = UserFriend(
            id=str(uuid4()),
            requester_id=user_id,
            addressee_id=friend_id,
            status="accepted",
            requested_at=datetime.utcnow(),
            responded_at=datetime.utcnow()
        )
        db.add(friendship)
        
    # 3. Add rooms
    for i in range(1, 3):
        room_id = str(uuid4())
        
        # Create room
        room = Room(
            id=room_id,
            title=f"Personal Study Room {i}",
            created_by=user_id,
            status="active",
            created_at=datetime.utcnow()
        )
        db.add(room)
        
        # Add user as member
        member = RoomMember(
            id=str(uuid4()),
            room_id=room_id,
            user_id=user_id,
            display_name=user.display_name,
            role="owner" if i == 1 else "member",
            joined_at=datetime.utcnow()
        )
        db.add(member)

    await db.commit()

    return {
        "message": f"Successfully setup personal mock data for user {user_id}",
        "user_id": user_id,
        "items_created": ["User (if new)", "2 Friends", "2 Rooms"]
    }


@router.delete("/clear", status_code=status.HTTP_200_OK)
async def clear_all_data(db: AsyncSession = Depends(get_db)):
    """
    Clear ALL data from all tables.
    """
    # Disable foreign key checks for easy deletion
    await db.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
    
    tables = [
        "ai_events", "live", "chat_messages", 
        "room_live_session_members", "room_live_sessions", 
        "room_members", "rooms", 
        "dm_messages", "dm_participants", "dm_threads", 
        "user_friends", "auth_tokens", "users"
    ]
    
    for table in tables:
        await db.execute(text(f"TRUNCATE TABLE {table}"))
        
    await db.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
    await db.commit()
    
    return {"message": "All data cleared successfully!"}


@router.post("/user-gen", status_code=status.HTTP_201_CREATED)
async def generate_user_heavy_data(
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db)
):
    """
    **User Data Generator (Heavy)**
    
    Generates a large amount of mock data for the CURRENTLY LOGGED-IN user.
    - 10 New Friends with DM threads (50 msgs each)
    - 5 New Rooms with those friends (Active, 100 msgs each)
    - 2 Past Live Sessions per room with AI events
    """
    
    total_messages_created = 0
    new_friends_count = 0
    new_rooms_count = 0
    
    # 1. Ensure user exists (Object fetch)
    result = await db.execute(select(User).where(User.id == current_user_id))
    user = result.scalar_one_or_none()
    
    if not user:
        # Should be handled by Auth but just in case
        return {"error": "User not found"}
        
    created_friends = []

    # 2. Create 10 Random Friends & DM Threads
    for i in range(10):
        friend_id = f"gen_f_{uuid4().hex[:8]}"
        friend = User(
            id=friend_id,
            display_name=f"Friend {uuid4().hex[:4]}",
            email=f"{friend_id}@example.com",
            locale=random.choice(["en", "ko", "ja"]),
            status="active",
            created_at=datetime.utcnow() - timedelta(days=60)
        )
        db.add(friend)
        created_friends.append(friend)
        
        # Friendship
        friendship = UserFriend(
            id=str(uuid4()),
            requester_id=current_user_id,
            addressee_id=friend_id,
            status="accepted",
            requested_at=datetime.utcnow() - timedelta(days=59),
            responded_at=datetime.utcnow() - timedelta(days=59)
        )
        db.add(friendship)
        
        # DM Thread
        thread = DmThread(
            id=str(uuid4()),
            user_friend_id=friendship.id,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=59)
        )
        db.add(thread)
        
        # Participants
        db.add(DmParticipant(id=str(uuid4()), thread_id=thread.id, user_id=current_user_id, joined_at=thread.created_at))
        db.add(DmParticipant(id=str(uuid4()), thread_id=thread.id, user_id=friend_id, joined_at=thread.created_at))
        
        # 50 DM Messages
        base_time = thread.created_at
        for k in range(50):
            is_me = k % 2 == 0
            sender = user if is_me else friend
            
            msg = DmMessage(
                id=str(uuid4()),
                thread_id=thread.id,
                seq=k+1,
                sender_type="human",
                sender_user_id=sender.id,
                message_type="text",
                text=f"DM Message #{k+1} from {sender.display_name}. Long time no see!",
                lang=sender.locale,
                created_at=base_time + timedelta(hours=k)
            )
            db.add(msg)
            total_messages_created += 1
            
    await db.flush()
    new_friends_count = 10
    
    # 3. Create 5 Rooms with mixed friends
    created_rooms = []
    for i in range(5):
        room_title = f"{user.display_name}'s Project {uuid4().hex[:4]}"
        room = Room(
            id=str(uuid4()),
            title=room_title,
            created_by=user.id,
            status="active",
            created_at=datetime.utcnow() - timedelta(days=30)
        )
        db.add(room)
        created_rooms.append(room)
        
        # Add Self
        me_member = RoomMember(
            id=str(uuid4()),
            room_id=room.id,
            user_id=user.id,
            display_name=user.display_name,
            role="owner",
            joined_at=room.created_at
        )
        db.add(me_member)
        
        # Add 3-5 random friends
        participants = [me_member] # keep list of Member objects
        joined_friends = random.sample(created_friends, k=random.randint(3, 8))
        for f in joined_friends:
            mem = RoomMember(
                id=str(uuid4()),
                room_id=room.id,
                user_id=f.id,
                display_name=f.display_name,
                role="member",
                joined_at=room.created_at
            )
            db.add(mem)
            participants.append(mem)
            
        await db.flush()
        
        # 100 Chat Messages
        base_time = room.created_at + timedelta(minutes=5)
        for k in range(100):
            sender_mem = random.choice(participants) # Pick random member including self
            
            msg = ChatMessage(
                id=str(uuid4()),
                room_id=room.id,
                seq=k+1,
                sender_type="human",
                sender_member_id=sender_mem.id,
                message_type="text",
                text=f"Chat in {room_title}. Msg #{k+1}. Hello @{participants[0].display_name}",
                lang="en", # simplifying
                created_at=base_time + timedelta(hours=k*2)
            )
            db.add(msg)
            total_messages_created += 1
            
        # 2 Past Live Sessions
        for s in range(2):
            ls = RoomLiveSession(
                id=str(uuid4()),
                room_id=room.id,
                title=f"Weekly Sync {s+1}",
                status="ended",
                started_by=user.id,
                started_at=room.created_at + timedelta(days=s*7),
                ended_at=room.created_at + timedelta(days=s*7, hours=1)
            )
            db.add(ls)
            
    await db.commit()
    new_rooms_count = 5

    return {
        "message": f"Generated heavy data for user {user.display_name} ({current_user_id})",
        "created_room_ids": [r.id for r in created_rooms],
        "stats": {
            "new_friends": new_friends_count,
            "new_rooms": new_rooms_count,
            "total_messages": total_messages_created,
            "dms_per_friend": 50,
            "msgs_per_room": 100
        }
    }

