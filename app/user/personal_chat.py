"""
Personal Chat System - WebSocket-based Real-time Chat
Refactored for Room-based Architecture

Features:
- Real-time WebSocket communication
- JWT authentication
- Multiple concurrent connections
- Race condition protection (asyncio.Lock)
- Automatic Room creation for 1-on-1 chats
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query, status
from sqlalchemy import select, or_, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.deps import get_db, get_current_user_id
from app.core.config import settings
from app.core import security
from app.models.user import User
from app.models.message import ChatMessage
from app.models.room import Room, RoomMember

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["personal_chat"])


class ConnectionManager:
    """
    WebSocket connection manager with race condition protection
    Handles multiple concurrent connections per user
    """
    
    def __init__(self):
        # user_id (str) -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, user_id: str, websocket: WebSocket) -> None:
        """Connect a new WebSocket for a user"""
        await websocket.accept()
        
        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
            logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections.get(user_id, []))}")
    
    async def disconnect(self, user_id: str, websocket: WebSocket) -> None:
        """Disconnect a WebSocket for a user"""
        async with self._lock:
            if user_id in self.active_connections:
                if websocket in self.active_connections[user_id]:
                    self.active_connections[user_id].remove(websocket)
                
                # Clean up empty lists
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            logger.info(f"User {user_id} disconnected")
    
    async def send_to_user(self, user_id: str, message: dict) -> bool:
        """
        Send message to all connections of a specific user
        Returns True if message was sent to at least one connection
        """
        async with self._lock:
            connections = self.active_connections.get(user_id, [])
            # Create a shallow copy to iterate safely
            connections = list(connections)
        
        if not connections:
            return False
        
        # Send to all connections concurrently
        send_tasks = []
        for connection in connections:
            send_tasks.append(self._safe_send(connection, message))
        
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        
        # Return True if at least one message was sent successfully
        return any(result is True for result in results)
    
    async def _safe_send(self, websocket: WebSocket, message: dict) -> bool:
        """Send message with error handling"""
        try:
            await websocket.send_json(message)
            return True
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return False


# Global connection manager instance
manager = ConnectionManager()


async def get_or_create_personal_room(db: AsyncSession, user1_id: str, user2_id: str) -> Room:
    """
    Find existing 1-on-1 room between two users or create a new one.
    """
    # 1. Try to find existing room with exactly these 2 members
    # This is complex in SQL. Simplified approach: find rooms where user1 is member, 
    # then check if user2 is also a member and count is 2.
    
    # Subquery to find rooms with exactly 2 members
    subquery = (
        select(RoomMember.room_id)
        .group_by(RoomMember.room_id)
        .having(func.count(RoomMember.room_id) == 2)
        .scalar_subquery()
    )
    
    # Find rooms where user1 is a member AND room is in subquery
    stmt = (
        select(Room)
        .join(RoomMember)
        .where(
            and_(
                RoomMember.user_id == user1_id,
                Room.id.in_(subquery)
            )
        )
        .options(selectinload(Room.members))
    )
    
    result = await db.execute(stmt)
    candidate_rooms = result.scalars().all()
    
    for room in candidate_rooms:
        member_ids = {m.user_id for m in room.members}
        if user2_id in member_ids:
            return room

    # 2. Create new room if not found
    new_room_id = str(uuid.uuid4())
    new_room = Room(
        id=new_room_id,
        created_by=user1_id,
        title="Personal Chat",
        status="active"
    )
    db.add(new_room)
    
    # Add members
    member1 = RoomMember(
        id=str(uuid.uuid4()),
        room_id=new_room_id,
        user_id=user1_id,
        display_name="User", # Should be fetched from User table ideally
        role="member"
    )
    member2 = RoomMember(
        id=str(uuid.uuid4()),
        room_id=new_room_id,
        user_id=user2_id,
        display_name="User", # Should be fetched
        role="member"
    )
    db.add(member1)
    db.add(member2)
    
    await db.commit()
    await db.refresh(new_room)
    # Re-fetch with members to be safe
    stmt = select(Room).where(Room.id == new_room.id).options(selectinload(Room.members))
    result = await db.execute(stmt)
    return result.scalar_one()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: AsyncSession = Depends(get_db)
):
    """
    WebSocket endpoint for real-time chat
    """
    # Authenticate
    try:
        user_id = security.verify_token(token)
        if not user_id:
            logger.warning("Invalid token in WebSocket connection")
            await websocket.close(code=1008, reason="Invalid token")
            return
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    # Connect user
    await manager.connect(user_id, websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "status": "connected",
            "message": f"Connected as user {user_id}"
        })
        
        while True:
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                
                # Check message type
                to_user_id = message_data.get("to")
                room_id = message_data.get("room_id")
                content = message_data.get("message")
                
                if not content:
                     await websocket.send_json({"status": "error", "message": "Content required"})
                     continue

                target_room = None
                
                # Case 1: Direct Message to User (Find/Create Room)
                if to_user_id:
                    target_room = await get_or_create_personal_room(db, user_id, to_user_id)
                # Case 2: Message to specific Room
                elif room_id:
                    result = await db.execute(select(Room).where(Room.id == room_id).options(selectinload(Room.members)))
                    target_room = result.scalar_one_or_none()
                    if not target_room:
                        await websocket.send_json({"status": "error", "message": "Room not found"})
                        continue
                else:
                    await websocket.send_json({"status": "error", "message": "Specify 'to' or 'room_id'"})
                    continue
                
                # Verify membership
                sender_member = next((m for m in target_room.members if m.user_id == user_id), None)
                if not sender_member:
                    await websocket.send_json({"status": "error", "message": "You are not a member of this room"})
                    continue

                # Save message
                new_msg = ChatMessage(
                    id=str(uuid.uuid4()),
                    room_id=target_room.id,
                    seq=int(datetime.utcnow().timestamp() * 1000), # Simple seq generation
                    sender_type="participant",
                    sender_member_id=sender_member.id,
                    message_type="manual",
                    text=content,
                    created_at=datetime.utcnow()
                )
                db.add(new_msg)
                await db.commit()
                
                # Prepare payload
                payload = {
                    "event": "new_message",
                    "room_id": target_room.id,
                    "from_user_id": user_id,
                    "message": content,
                    "message_id": new_msg.id,
                    "created_at": new_msg.created_at.isoformat()
                }
                
                # Broadcast to ALL room members (except sender? usually sender wants ack)
                # Here we send to everyone in the room found in the DB member list
                # This assumes they are connected to THIS server instance.
                for member in target_room.members:
                    if member.user_id:
                        await manager.send_to_user(member.user_id, payload)
                        
                # Acknowledge to sender if not covered above (if send_to_user includes sender)
                # manager.send_to_user includes the sender if they are in the member list.
                
            except json.JSONDecodeError:
                await websocket.send_json({"status": "error", "message": "Invalid JSON"})
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                await websocket.send_json({"status": "error", "message": "Internal error"})
                
    except WebSocketDisconnect:
        await manager.disconnect(user_id, websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(user_id, websocket)


@router.get("/history/{user_id}")
async def get_direct_history(
    target_user_id: str,
    current_user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
    limit: int = 50
):
    """
    Get history for a direct chat with another user
    """
    # Find the room
    # Reuse logical query or assume we can find it. 
    # For now, simplistic: find common room
    
    # ... (Reuse get_or_create logic or similar query) ...
    # To keep it simple for now, we'll implement a basic query
    
    # TODO: Implement full history retrieval
    return {"status": "not_implemented_yet", "message": "Use /rooms endpoints for full history"}