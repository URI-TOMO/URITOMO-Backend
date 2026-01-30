"""
Personal Chat System - WebSocket-based Real-time Chat

Features:
- Real-time WebSocket communication
- JWT authentication
- Multiple concurrent connections
- Race condition protection (asyncio.Lock)
- Error handling
- Message history
- Read status tracking
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
import asyncio
import json
from datetime import datetime

from app.core.deps import get_db, get_current_user_id
from app.models.message_personal import Message

router = APIRouter(prefix="/chat", tags=["personal_chat"])


class ConnectionManager:
    """
    WebSocket connection manager with race condition protection
    Handles multiple concurrent connections per user
    """
    
    def __init__(self):
        # user_id -> List[WebSocket]
        self.active_connections: Dict[int, List[WebSocket]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, user_id: int, websocket: WebSocket) -> None:
        """Connect a new WebSocket for a user"""
        await websocket.accept()
        
        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
    
    async def disconnect(self, user_id: int, websocket: WebSocket) -> None:
        """Disconnect a WebSocket for a user"""
        async with self._lock:
            if user_id in self.active_connections:
                if websocket in self.active_connections[user_id]:
                    self.active_connections[user_id].remove(websocket)
                
                # Clean up empty lists
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
    
    async def send_to_user(self, user_id: int, message: dict) -> bool:
        """
        Send message to all connections of a specific user
        Returns True if message was sent to at least one connection
        """
        async with self._lock:
            connections = self.active_connections.get(user_id, [])
        
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
            print(f"Error sending message: {e}")
            return False


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time chat
    
    Usage:
        ws://localhost:8000/ws?token=<jwt_token>
    
    Message format (client -> server):
        {
            "to": <user_id>,
            "message": "<message_content>"
        }
    
    Message format (server -> client):
        {
            "from": <sender_id>,
            "message": "<message_content>",
            "created_at": "<timestamp>"
        }
    """
    # Authenticate user from token
    try:
        # TODO: Implement token verification logic
        # For now, using placeholder user_id extraction
        from jose import jwt
        from app.core.config import settings
        
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=["HS256"])
        user_id = int(payload.get("sub"))
    except Exception as e:
        await websocket.close(code=1008, reason="Invalid token")
        return
    
    # Connect user
    await manager.connect(user_id, websocket)
    
    try:
        # Send welcome message
        await websocket.send_json({
            "status": "connected",
            "message": f"Connected as user {user_id}"
        })
        
        # Message receiving loop
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                to_user_id = message_data.get("to")
                message_content = message_data.get("message")
                
                if not to_user_id or not message_content:
                    await websocket.send_json({
                        "status": "error",
                        "message": "Invalid message format. Required: {to: <user_id>, message: <text>}"
                    })
                    continue
                
                # Save message to database
                db_message = Message(
                    sender_id=user_id,
                    receiver_id=to_user_id,
                    content=message_content,
                    is_read=False,
                    created_at=datetime.utcnow()
                )
                db.add(db_message)
                db.commit()
                db.refresh(db_message)
                
                # Prepare message for recipient
                outgoing_message = {
                    "from": user_id,
                    "message": message_content,
                    "created_at": db_message.created_at.isoformat(),
                    "message_id": db_message.id
                }
                
                # Send to recipient
                sent = await manager.send_to_user(to_user_id, outgoing_message)
                
                # Send confirmation to sender
                if sent:
                    await websocket.send_json({
                        "status": "sent",
                        "message": f"Message delivered to user {to_user_id}",
                        "message_id": db_message.id
                    })
                else:
                    await websocket.send_json({
                        "status": "offline",
                        "message": f"User {to_user_id} is offline. Message saved.",
                        "message_id": db_message.id
                    })
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "status": "error",
                    "message": "Invalid JSON format"
                })
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "message": f"Error processing message: {str(e)}"
                })
                
    except WebSocketDisconnect:
        await manager.disconnect(user_id, websocket)
        print(f"User {user_id} disconnected")
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
        await manager.disconnect(user_id, websocket)


@router.get("/history/{target_user_id}")
async def get_chat_history(
    target_user_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db),
    limit: int = Query(50, ge=1, le=100)
):
    """
    Get chat history between current user and target user
    
    Returns:
        List of messages (most recent first)
    """
    messages = db.query(Message).filter(
        (
            (Message.sender_id == int(current_user_id)) & 
            (Message.receiver_id == target_user_id)
        ) | (
            (Message.sender_id == target_user_id) & 
            (Message.receiver_id == int(current_user_id))
        )
    ).order_by(Message.created_at.desc()).limit(limit).all()
    
    return [
        {
            "id": msg.id,
            "sender_id": msg.sender_id,
            "receiver_id": msg.receiver_id,
            "content": msg.content,
            "is_read": msg.is_read,
            "created_at": msg.created_at.isoformat()
        }
        for msg in messages
    ]


@router.post("/read/{from_user_id}")
async def mark_messages_as_read(
    from_user_id: int,
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Mark all messages from a specific user as read
    
    Returns:
        Number of messages marked as read
    """
    updated = db.query(Message).filter(
        Message.sender_id == from_user_id,
        Message.receiver_id == int(current_user_id),
        Message.is_read == False
    ).update({"is_read": True})
    
    db.commit()
    
    return {
        "status": "success",
        "marked_as_read": updated
    }


@router.get("/unread/count")
async def get_unread_count(
    current_user_id: str = Depends(get_current_user_id),
    db: Session = Depends(get_db)
):
    """
    Get count of unread messages for current user
    
    Returns:
        Total unread count and per-user breakdown
    """
    unread_messages = db.query(
        Message.sender_id,
        db.func.count(Message.id).label("count")
    ).filter(
        Message.receiver_id == int(current_user_id),
        Message.is_read == False
    ).group_by(Message.sender_id).all()
    
    total = sum(msg.count for msg in unread_messages)
    
    return {
        "total": total,
        "by_user": {
            msg.sender_id: msg.count 
            for msg in unread_messages
        }
    }