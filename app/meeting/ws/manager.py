from typing import Dict, List, Set
from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        # session_id -> List[WebSocket]
        self.active_sessions: Dict[str, List[WebSocket]] = {}
        # session_id -> Set[user_id]
        self.session_users: Dict[str, Set[str]] = {}

    async def connect(self, session_id: str, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = []
            self.session_users[session_id] = set()
        
        self.active_sessions[session_id].append(websocket)
        if user_id:
            self.session_users[session_id].add(user_id)

    def disconnect(self, session_id: str, websocket: WebSocket, user_id: str = None):
        if session_id in self.active_sessions:
            if websocket in self.active_sessions[session_id]:
                self.active_sessions[session_id].remove(websocket)
            if not self.active_sessions[session_id]:
                del self.active_sessions[session_id]
        
        if user_id and session_id in self.session_users:
            # Note: This is simplified. One user might have multiple connections.
            # For now we'll just keep it simple.
            pass

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_sessions:
            for connection in self.active_sessions[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    # Connection might be dead
                    pass

manager = ConnectionManager()
