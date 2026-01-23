from typing import Dict, List, Set
from fastapi import WebSocket

import logging

logger = logging.getLogger("uritomo.ws")

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
            
        logger.info(f"WS Connected | Session: {session_id} | User: {user_id} | Total Connections in Session: {len(self.active_sessions[session_id])}")

    def disconnect(self, session_id: str, websocket: WebSocket, user_id: str = None):
        if session_id in self.active_sessions:
            if websocket in self.active_sessions[session_id]:
                self.active_sessions[session_id].remove(websocket)
            if not self.active_sessions[session_id]:
                del self.active_sessions[session_id]
                if session_id in self.session_users:
                    del self.session_users[session_id]
        
        # Note: We simply remove user from set if needed, but since it's a set of distinct user_ids, 
        # we strictly shouldn't remove it if they have another tab open. 
        # For simplicity in this debug view, we won't strictly manage the set on disconnect 
        # unless we track connection counts per user.
        
        logger.info(f"WS Disconnected | Session: {session_id} | User: {user_id}")

    async def broadcast(self, session_id: str, message: dict):
        if session_id in self.active_sessions:
            count = len(self.active_sessions[session_id])
            logger.debug(f"WS Broadcast | Session: {session_id} | Targets: {count} | MsgType: {message.get('type')}")
            for connection in self.active_sessions[session_id]:
                try:
                    await connection.send_json(message)
                except:
                    # Connection might be dead
                    pass

    def get_stats(self):
        """
        Return a snapshot of all active connections
        """
        stats = {}
        for session_id, sockets in self.active_sessions.items():
            users = list(self.session_users.get(session_id, set()))
            stats[session_id] = {
                "active_connections_count": len(sockets),
                "active_users": users,
                "active_users_count": len(users)
            }
        return stats

manager = ConnectionManager()
