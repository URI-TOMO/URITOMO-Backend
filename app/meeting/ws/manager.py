from typing import Dict, List, Set
from fastapi import WebSocket

import logging

logger = logging.getLogger("uritomo.ws")

class ConnectionManager:
    def __init__(self):
        # room_id -> List[WebSocket]
        self.active_rooms: Dict[str, List[WebSocket]] = {}
        # room_id -> Set[user_id]
        self.room_users: Dict[str, Set[str]] = {}

    async def connect(self, room_id: str, websocket: WebSocket, user_id: str = None):
        await websocket.accept()
        if room_id not in self.active_rooms:
            self.active_rooms[room_id] = []
            self.room_users[room_id] = set()
        
        self.active_rooms[room_id].append(websocket)
        if user_id:
            self.room_users[room_id].add(user_id)
            
        if user_id:
            logger.info(f"💬 CHAT WS Connected | Room: {room_id} | User: {user_id} | Total Connections in Room: {len(self.active_rooms[room_id])}")
        else:
            logger.info(f"🔌 WS Connected | Room: {room_id} | User: {user_id} | Total Connections in Room: {len(self.active_rooms[room_id])}")

    def disconnect(self, room_id: str, websocket: WebSocket, user_id: str = None):
        if room_id in self.active_rooms:
            if websocket in self.active_rooms[room_id]:
                self.active_rooms[room_id].remove(websocket)
            if not self.active_rooms[room_id]:
                del self.active_rooms[room_id]
                if room_id in self.room_users:
                    del self.room_users[room_id]
        
        # Note: We simply remove user from set if needed, but since it's a set of distinct user_ids, 
        # we strictly shouldn't remove it if they have another tab open. 
        # For simplicity in this debug view, we won't strictly manage the set on disconnect 
        # unless we track connection counts per user.
        
        if user_id:
            logger.info(f"💬 CHAT WS Disconnected | Room: {room_id} | User: {user_id}")
        else:
            logger.info(f"🔌 WS Disconnected | Room: {room_id} | User: {user_id}")

    async def broadcast(self, room_id: str, message: dict):
        if room_id in self.active_rooms:
            count = len(self.active_rooms[room_id])
            try:
                import json
                payload = json.dumps(message, ensure_ascii=False, default=str)
            except Exception:
                payload = str(message)
            logger.info(f"🩵🩵🩵🩵🩵🩵 WS Broadcast | Room: {room_id} | Targets: {count} | Payload: {payload}")
            for connection in self.active_rooms[room_id]:
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
        for room_id, sockets in self.active_rooms.items():
            users = list(self.room_users.get(room_id, set()))
            stats[room_id] = {
                "active_connections_count": len(sockets),
                "active_users": users,
                "active_users_count": len(users)
            }
        return stats

manager = ConnectionManager()
