"""
WebSocket Connection Manager
"""

from typing import Dict, List, Set
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        # meeting_id -> list of active websockets
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, meeting_id: int):
        await websocket.accept()
        if meeting_id not in self.active_connections:
            self.active_connections[meeting_id] = set()
        self.active_connections[meeting_id].add(websocket)

    def disconnect(self, websocket: WebSocket, meeting_id: int):
        if meeting_id in self.active_connections:
            self.active_connections[meeting_id].discard(websocket)
            if not self.active_connections[meeting_id]:
                del self.active_connections[meeting_id]

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        await websocket.send_json(message)

    async def broadcast(self, message: dict, meeting_id: int):
        if meeting_id in self.active_connections:
            for connection in self.active_connections[meeting_id]:
                await connection.send_json(message)


manager = ConnectionManager()
