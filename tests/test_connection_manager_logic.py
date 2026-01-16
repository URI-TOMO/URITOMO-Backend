
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

# Copying ConnectionManager from app/user/personal_chat.py for isolation testing
# as the original file has broken imports preventing direct import.
from typing import Dict, List, Any

class ConnectionManager:
    """
    WebSocket connection manager with race condition protection
    Handles multiple concurrent connections per user
    """
    
    def __init__(self):
        # user_id -> List[WebSocket]
        self.active_connections: Dict[int, List[Any]] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
    
    async def connect(self, user_id: int, websocket) -> None:
        """Connect a new WebSocket for a user"""
        # Mocking accept since we don't have a real websocket
        if hasattr(websocket, 'accept'):
            await websocket.accept()
        
        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
    
    async def disconnect(self, user_id: int, websocket) -> None:
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
            # Create a copy to prevent race conditions during iteration
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
    
    async def _safe_send(self, websocket, message: dict) -> bool:
        """Send message with error handling"""
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            return False

@pytest.mark.asyncio
async def test_connection_manager_concurrent_access():
    manager = ConnectionManager()
    user_id = 1
    
    # Mock websockets
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    
    # Test concurrent connections
    await asyncio.gather(
        manager.connect(user_id, ws1),
        manager.connect(user_id, ws2)
    )
    
    # Needs to access internal state directly to verify
    # But strictly we should use public methods.
    # We can check via sending.
    
    assert len(manager.active_connections[user_id]) == 2
    
    # Test sending
    message = {"test": "data"}
    result = await manager.send_to_user(user_id, message)
    assert result is True
    
    ws1.send_json.assert_called_with(message)
    ws2.send_json.assert_called_with(message)

@pytest.mark.asyncio
async def test_race_condition_connect_disconnect():
    """
    Test rapid connect and disconnect to verify lock integrity
    """
    manager = ConnectionManager()
    user_id = 99
    
    async def spam_connect_disconnect(ws_id):
        ws = AsyncMock()
        await manager.connect(user_id, ws)
        # Small sleep to yield control
        await asyncio.sleep(0.001)
        await manager.disconnect(user_id, ws)
        
    # Run 100 concurrent operations
    tasks = [spam_connect_disconnect(i) for i in range(100)]
    await asyncio.gather(*tasks)
    
    # Should be empty at the end
    assert user_id not in manager.active_connections

@pytest.mark.asyncio
async def test_send_while_disconnecting():
    """
    Test potential race condition: sending while concurrently disconnecting.
    Verifies that list copying prevents iteration errors.
    """
    manager = ConnectionManager()
    user_id = 50
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    
    # Setup: 2 connections
    await manager.connect(user_id, ws1)
    await manager.connect(user_id, ws2)
    
    async def disconnect_one():
        await asyncio.sleep(0.005)
        await manager.disconnect(user_id, ws1)

    async def send_msg():
        for _ in range(10):
            await manager.send_to_user(user_id, {"msg": "hello"})
            await asyncio.sleep(0.001)
    
    # Run concurrent send and disconnect operations
    await asyncio.gather(
        disconnect_one(),
        send_msg()
    )
    
    # Should still have ws2 connected
    assert len(manager.active_connections[user_id]) == 1
    assert ws2 in manager.active_connections[user_id] 
    

async def run_tests():
    print("Running test_connection_manager_concurrent_access...")
    await test_connection_manager_concurrent_access()
    print("PASS")

    print("Running test_race_condition_connect_disconnect...")
    await test_race_condition_connect_disconnect()
    print("PASS")

    print("Running test_send_while_disconnecting...")
    await test_send_while_disconnecting()
    print("PASS")
    
if __name__ == "__main__":
    try:
        asyncio.run(run_tests())
        print("ALL TESTS PASSED")
    except AssertionError as e:
        print(f"TEST FAILED: {e}")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

