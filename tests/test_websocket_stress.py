"""
Stress test for WebSocket personal chat system
Tests high-concurrency scenarios and race conditions
"""
import asyncio
import pytest
from unittest.mock import AsyncMock
from typing import List, Dict, Any


class ConnectionManager:
    """Copy from personal_chat.py for testing"""
    
    def __init__(self):
        self.active_connections: Dict[int, List[Any]] = {}
        self._lock = asyncio.Lock()
    
    async def connect(self, user_id: int, websocket) -> None:
        if hasattr(websocket, 'accept'):
            await websocket.accept()
        
        async with self._lock:
            if user_id not in self.active_connections:
                self.active_connections[user_id] = []
            self.active_connections[user_id].append(websocket)
    
    async def disconnect(self, user_id: int, websocket) -> None:
        async with self._lock:
            if user_id in self.active_connections:
                if websocket in self.active_connections[user_id]:
                    self.active_connections[user_id].remove(websocket)
                
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
    
    async def send_to_user(self, user_id: int, message: dict) -> bool:
        async with self._lock:
            connections = list(self.active_connections.get(user_id, []))
        
        if not connections:
            return False
        
        send_tasks = []
        for connection in connections:
            send_tasks.append(self._safe_send(connection, message))
        
        results = await asyncio.gather(*send_tasks, return_exceptions=True)
        return any(result is True for result in results)
    
    async def _safe_send(self, websocket, message: dict) -> bool:
        try:
            await websocket.send_json(message)
            return True
        except Exception:
            return False


@pytest.mark.asyncio
async def test_high_concurrency_multiple_users():
    """
    Test 100 users with 5 connections each = 500 total connections
    """
    manager = ConnectionManager()
    num_users = 100
    connections_per_user = 5
    
    # Create connections
    user_websockets = {}
    for user_id in range(num_users):
        user_websockets[user_id] = [AsyncMock() for _ in range(connections_per_user)]
    
    # Connect all users concurrently
    connect_tasks = []
    for user_id, ws_list in user_websockets.items():
        for ws in ws_list:
            connect_tasks.append(manager.connect(user_id, ws))
    
    await asyncio.gather(*connect_tasks)
    
    # Verify all connected
    assert len(manager.active_connections) == num_users
    for user_id in range(num_users):
        assert len(manager.active_connections[user_id]) == connections_per_user
    
    print(f"? Successfully connected {num_users * connections_per_user} WebSockets")


@pytest.mark.asyncio
async def test_concurrent_send_to_multiple_users():
    """
    Test sending messages to multiple users simultaneously
    Simulates chat room broadcast scenario
    """
    manager = ConnectionManager()
    num_users = 50
    
    # Setup users
    for user_id in range(num_users):
        ws = AsyncMock()
        await manager.connect(user_id, ws)
    
    # Send to all users concurrently
    message = {"type": "broadcast", "content": "Hello everyone!"}
    send_tasks = [
        manager.send_to_user(user_id, message) 
        for user_id in range(num_users)
    ]
    
    results = await asyncio.gather(*send_tasks)
    
    # All should succeed
    assert all(results)
    print(f"? Successfully broadcast to {num_users} users")


@pytest.mark.asyncio
async def test_rapid_connect_disconnect_same_user():
    """
    Test rapid connect/disconnect cycles for the same user
    Simulates unstable network connection
    """
    manager = ConnectionManager()
    user_id = 1
    cycles = 200
    
    async def cycle():
        ws = AsyncMock()
        await manager.connect(user_id, ws)
        await asyncio.sleep(0.001)  # Small delay
        await manager.disconnect(user_id, ws)
    
    tasks = [cycle() for _ in range(cycles)]
    await asyncio.gather(*tasks)
    
    # Should be clean at the end
    assert user_id not in manager.active_connections
    print(f"? Completed {cycles} rapid connect/disconnect cycles")


@pytest.mark.asyncio
async def test_send_during_mass_disconnect():
    """
    Test sending messages while many users are disconnecting
    Critical race condition test
    """
    manager = ConnectionManager()
    num_users = 100
    
    # Connect all users
    for user_id in range(num_users):
        ws = AsyncMock()
        await manager.connect(user_id, ws)
    
    async def disconnect_all():
        """Disconnect all users gradually"""
        for user_id in range(num_users):
            if user_id in manager.active_connections:
                ws = manager.active_connections[user_id][0]
                await manager.disconnect(user_id, ws)
                await asyncio.sleep(0.001)
    
    async def send_to_all():
        """Try sending to all users while they're disconnecting"""
        message = {"test": "data"}
        for _ in range(10):
            send_tasks = [
                manager.send_to_user(user_id, message) 
                for user_id in range(num_users)
            ]
            await asyncio.gather(*send_tasks, return_exceptions=True)
            await asyncio.sleep(0.005)
    
    # Run both operations concurrently
    await asyncio.gather(
        disconnect_all(),
        send_to_all()
    )
    
    # Should be empty at the end
    assert len(manager.active_connections) == 0
    print("? Successfully handled concurrent send during mass disconnect")


@pytest.mark.asyncio
async def test_multiple_connections_per_user_send():
    """
    Test that messages are delivered to all connections of a user
    """
    manager = ConnectionManager()
    user_id = 1
    num_connections = 10
    
    websockets = [AsyncMock() for _ in range(num_connections)]
    
    # Connect all websockets for same user
    for ws in websockets:
        await manager.connect(user_id, ws)
    
    # Send one message
    message = {"from": 2, "message": "Hello!"}
    result = await manager.send_to_user(user_id, message)
    
    assert result is True
    
    # Verify all connections received the message
    for ws in websockets:
        ws.send_json.assert_called_once_with(message)
    
    print(f"? Message delivered to all {num_connections} connections")


@pytest.mark.asyncio
async def test_send_to_offline_user():
    """
    Test sending to a user who is not connected
    """
    manager = ConnectionManager()
    
    # Send to non-existent user
    result = await manager.send_to_user(999, {"msg": "test"})
    
    assert result is False
    print("? Correctly handled send to offline user")


@pytest.mark.asyncio
async def test_websocket_send_failure_handling():
    """
    Test that failures in websocket.send_json are handled gracefully
    """
    manager = ConnectionManager()
    user_id = 1
    
    # Create a websocket that fails on send
    failing_ws = AsyncMock()
    failing_ws.send_json.side_effect = Exception("Connection broken")
    
    working_ws = AsyncMock()
    
    await manager.connect(user_id, failing_ws)
    await manager.connect(user_id, working_ws)
    
    # Send message
    result = await manager.send_to_user(user_id, {"msg": "test"})
    
    # Should still succeed because one connection works
    assert result is True
    working_ws.send_json.assert_called_once()
    
    print("? Gracefully handled WebSocket send failure")


async def run_all_stress_tests():
    """Run all stress tests"""
    print("\n" + "="*60)
    print("WebSocket Personal Chat - Stress Test Suite")
    print("="*60 + "\n")
    
    tests = [
        ("High Concurrency (500 connections)", test_high_concurrency_multiple_users),
        ("Concurrent Broadcast", test_concurrent_send_to_multiple_users),
        ("Rapid Connect/Disconnect", test_rapid_connect_disconnect_same_user),
        ("Send During Mass Disconnect", test_send_during_mass_disconnect),
        ("Multiple Connections Per User", test_multiple_connections_per_user_send),
        ("Offline User", test_send_to_offline_user),
        ("WebSocket Failure Handling", test_websocket_send_failure_handling),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"Running: {test_name}...", end=" ")
        try:
            await test_func()
            passed += 1
            print("? PASS")
        except AssertionError as e:
            failed += 1
            print(f"? FAIL: {e}")
        except Exception as e:
            failed += 1
            print(f"? ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_stress_tests())
    exit(0 if success else 1)
