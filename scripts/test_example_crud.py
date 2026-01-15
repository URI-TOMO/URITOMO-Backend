#!/usr/bin/env python3
"""
Test the Example CRUD API endpoints.
This script demonstrates how to use the User and Room CRUD APIs.
"""

import asyncio
import httpx
from typing import Optional


BASE_URL = "http://localhost:8000/api/v1/example"


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str):
    print(f"{Colors.OKGREEN}✅ {message}{Colors.ENDC}")


def print_error(message: str):
    print(f"{Colors.FAIL}❌ {message}{Colors.ENDC}")


def print_info(message: str):
    print(f"{Colors.OKCYAN}ℹ️  {message}{Colors.ENDC}")


def print_header(message: str):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*60}")
    print(f"  {message}")
    print(f"{'='*60}{Colors.ENDC}\n")


async def test_user_crud(client: httpx.AsyncClient):
    """Test User CRUD operations"""
    
    print_header("Testing User CRUD Operations")
    
    # 1. Create User
    print_info("Creating a new user...")
    user_data = {
        "id": "test_user_001",
        "email": "test@example.com",
        "display_name": "Test User",
        "locale": "ko",
        "status": "active"
    }
    
    response = await client.post(f"{BASE_URL}/users", json=user_data)
    if response.status_code == 201:
        print_success(f"User created: {response.json()}")
    else:
        print_error(f"Failed to create user: {response.text}")
        return None
    
    # 2. Get User
    print_info("Retrieving user...")
    response = await client.get(f"{BASE_URL}/users/test_user_001")
    if response.status_code == 200:
        print_success(f"User retrieved: {response.json()}")
    else:
        print_error(f"Failed to retrieve user: {response.text}")
    
    # 3. Update User
    print_info("Updating user...")
    update_data = {
        "display_name": "Updated Test User",
        "locale": "en"
    }
    response = await client.patch(f"{BASE_URL}/users/test_user_001", json=update_data)
    if response.status_code == 200:
        print_success(f"User updated: {response.json()}")
    else:
        print_error(f"Failed to update user: {response.text}")
    
    # 4. Get All Users
    print_info("Retrieving all users...")
    response = await client.get(f"{BASE_URL}/users?skip=0&limit=10")
    if response.status_code == 200:
        users = response.json()
        print_success(f"Retrieved {len(users)} users")
    else:
        print_error(f"Failed to retrieve users: {response.text}")
    
    return "test_user_001"


async def test_room_crud(client: httpx.AsyncClient, user_id: str):
    """Test Room CRUD operations"""
    
    print_header("Testing Room CRUD Operations")
    
    # 1. Create Room
    print_info("Creating a new room...")
    room_data = {
        "id": "test_room_001",
        "title": "Test Room",
        "created_by": user_id,
        "status": "active"
    }
    
    response = await client.post(f"{BASE_URL}/rooms", json=room_data)
    if response.status_code == 201:
        print_success(f"Room created: {response.json()}")
    else:
        print_error(f"Failed to create room: {response.text}")
        return
    
    # 2. Get Room
    print_info("Retrieving room...")
    response = await client.get(f"{BASE_URL}/rooms/test_room_001")
    if response.status_code == 200:
        print_success(f"Room retrieved: {response.json()}")
    else:
        print_error(f"Failed to retrieve room: {response.text}")
    
    # 3. Update Room
    print_info("Updating room...")
    update_data = {
        "title": "Updated Test Room"
    }
    response = await client.patch(f"{BASE_URL}/rooms/test_room_001", json=update_data)
    if response.status_code == 200:
        print_success(f"Room updated: {response.json()}")
    else:
        print_error(f"Failed to update room: {response.text}")
    
    # 4. Get All Rooms
    print_info("Retrieving all rooms...")
    response = await client.get(f"{BASE_URL}/rooms?skip=0&limit=10")
    if response.status_code == 200:
        rooms = response.json()
        print_success(f"Retrieved {len(rooms)} rooms")
    else:
        print_error(f"Failed to retrieve rooms: {response.text}")
    
    # 5. Get Rooms by Creator
    print_info(f"Retrieving rooms created by {user_id}...")
    response = await client.get(f"{BASE_URL}/rooms?creator_id={user_id}")
    if response.status_code == 200:
        rooms = response.json()
        print_success(f"Retrieved {len(rooms)} rooms by creator")
    else:
        print_error(f"Failed to retrieve rooms by creator: {response.text}")
    
    # 6. Soft Delete Room
    print_info("Soft deleting room...")
    response = await client.delete(f"{BASE_URL}/rooms/test_room_001")
    if response.status_code == 200:
        print_success(f"Room soft deleted: {response.json()}")
    else:
        print_error(f"Failed to soft delete room: {response.text}")


async def cleanup(client: httpx.AsyncClient):
    """Clean up test data"""
    
    print_header("Cleaning Up Test Data")
    
    # Delete room (hard delete)
    print_info("Hard deleting room...")
    response = await client.delete(f"{BASE_URL}/rooms/test_room_001?hard=true")
    if response.status_code == 200:
        print_success("Room deleted")
    else:
        print_info("Room already deleted or not found")
    
    # Delete user
    print_info("Deleting user...")
    response = await client.delete(f"{BASE_URL}/users/test_user_001")
    if response.status_code == 200:
        print_success("User deleted")
    else:
        print_info("User already deleted or not found")


async def main():
    """Main test function"""
    
    print(f"{Colors.BOLD}{Colors.HEADER}")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║         URITOMO Example CRUD API Test Suite               ║")
    print("╚════════════════════════════════════════════════════════════╝")
    print(f"{Colors.ENDC}")
    
    print_info(f"Testing API at: {BASE_URL}")
    print_info("Make sure the API server is running!")
    print()
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            # Test if server is running
            response = await client.get("http://localhost:8000/docs")
            if response.status_code != 200:
                print_error("API server is not running!")
                print_info("Start the server with: uvicorn app.main:app --reload")
                return
        except httpx.ConnectError:
            print_error("Cannot connect to API server!")
            print_info("Start the server with: uvicorn app.main:app --reload")
            return
        
        try:
            # Run tests
            user_id = await test_user_crud(client)
            
            if user_id:
                await test_room_crud(client, user_id)
                await cleanup(client)
            
            print_header("Test Suite Completed")
            print_success("All tests completed successfully!")
            
        except Exception as e:
            print_error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
