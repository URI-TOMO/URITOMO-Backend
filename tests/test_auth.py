"""
Auth Endpoint Tests
"""

import pytest
from httpx import AsyncClient

# Mock data
USER_EMAIL = "test@example.com"
USER_PWD = "password123"

@pytest.mark.asyncio
async def test_register_and_login(client: AsyncClient):
    # 1. Register
    reg_response = await client.post(
        "/api/v1/auth/register",
        json={
            "email": USER_EMAIL,
            "password": USER_PWD,
            "full_name": "Test User"
        }
    )
    # Check 201 Created or 400 if already exists (re-run)
    assert reg_response.status_code in [201, 400]
    
    if reg_response.status_code == 201:
        data = reg_response.json()
        assert data["email"] == USER_EMAIL
        assert "id" in data

    # 2. Login
    login_response = await client.post(
        "/api/v1/auth/login",
        data={ # OAuth2 uses form data
            "username": USER_EMAIL,
            "password": USER_PWD
        }
    )
    assert login_response.status_code == 200
    token_data = login_response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
