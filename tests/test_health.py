"""
Health Endpoint Tests
"""

import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "api" in data
    assert data["api"] == "ok"
    # Note: DB/Redis might show error in mock test environment if not mocked
