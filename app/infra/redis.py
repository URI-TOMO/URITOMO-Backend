"""
Redis infrastructure configuration

Redis client management.
"""

from typing import AsyncGenerator, Optional

from redis import asyncio as aioredis
from redis.asyncio import Redis

from app.core.config import settings

# Global redis pool
pool: Optional[aioredis.ConnectionPool] = None


async def init_redis_pool():
    """Initialize Redis connection pool"""
    global pool
    pool = aioredis.ConnectionPool.from_url(
        settings.redis_url,
        db=settings.redis_db,
        encoding="utf-8",
        decode_responses=True,
    )


async def close_redis_pool():
    """Close Redis connection pool"""
    global pool
    if pool:
        await pool.disconnect()


async def get_redis() -> AsyncGenerator[Redis, None]:
    """
    Dependency for getting Redis client.
    Uses connection pool.
    """
    if pool is None:
        await init_redis_pool()
    
    client = aioredis.Redis(connection_pool=pool)
    try:
        yield client
    finally:
        await client.close()
