"""
Dependency Injection

FastAPI dependencies for routes.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.token import security_scheme, get_current_user_id, CurrentUserDep
from app.infra.db import get_db
from app.infra.redis import get_redis
from app.infra.qdrant import get_qdrant
from app.infra.queue import JobQueue, QueueFactory

# Type aliases for common dependencies
SessionDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]
QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]




async def get_queue(name: str = "default") -> AsyncGenerator[JobQueue, None]:
    """Get job queue instance (requires sync redis connection usually, but RQ uses sync redis)"""
    # Note: RQ requires sync Redis client. 
    # For now we handle it simplistically. Ideally we separate sync/async redis.
    import redis
    sync_redis = redis.Redis.from_url(settings.redis_url)
    try:
        yield QueueFactory.get_queue(sync_redis, name)
    finally:
        sync_redis.close()


QueueDep = Annotated[JobQueue, Depends(get_queue)]
