"""
Dependency Injection

FastAPI dependencies for routes.
"""

from typing import Annotated, AsyncGenerator

from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from qdrant_client import AsyncQdrantClient

from app.core.config import settings
from app.core.errors import AuthenticationError
from app.core.security import verify_token
from app.infra.db import get_db
from app.infra.redis import get_redis
from app.infra.qdrant import get_qdrant
from app.infra.queue import JobQueue, QueueFactory

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.api_prefix}/auth/login")

# Type aliases for common dependencies
SessionDep = Annotated[AsyncSession, Depends(get_db)]
RedisDep = Annotated[Redis, Depends(get_redis)]
QdrantDep = Annotated[AsyncQdrantClient, Depends(get_qdrant)]


async def get_current_user_id(token: Annotated[str, Depends(oauth2_scheme)]) -> str:
    """
    Validate token and return current user ID.
    Does not fetch full user object to save DB call.
    """
    user_id = verify_token(token)
    if not user_id:
        raise AuthenticationError("Could not validate credentials")
    return user_id


CurrentUserDep = Annotated[str, Depends(get_current_user_id)]


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
