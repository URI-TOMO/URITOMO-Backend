"""
Health check endpoints
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from redis.asyncio import Redis
from qdrant_client import AsyncQdrantClient

from app.core.deps import get_db, get_redis, get_qdrant

router = APIRouter()


@router.get("/health")
async def health_check(
    db: AsyncSession = Depends(get_db),
    redis: Redis = Depends(get_redis),
    qdrant: AsyncQdrantClient = Depends(get_qdrant),
):
    """
    Check health of all dependent services
    """
    status = {
        "api": "ok",
        "db": "unknown",
        "redis": "unknown",
        "qdrant": "unknown",
    }
    
    # Check DB
    try:
        await db.execute(text("SELECT 1"))
        status["db"] = "ok"
    except Exception as e:
        status["db"] = f"error: {str(e)}"

    # Check Redis
    try:
        await redis.ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"

    # Check Qdrant
    try:
        # Simple call to check connection
        # get_collections is lightweight
        await qdrant.get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {str(e)}"

    return status
