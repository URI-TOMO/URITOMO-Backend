from fastapi import APIRouter, Depends, HTTPException
from redis.asyncio import Redis
from app.infra.redis import get_redis

router = APIRouter()

@router.post("/set")
async def set_redis_value(key: str, value: str, redis: Redis = Depends(get_redis)):
    """Set a value in Redis."""
    await redis.set(key, value)
    return {"status": "ok", "key": key, "value": value}

@router.get("/get/{key}")
async def get_redis_value(key: str, redis: Redis = Depends(get_redis)):
    """Get a value from Redis."""
    value = await redis.get(key)
    if value is None:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"key": key, "value": value}
