from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.infra.db import get_db

router = APIRouter()

@router.get("/health")
async def check_mysql_health(db: AsyncSession = Depends(get_db)):
    """Check MySQL connection by executing SELECT 1."""
    try:
        result = await db.execute(text("SELECT 1"))
        return {"status": "ok", "result": result.scalar()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
