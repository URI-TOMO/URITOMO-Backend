from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.infra.db import get_db
from app.models.user import User

router = APIRouter()

@router.post("/login/setup-mock", tags=["debug"])
async def setup_login_mock_data(
    email: str = Query("test@example.com", description="Email for the test user"),
    user_id: str = Query("test-user-id", description="Fixed user ID for testing"),
    db: AsyncSession = Depends(get_db)
):
    """
    Sets up a specific test user for login debugging.
    """
    stmt = select(User).where(User.email == email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        id_stmt = select(User).where(User.id == user_id)
        id_result = await db.execute(id_stmt)
        if id_result.scalar_one_or_none():
            user_id = f"{user_id}_new"

        user = User(
            id=user_id,
            display_name=f"TestUser_{user_id[:4]}",
            email=email,
            locale="ja",
            status="active"
        )
        db.add(user)
    else:
        user.status = "active"

    try:
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to setup login mock data: {str(e)}"
        )

    return {
        "status": "success",
        "message": f"Setup mock user for login: {email}",
        "user_id": user.id,
        "email": email
    }
