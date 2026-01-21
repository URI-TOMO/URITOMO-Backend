from fastapi import APIRouter, HTTPException
from app.core.token import CurrentUserDep
from app.example.user.crud import UserCRUD
from app.example.user.schemas import UserResponse
from app.core.deps import SessionDep

router = APIRouter(prefix="/example/token", tags=["Example Token Auth"])

@router.get("/me", response_model=UserResponse)
async def get_my_info(
    user_id: CurrentUserDep,
    db: SessionDep
):
    """
    Get current user information based on the JWT token.
    The user_id is extracted from the 'sub' claim of the token.
    
    To test this:
    1. Create a user via /example/users
    2. Obtain a token (you can use create_access_token in a temporary script)
    3. Call this endpoint with 'Authorization: Bearer <token>'
    """
    user = await UserCRUD.get_by_id(db, user_id)
    if not user:
        raise HTTPException(
            status_code=404, 
            detail=f"User with ID {user_id} not found in database. Make sure you created the user first."
        )
    return user
