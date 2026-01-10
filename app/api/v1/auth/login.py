"""
Simple Login Endpoint
"""

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/login")
async def login(name: str, password: str):
    """
    Simple login endpoint - returns access token
    
    Args:
        name: Username
        password: User password
    
    Returns:
        access_token: JWT token for authentication
    """
    # TODO: Add actual authentication logic here
    # For now, simple validation
    if not name or not password:
        raise HTTPException(
            status_code=400,
            detail="Name and password are required"
        )
    
    # TODO: Verify credentials against database
    # TODO: Generate real JWT token
    
    # Temporary mock response
    access_token = f"mock_token_for_{name}"
    
    return {
        "access_token": access_token
    }
