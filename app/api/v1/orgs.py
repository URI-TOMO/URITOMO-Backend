"""
Organization Endpoints
"""

from typing import List, Any

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, CurrentUserDep
from app.schemas.org import OrgCreate, OrgResponse, OrgUpdate
from app.services.org_service import OrgService

router = APIRouter()


@router.post("/", response_model=OrgResponse, status_code=status.HTTP_201_CREATED)
async def create_org(
    org_in: OrgCreate,
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Create new organization
    """
    org_service = OrgService(db)
    org = await org_service.create_org(org_in, int(current_user_id))
    return org


@router.get("/", response_model=List[OrgResponse])
async def read_orgs(
    current_user_id: CurrentUserDep,
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Retrieve organizations user belongs to
    """
    org_service = OrgService(db)
    orgs = await org_service.get_user_orgs(int(current_user_id))
    return orgs


@router.get("/{org_id}", response_model=OrgResponse)
async def read_org(
    org_id: int,
    current_user_id: CurrentUserDep,  # In real app, verify membership
    db: AsyncSession = Depends(get_db),
) -> Any:
    """
    Get specific organization
    """
    org_service = OrgService(db)
    org = await org_service.get_org(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    # TODO: Verify user is member of org
    return org
