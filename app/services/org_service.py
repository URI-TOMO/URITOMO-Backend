"""
Organization Service
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.org import Organization
from app.models.user import UserOrg
from app.schemas.org import OrgCreate, OrgUpdate


class OrgService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_org(self, org_id: int) -> Optional[Organization]:
        result = await self.session.execute(select(Organization).where(Organization.id == org_id))
        return result.scalars().first()

    async def create_org(self, org_in: OrgCreate, owner_id: int) -> Organization:
        # Create org
        org = Organization(
            name=org_in.name,
            slug=org_in.slug,
            default_source_lang=org_in.default_source_lang,
            default_target_lang=org_in.default_target_lang,
        )
        self.session.add(org)
        await self.session.flush() # flush to get ID
        
        # Add creator as owner
        user_org = UserOrg(user_id=owner_id, org_id=org.id, role="owner")
        self.session.add(user_org)
        
        await self.session.commit()
        await self.session.refresh(org)
        return org

    async def get_user_orgs(self, user_id: int) -> List[Organization]:
        stmt = (
            select(Organization)
            .join(UserOrg)
            .where(UserOrg.user_id == user_id)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())
