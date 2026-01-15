"""
Qdrant infrastructure configuration

Qdrant vector database client management.
"""

from typing import AsyncGenerator, Optional

from qdrant_client import AsyncQdrantClient, models

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Global client
client: Optional[AsyncQdrantClient] = None


async def init_qdrant_client():
    """Initialize Qdrant client"""
    global client
    client = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
        timeout=10,
    )


async def close_qdrant_client():
    """Close Qdrant client"""
    global client
    if client:
        await client.close()


async def get_qdrant() -> AsyncGenerator[AsyncQdrantClient, None]:
    """Dependency for getting Qdrant client"""
    if client is None:
        await init_qdrant_client()
    
    yield client


async def ensure_collections_exist():
    """Ensure required collections exist in Qdrant"""
    if client is None:
        await init_qdrant_client()
        
    collections = await client.get_collections()
    collection_names = [c.name for c in collections.collections]
    
    # 1. Culture Cards Collection
    if settings.culture_cards_collection not in collection_names:
        logger.info(f"Creating collection: {settings.culture_cards_collection}")
        await client.create_collection(
            collection_name=settings.culture_cards_collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )

    # Note: Glossary collections are dynamic per organization (glossary_{org_id})
    # We will create them on demand when an organization is created or first glossary entry is added.
