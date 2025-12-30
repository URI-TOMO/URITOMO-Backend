from fastapi import APIRouter, Depends, HTTPException
from qdrant_client import AsyncQdrantClient, models
from app.infra.qdrant import get_qdrant
import uuid
import random

router = APIRouter()

@router.post("/create_collection")
async def create_qdrant_collection(cls_name: str, qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    """Create a test collection in Qdrant."""
    try:
        await qdrant.create_collection(
            collection_name=cls_name,
            vectors_config=models.VectorParams(
                size=4,  # Small dimension for testing
                distance=models.Distance.COSINE,
            ),
        )
        return {"status": "ok", "collection": cls_name}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/upsert")
async def upsert_vector(cls_name: str, qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    """Upsert a random vector."""
    try:
        point_id = str(uuid.uuid4())
        vector = [random.random() for _ in range(4)]
        
        await qdrant.upsert(
            collection_name=cls_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"type": "test"}
                )
            ]
        )
        return {"status": "ok", "id": point_id, "vector": vector}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/search")
async def search_vector(cls_name: str, qdrant: AsyncQdrantClient = Depends(get_qdrant)):
    """Search for a random vector."""
    try:
        query_vector = [random.random() for _ in range(4)]
        
        results = await qdrant.search(
            collection_name=cls_name,
            query_vector=query_vector,
            limit=3
        )
        return {"status": "ok", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
