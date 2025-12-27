"""
RAG Service

Handles interaction with Qdrant for retrieving cultural context and glossary terms.
"""

from typing import List, Dict, Any, Optional

from qdrant_client import AsyncQdrantClient, models
from app.core.config import settings
from app.core.logging import get_logger
from app.schemas.rag import CultureCard, RagSearchResult

logger = get_logger(__name__)


class RagService:
    def __init__(self, qdrant: AsyncQdrantClient):
        self.qdrant = qdrant

    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text.
        For MVP/Mock, returns random vector or deterministic hash-based vector.
        In production, call OpenAI or local model.
        """
        if settings.use_mock_embedding:
            # Deterministic mock embedding based on hash
            import numpy as np
            import hashlib
            
            # Simple hash to seed
            hash_obj = hashlib.md5(text.encode())
            seed = int(hash_obj.hexdigest(), 16) % (2**32)
            rng = np.random.default_rng(seed)
            
            # Create random vector
            vector = rng.random(settings.embedding_dimension).astype(np.float32)
            # Normalize
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector.tolist()
        else:
            # Call OpenAI Embedding API
            # Placeholder for actual implementation using openai_client
            # For now assuming mock if not implemented
            from app.services.llm_clients.openai_client import OpenAIClient
            client = OpenAIClient()
            return await client.get_embedding(text)

    async def search_culture_cards(
        self, query: str, top_k: int = 3, threshold: float = 0.7
    ) -> List[RagSearchResult]:
        """Search culture cards collection"""
        try:
            vector = await self._get_embedding(query)
            
            results = await self.qdrant.search(
                collection_name=settings.culture_cards_collection,
                query_vector=vector,
                limit=top_k,
                score_threshold=threshold,
            )
            
            return [
                RagSearchResult(
                    id=str(res.id),
                    score=res.score,
                    payload=res.payload,
                )
                for res in results
            ]
        except Exception as e:
            logger.error(f"RAG search failed: {e}")
            return []

    async def upsert_culture_cards(self, cards: List[CultureCard]):
        """Upsert culture cards"""
        points = []
        for i, card in enumerate(cards):
            # Generate ID if not present (using hash of phrase)
            import uuid
            import hashlib
            
            if not card.card_id:
                card.card_id = str(uuid.UUID(hex=hashlib.md5(card.phrase.encode()).hexdigest()))
            
            # Emded phrase + intent + meaning
            text_to_embed = f"{card.phrase} {card.meaning_summary} {' '.join(card.intents)}"
            vector = await self._get_embedding(text_to_embed)
            
            points.append(
                models.PointStruct(
                    id=card.card_id,
                    vector=vector,
                    payload=card.model_dump(),
                )
            )
            
        await self.qdrant.upsert(
            collection_name=settings.culture_cards_collection,
            points=points,
        )
        logger.info(f"Upserted {len(points)} culture cards")

    async def search_glossary(
        self, org_id: int, query: str, top_k: int = 5
    ) -> List[RagSearchResult]:
        """Search glossary for an organization"""
        collection_name = f"{settings.glossary_collection_prefix}{org_id}"
        
        # Check if collection exists first? 
        # For performance, might rely on error handling or cache existence
        exists = await self.qdrant.collection_exists(collection_name)
        if not exists:
            return []

        try:
            vector = await self._get_embedding(query)
            
            results = await self.qdrant.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=top_k,
            )
            
            return [
                RagSearchResult(
                    id=str(res.id),
                    score=res.score,
                    payload=res.payload,
                )
                for res in results
            ]
        except Exception as e:
            logger.error(f"Glossary search failed: {e}")
            return []
