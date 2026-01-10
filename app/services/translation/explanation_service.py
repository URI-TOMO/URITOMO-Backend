"""
Explanation Service

Decides if an explanation is needed and generates it using RAG.
"""

import json
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.services.translation.rag_service import RagService, CultureCard
from app.services.translation.llm_clients.openai_client import OpenAIClient
from app.schemas.rag import ExplainDecision
from app.models.translation import TranslationSegment

logger = get_logger(__name__)


class ExplanationService:
    def __init__(self, rag_service: RagService):
        self.rag = rag_service
        self.openai = OpenAIClient()

    async def check_explanation_needed(self, text: str) -> ExplainDecision:
        """
        Determine if explanation is needed using Rule-based + RAG checks.
        """
        # 1. RAG Search for key terms (Rule-based surrogate)
        # Search for exact or near-exact matches in culture cards
        results = await self.rag.search_culture_cards(text, top_k=3, threshold=0.85)
        
        matched_cards: List[CultureCard] = []
        for res in results:
            card = CultureCard(**res.payload)
            matched_cards.append(card)
            
        if matched_cards:
            # High confidence match -> Explain
            phrases = [c.phrase for c in matched_cards]
            return ExplainDecision(
                should_explain=True,
                reason=f"Matched cultural/risk terms: {', '.join(phrases)}",
                matched_cards=matched_cards
            )

        # 2. Heuristic check (Length, specific patterns)
        # TODO: Add regex patterns here if needed (e.g. looking for "本音Tatemae" patterns)
        
        return ExplainDecision(should_explain=False, reason="No triggers found")

    async def generate_explanation(
        self, 
        original_text: str, 
        translated_text: str, 
        cards: List[CultureCard]
    ) -> str:
        """
        Generate a 2-line explanation using LLM + RAG context
        """
        if settings.use_mock_translation:
            # Mock explanation
            phrases = [c.phrase for c in cards]
            return f"Context: {', '.join(phrases)}. Usually implies politeness/ambiguity."

        # Prepare context from cards
        context_str = "\n".join([
            f"- Term: {c.phrase}\n  Meaning: {c.meaning_summary}\n  Intents: {', '.join(c.intents)}"
            for c in cards
        ])
        
        prompt = f"""
You are a cultural interpreter.
Using the provided context, explain the nuance of the original text compared to the literal translation.
Keep it strictly under 2 sentences. Focus on potential misunderstanding or hidden intent.

Original: {original_text}
Translation: {translated_text}

Context:
{context_str}

Output only the explanation.
"""
        messages = [{"role": "user", "content": prompt}]
        
        explanation = await self.openai.chat_completion(messages, temperature=0.3)
        return explanation
