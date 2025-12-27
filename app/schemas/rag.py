"""
RAG Schemas

Models for RAG operations (search, formatting).
"""

from typing import List, Optional

from pydantic import BaseModel


class CultureCard(BaseModel):
    card_id: Optional[str] = None
    phrase: str
    meaning_summary: str
    intents: List[str]
    risk_level: str  # low, medium, high
    suggested_questions: List[str] = []


class ExplainDecision(BaseModel):
    should_explain: bool
    reason: str
    matched_cards: List[CultureCard] = []


class RagSearchResult(BaseModel):
    id: str  # Point ID
    score: float
    payload: dict
