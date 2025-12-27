"""
Script to seed culture cards into Qdrant.
Run with: python scripts/seed_culture_cards.py
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from app.core.config import settings
from app.infra.qdrant import init_qdrant_client, get_qdrant, ensure_collections_exist
from app.services.rag_service import RagService, CultureCard
from app.core.logging import setup_logging

# Sample Culture Cards Data (~50 items subset)
SAMPLE_CARDS = [
    # General Business / Greetings
    CultureCard(
        phrase="よろしくお願いします",
        meaning_summary="Multi-purpose phrase: 'Nice to meet you', 'Please help me', 'I rely on you'.",
        intents=["greeting", "request", "closing"],
        risk_level="low",
        suggested_questions=["Greeting or request?"]
    ),
    CultureCard(
        phrase="お世話になっております",
        meaning_summary="Standard business greeting: 'Thank you for your continued support'.",
        intents=["greeting", "opening"],
        risk_level="low",
    ),
    CultureCard(
        phrase="ご苦労様です / お疲れ様です",
        meaning_summary="'Otsukaresama' is safer. 'Gokurosama' implies searching down.",
        intents=["greeting", "acknowledgment"],
        risk_level="medium",
        suggested_questions=["Is speaker superior?"]
    ),
    
    # Decisions / Ambiguity
    CultureCard(
        phrase="検討します (Kentou shimasu)",
        meaning_summary="Literally 'I will consider it', but often means 'No' or 'Indefinite hold'.",
        intents=["refusal", "delay", "polite_no"],
        risk_level="high",
        suggested_questions=["Does this mean No?"]
    ),
    CultureCard(
        phrase="善処します (Zensho shimasu)",
        meaning_summary="'I will do my best', but usually means nothing will happen.",
        intents=["deflection", "polite_refusal"],
        risk_level="high"
    ),
    CultureCard(
        phrase="難しいですね (Muzukashii desu ne)",
        meaning_summary="Literally 'It's difficult', effectively means 'Impossible' or 'No'.",
        intents=["refusal"],
        risk_level="high"
    ),
    CultureCard(
        phrase="持ち帰って確認します",
        meaning_summary="'I will take it back and check'. Standard protocol to avoid immediate decision.",
        intents=["delay", "protocol"],
        risk_level="medium"
    ),

    # Effort / Attitude
    CultureCard(
        phrase="頑張ります (Ganbarimasu)",
        meaning_summary="'I will do my best'. Emphasizes effort/process over specific outcome guarantee.",
        intents=["commitment", "enthusiasm"],
        risk_level="medium"
    ),
    
    # Agreement / Listening
    CultureCard(
        phrase="はい (Hai)",
        meaning_summary="Can mean 'I'm listening', not necessarily 'I agree'.",
        intents=["acknowledgment", "agreement"],
        risk_level="high",
        suggested_questions=["Agreement or just listening?"]
    ),
    
    # Apology
    CultureCard(
        phrase="申し訳ございません",
        meaning_summary="Formal apology. Very strong.",
        intents=["apology"],
        risk_level="medium"
    ),
    
    # Add more to reach 50 theoretically... using 10 for demo script effectiveness
]


async def seed():
    setup_logging()
    print(f"Seeding culture cards to {settings.qdrant_url}...")
    
    # Init client
    await init_qdrant_client()
    
    # Ensure collection
    await ensure_collections_exist()
    
    # Get client wrapper
    qdrant = None
    async for q in get_qdrant():
        qdrant = q
        break
        
    rag_service = RagService(qdrant)
    
    # Upsert
    await rag_service.upsert_culture_cards(SAMPLE_CARDS)
    
    print("Done!")
    # Allow background tasks like client close cleanup if needed (though we just exit)

if __name__ == "__main__":
    asyncio.run(seed())
