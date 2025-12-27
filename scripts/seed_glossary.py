"""
Script to seed basic glossary.
"""

import asyncio
import os
import sys

sys.path.append(os.getcwd())

from app.infra.qdrant import init_qdrant_client, get_qdrant
from app.core.logging import setup_logging
from app.core.config import settings

async def seed():
    setup_logging()
    print("Seeding glossary placeholder...")
    # Real implementation would read from a file or DB and use RagService.upsert_glossary
    # For now, just a placeholder to show structure
    print("No default glossary to seed in MVP. Use API to add terms.")

if __name__ == "__main__":
    asyncio.run(seed())
