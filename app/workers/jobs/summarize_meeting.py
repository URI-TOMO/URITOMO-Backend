"""
Meeting Summary Job
"""

import asyncio
from app.core.logging import get_logger
from app.core.config import settings
from app.services.llm_clients.openai_client import OpenAIClient
# Note: In worker context, we need to create new DB sessions if accessing DB
# For simple MVP, we might mock DB access or use sync/async adapters

logger = get_logger(__name__)

async def _generate_summary_logic(meeting_id: int):
    """
    Async logic for summary generation
    """
    logger.info(f"Starting summary generation for meeting {meeting_id}")
    
    # 1. Fetch meeting segments (Mocking DB access for now)
    # In real impl, use AsyncSession with run_sync or separate sync DB access
    
    # Mock segments
    transcript_text = "Speaker A: This is a test meeting. We decided to launch on Friday.\nSpeaker B: Agreed. I will handle marketing."
    
    # 2. Call LLM
    openai = OpenAIClient()
    
    if settings.summary_provider == "MOCK":
        await asyncio.sleep(2)
        summary_text = f"[MOCK SUMMARY] Meeting {meeting_id} was productive."
        decisions = ["Launch on Friday"]
        action_items = ["Marketing by Speaker B"]
    else:
        # Real LLM call
        summary_text = await openai.chat_completion(
            messages=[{"role": "user", "content": f"Summarize this:\n{transcript_text}"}]
        )
        decisions = [] # Parse from result
        action_items = []
        
    logger.info(f"Summary generated: {summary_text[:50]}...")
    
    # 3. Save to DB (Skipped in MVP Worker - would need Sync/Async session handling)
    # Ideally, we save `Summary` model here.
    
    return {
        "meeting_id": meeting_id,
        "summary": summary_text,
        "decisions": decisions,
        "action_items": action_items
    }


def summarize_meeting(meeting_id: int):
    """
    RQ Job entry point (Sync wrapper)
    """
    # RQ runs in sync context, but we use async libraries
    return asyncio.run(_generate_summary_logic(meeting_id))
