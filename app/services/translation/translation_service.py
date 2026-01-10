"""
Translation Service

Orchestrates translation using available providers (OpenAI, DeepL, Mock).
"""

import asyncio
import time
from typing import Optional, Tuple

from app.core.config import settings
from app.core.logging import get_logger
from app.services.translation.llm_clients.openai_client import OpenAIClient
from app.services.translation.llm_clients.deepl_client import DeepLClient

logger = get_logger(__name__)


class TranslationService:
    def __init__(self):
        self.openai = OpenAIClient()
        self.deepl = DeepLClient()

    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        style_profile: str = "business",
        provider: Optional[str] = None
    ) -> Tuple[str, float]: # returns (text, latency_ms)
        """
        Translate text.
        Returns: (Translated text, Latency in ms)
        """
        start_time = time.time()
        provider = provider or settings.translation_provider
        
        try:
            # 1. Mock Provider
            if provider == "MOCK":
                # Simulate latency
                await asyncio.sleep(0.5)
                # Simple prefix translation
                translated = f"[{target_lang}] {text}"
                
            # 2. DeepL Provider
            elif provider == "DEEPL":
                translated = await self.deepl.translate(text, target_lang)
                
            # 3. OpenAI Provider
            else: # OPENAI
                style_msg = "Use polite business formal tone." if style_profile == "business" else ""
                translated = await self.openai.translate(
                    text, source_lang, target_lang, style_guide=style_msg
                )
                
        except Exception as e:
            logger.error(f"Translation failed with {provider}: {e}")
            # Fallback to Mock in dev, or raise in prod
            if settings.debug:
                translated = f"[FALLBACK] {text}"
            else:
                raise

        latency = (time.time() - start_time) * 1000
        return translated, latency
