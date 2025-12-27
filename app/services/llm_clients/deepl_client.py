"""
DeepL Client wrapper

Handles interactions with DeepL API for translation.
"""

import httpx
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DeepLClient:
    def __init__(self):
        self.api_key = settings.deepl_api_key
        self.base_url = settings.deepl_api_url

    async def translate(self, text: str, target_lang: str) -> str:
        """
        Translate text using DeepL API.
        Note: DeepL uses specific language codes (e.g., 'EN-US', 'JA', 'KO').
        """
        if not self.api_key:
            raise ValueError("DeepL API key not provided")

        # Map common lang codes to DeepL format if needed
        # Simple mapping for MVP
        lang_map = {
            "en": "EN-US",
            "ja": "JA",
            "ko": "KO",
        }
        target = lang_map.get(target_lang.lower(), target_lang.upper())

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    data={
                        "auth_key": self.api_key,
                        "text": text,
                        "target_lang": target,
                    },
                    timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                return result["translations"][0]["text"]
        except Exception as e:
            logger.error(f"DeepL API Error: {e}")
            raise
