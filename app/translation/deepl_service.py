"""
DeepL Translation Service
"""

import deepl
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class DeepLService:
    def __init__(self):
        self.client = None
        self.enabled = False
        
        if settings.translation_provider == "DEEPL" and settings.deepl_api_key:
            try:
                self.client = deepl.Translator(settings.deepl_api_key)
                self.enabled = True
                logger.info("DeepL translation service initialized")
            except Exception as e:
                logger.error(f"Failed to initialize DeepL service: {e}")
        elif settings.translation_provider == "MOCK":
            logger.info("DeepL service disabled (Provider: MOCK)")
        elif not settings.deepl_api_key:
            logger.warning("DeepL API key is missing. DeepL service disabled.")

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using DeepL.
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language (e.g., "Korean", "Japanese")
            target_lang (str): Target language
            
        Returns:
            str: Translated text
        """
        if not text:
            return ""
            
        # Map languages to DeepL codes
        # DeepL uses KO for Korean, JA for Japanese
        source_code = self._map_language_code(source_lang)
        target_code = self._map_language_code(target_lang)
        
        if settings.translation_provider == "MOCK" or not self.enabled:
            return self._mock_translate(text, source_code, target_code)
            
        try:
            # DeepL Python library automatically handles source_lang=None (auto-detect)
            # but we'll be explicit if possible
            result = self.client.translate_text(
                text,
                source_lang=source_code,
                target_lang=target_code
            )
            
            # result can be a list if multiple texts provided, but here we send one
            if isinstance(result, list):
                return result[0].text
            return result.text
            
        except deepl.DeepLException as e:
            logger.error(f"DeepL translation error: {e}")
            # Fallback to returning original or mock on error? 
            # For now, let's return a specific error string or the original to avoid crashing
            return f"[Error: {text}]"
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            return text

    def _map_language_code(self, lang: str) -> str:
        """Map language names to DeepL codes"""
        lang_lower = lang.lower()
        if "korea" in lang_lower:
            return "KO"
        if "japan" in lang_lower:
            return "JA"
        return None

    def _mock_translate(self, text: str, source_code: str, target_code: str) -> str:
        """Mock translation for development"""
        prefix = f"[{target_code}]" if target_code else "[TRANS]"
        return f"{prefix} {text}"


# Global instance
deepl_service = DeepLService()
