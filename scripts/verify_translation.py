import sys
import os
import asyncio

# Add project root to path
sys.path.append(os.getcwd())

from app.core.config import settings
from app.translation.deepl_service import deepl_service

def test_deepl_translation():
    print("Testing REAL DeepL translation...")
    # Force DEEPL provider
    settings.translation_provider = "DEEPL"
    
    # Re-initialize service to pick up new settings
    from app.translation.deepl_service import DeepLService
    # We create a new instance to bypass the global instance which was initialized with previous settings
    service = DeepLService()
    
    text = "안녕하세요"
    # Note: Ensure DEEPL_API_KEY is set in .env or environment variables
    if not settings.deepl_api_key:
        print("ERROR: DEEPL_API_KEY is missing!")
        return

    translated = service.translate_text(text, "Korean", "Japanese")
    print(f"Original: {text}")
    print(f"Translated: {translated}")
    
    # Real DeepL translation for "안녕하세요" should be "こんにちは" or simliar, NOT containing [JA] tag
    if "こんにちは" in translated:
        print("SUCCESS: DeepL translation working")
    elif "[JA]" in translated:
        print("FAILURE: Still using Mock translation")
    else:
        print(f"WARNING: Unexpected translation result: {translated}")

if __name__ == "__main__":
    test_deepl_translation()
