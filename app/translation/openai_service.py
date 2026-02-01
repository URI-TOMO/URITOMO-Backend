from typing import List, Dict
import json
import openai
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class OpenAIService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self.model = settings.openai_model

    async def get_description_for_terms(self, text_data: str) -> List[Dict[str, str]]:
        """
        Send aggregated STT text to OpenAI and get explanations for difficult/advanced terms.
        """
        if not self.client:
            logger.warning("OpenAI API key is not configured. Returning mock description.")
            return [
                {"term": "Architecture", "explanation": "The complex or carefully designed structure of something."},
                {"term": "FastAPI", "explanation": "A modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints."}
            ]

        prompt = f"""
        Below is a transcript from a meeting. 
        Please identify difficult words, technical terms, or advanced vocabulary used in this transcript.
        For each identified term, provide a clear and detailed explanation in both Korean and Japanese.
        Return the result as a JSON list of objects, each having "term", "explanation_ko", and "explanation_ja" keys.

        Transcript:
        \"\"\"{text_data}\"\"\"
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that explains difficult terminology from meeting transcripts."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            # The model might return a wrap object like {"terms": [...]}
            if "terms" in result:
                return result["terms"]
            return result if isinstance(result, list) else [result]
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return []

    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translate text using OpenAI.
        Returns translated text only.
        """
        if not text:
            return ""

        if not self.client:
            logger.warning("OpenAI API key is not configured. Returning mock translation.")
            return self._mock_translate(text, target_lang)

        prompt = (
            "Translate the following text from {source_lang} to {target_lang}. "
            "Return a JSON object with a single key \"translation\" and no other text.\n\n"
            "Text:\n"
            "\"\"\"{text}\"\"\""
        ).format(source_lang=source_lang, target_lang=target_lang, text=text)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise translation engine."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )

            result = json.loads(response.choices[0].message.content)
            if isinstance(result, dict) and "translation" in result:
                return str(result["translation"])
            if isinstance(result, str):
                return result
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI translation failed: {e}")
            return self._mock_translate(text, target_lang)

    def _mock_translate(self, text: str, target_lang: str) -> str:
        prefix = f"[{target_lang}]" if target_lang else "[TRANS]"
        return f"{prefix} {text}"

openai_service = OpenAIService()
