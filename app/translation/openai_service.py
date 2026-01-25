from typing import List, Dict
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

openai_service = OpenAIService()
