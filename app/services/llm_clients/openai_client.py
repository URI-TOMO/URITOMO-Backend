"""
OpenAI Client wrapper

Handles interactions with OpenAI API for translation, embedding, and summary.
"""

from typing import List, Optional, Any, AsyncGenerator

from openai import AsyncOpenAI
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIClient:
    def __init__(self):
        self.api_key = settings.openai_api_key
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get text embedding"""
        if not self.client:
            # Fallback to mock if initialized without key (should be handled by caller usually)
            raise ValueError("OpenAI API key not provided")
            
        try:
            response = await self.client.embeddings.create(
                input=text,
                model=settings.openai_embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI Embedding Error: {e}")
            raise

    async def chat_completion(
        self, 
        messages: List[dict], 
        model: Optional[str] = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        """Standard chat completion"""
        if not self.client:
            raise ValueError("OpenAI API key not provided")

        try:
            kwargs = {
                "model": model or settings.openai_model,
                "messages": messages,
                "temperature": temperature,
            }
            
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = await self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI Chat Error: {e}")
            raise

    async def chat_completion_stream(
        self, 
        messages: List[dict],
        model: Optional[str] = None,
        temperature: float = 0.3
    ) -> AsyncGenerator[str, None]:
        """Streaming chat completion"""
        if not self.client:
            raise ValueError("OpenAI API key not provided")

        try:
            stream = await self.client.chat.completions.create(
                model=model or settings.openai_model,
                messages=messages,
                temperature=temperature,
                stream=True,
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI Stream Error: {e}")
            raise

    async def translate(
        self, 
        text: str, 
        source_lang: str, 
        target_lang: str, 
        style_guide: str = ""
    ) -> str:
        """Specialized translation method"""
        
        system_prompt = f"""
You are a professional interpreter.
Translate the following text from {source_lang} to {target_lang}.
{style_guide}
Only output the translated text.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text}
        ]
        
        return await self.chat_completion(messages)
