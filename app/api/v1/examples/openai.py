from fastapi import APIRouter, HTTPException
import httpx
from pydantic import BaseModel
from app.core.config import settings

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

@router.post("/chat")
async def chat_openai(request: ChatRequest):
    """Simple chat completion using OpenAI API key from env."""
    if not settings.OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": settings.openai_model,
        "messages": [{"role": "user", "content": request.message}]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=f"OpenAI API Error: {e.response.text}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
