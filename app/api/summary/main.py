from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Dict, List, Optional
import json
import os
from datetime import datetime

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import Room, RoomMember, RoomSummary
from app.models.stt import RoomSttResult
from uuid import uuid4
from app.translation.openai_service import openai_service
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/summary", tags=["summary"])

# ============ Schemas ============

class SummarizationContent(BaseModel):
    main_point: str
    task: str
    decided: str

class SummarizationData(BaseModel):
    summarization: SummarizationContent
    meeting_date: str
    past_time: str
    meeting_member: int
    message_count: int

class SummarizationResponse(BaseModel):
    summary: SummarizationData

# ============ Endpoints ============

@router.post("/summarization/{room_id}", response_model=SummarizationResponse)
async def get_summarization(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の要約（メインポイント、タスク、決定事項）を取得します。
    RoomSttResultテーブルから音声文字起こし（STT）データのみを使用して、
    OpenAI を使用して要約を生成します。
    """
    try:
        logger.info(f"Generating summary for room {room_id}")
        
        # --- Mock Data Logic Start ---
        mock_file_path = os.path.join(os.path.dirname(__file__), "mock_stt_data.json")
        logger.info(f"Checking for mock file at: {os.path.abspath(mock_file_path)}")
        use_mock = False
        room = None
        members = []
        stt_messages = []
        
        if os.path.exists(mock_file_path):
            logger.info("Mock file found")
            try:
                with open(mock_file_path, "r", encoding="utf-8") as f:
                    mock_data = json.load(f)
                
                # Check for exact match or trimmed/case-insensitive match for convenience
                target_id = room_id.strip()
                match_id = next((k for k in mock_data.keys() if k.lower() == target_id.lower()), None)
                
                if match_id:
                    logger.info(f"Using mock data for room {target_id} (matched key: {match_id})")
                    data = mock_data[match_id]
                    
                    # Mock Room object
                    class MockRoom:
                        def __init__(self, d):
                            self.id = d.get("id", match_id)
                            self.created_at = datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now()
                            self.ended_at = datetime.fromisoformat(d["ended_at"]) if d.get("ended_at") else None
                    
                    room = MockRoom(data.get("room", {}))
                    
                    # Mock Member objects
                    class MockMember:
                        def __init__(self, d):
                            self.id = d.get("id", "unknown")
                            self.display_name = d.get("display_name", "Unknown Member")
                    
                    members = [MockMember(m) for m in data.get("members", [])]
                    
                    # Mock STT messages
                    class MockStt:
                        def __init__(self, d):
                            self.member_id = d.get("member_id", "unknown")
                            self.stt_text = d.get("stt_text", "")
                            self.seq = d.get("seq", 0)
                            self.created_at = datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.now()
                    
                    stt_results = data.get("stt_results", [])
                    stt_messages = [MockStt(msg) for msg in stt_results]
                    use_mock = True
                else:
                    logger.warning(f"Room ID '{room_id}' not found in mock data keys: {list(mock_data.keys())}")
            except Exception as e:
                logger.error(f"Error loading/parsing mock data: {e}", exc_info=True)
        else:
            logger.warning(f"Mock file NOT found at: {os.path.abspath(mock_file_path)}")

        if not use_mock:
            # 1. Fetch room info
            room_stmt = select(Room).where(Room.id == room_id)
            room_result = await db.execute(room_stmt)
            room = room_result.scalar_one_or_none()
            
            if not room:
                logger.warning(f"Room {room_id} not found")
                raise HTTPException(status_code=404, detail="Meeting room not found")
            
            # 2. Fetch member info
            member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
            member_result = await db.execute(member_stmt)
            members = member_result.scalars().all()
            
            # 3. STT（音声文字起こし）データを取得
            stt_stmt = select(RoomSttResult).where(
                RoomSttResult.room_id == room_id
            ).order_by(RoomSttResult.seq.asc())
            
            stt_result = await db.execute(stt_stmt)
            stt_messages = stt_result.scalars().all()
        # --- Mock Data Logic End ---

        member_count = len(members)
        participant_names = [m.display_name for m in members]
        member_map = {m.id: m.display_name for m in members}
        
        logger.info(f"Room {room_id} has {member_count} members")
        logger.debug(f"Fetched {len(stt_messages)} STT messages")
        
        # 4. STTメッセージが存在するかチェック
        total_messages = len(stt_messages)
        if total_messages == 0:
            logger.warning(f"No STT messages found for room {room_id}")
            return SummarizationResponse(
                summary=SummarizationData(
                    summarization=SummarizationContent(
                        main_point="No voice recordings in this meeting",
                        task="N/A",
                        decided="N/A"
                    ),
                    meeting_date=room.created_at.strftime("%Y-%m-%d"),
                    past_time="Unknown",
                    meeting_member=member_count,
                    message_count=0
                )
            )
        
        # 5. STTデータを時系列（シーケンス番号順）でフォーマット
        text_parts = ["=== VOICE TRANSCRIPT (Speech-to-Text) ==="]
        
        for msg in stt_messages:
            sender_name = member_map.get(msg.member_id, "Unknown")
            time_str = msg.created_at.strftime('%H:%M:%S')
            seq = msg.seq
            text_parts.append(f"[{time_str}] [{seq}] {sender_name}: {msg.stt_text}")
        
        conversation_text = "\n".join(text_parts)
        
        logger.debug(f"STT transcript length: {len(conversation_text)} characters")
        
        # 6. 音声文字起こし用のプロンプトを構築
        prompt = f"""
You are a professional meeting summarizer specializing in analyzing voice transcriptions from meetings.

The following is a transcript of a meeting generated by speech-to-text (STT) technology. 
The transcript is presented in chronological order with timestamps.

Note: As this is STT output, there may be:
- Minor transcription errors
- Informal spoken language
- Incomplete sentences or fillers (um, uh, etc.)

Your task is to analyze the conversation flow and extract meaningful information.

Voice Transcript:
{conversation_text}

Participants: {', '.join(participant_names)}

Please provide a comprehensive summary with:

1. **Main Points**: What were the key topics discussed during the meeting? Summarize the main ideas and important discussions in a clear, organized manner.

2. **Tasks/Action Items**: What specific actions or tasks were mentioned? Who is responsible for each task? Include any deadlines or timeframes mentioned.

3. **Decisions**: What key decisions were made? What agreements or conclusions were reached during the meeting?

Respond ONLY with valid JSON in this exact format:
{{
    "main_point": "Clear and comprehensive summary of the main discussion topics",
    "task": "List of action items with responsible persons and deadlines",
    "decided": "Key decisions and agreements made during the meeting"
}}
"""
        
        # 7. Call OpenAI
        logger.debug(f"Prompt sent to OpenAI:\n{prompt}")
        logger.debug("Sending request to OpenAI API")
        
        response = openai_service.client.chat.completions.create(
            model=openai_service.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional meeting summarizer specializing in voice transcriptions. You provide comprehensive, concise summaries in valid JSON format, understanding the nuances of spoken language captured via speech-to-text. IMPORTANT: Always provide the summarization content in Japanese."
                },
                {"role": "user", "content": f"{prompt}\n\nPlease output the 'main_point', 'task', and 'decided' fields in Japanese."}
            ],
            response_format={"type": "json_object"},
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # 9. Calculate duration
        if room.ended_at:
            duration = room.ended_at - room.created_at
            past_time = int(duration.total_seconds() / 60)
        else:
            past_time = 0
        past_time_str = f"{past_time} min" if past_time > 0 else "Unknown"
        
        meeting_date_str = room.created_at.strftime("%Y-%m-%d")
        
        # 10. Save to Database
        try:
            summary_id = f"sum_{uuid4().hex}"
            new_summary = RoomSummary(
                id=summary_id,
                room_id=room_id,
                main_point=result.get("main_point", ""),
                task=result.get("task", ""),
                decided=result.get("decided", ""),
                meeting_date=room.created_at,
                past_time=past_time_str,
                member_count=member_count,
                message_count=total_messages,
                created_at=datetime.utcnow()
            )
            
            # Note: In mock mode, if room_id doesn't exist in actual DB, 
            # this will fail due to foreign key constraint.
            if not use_mock:
                db.add(new_summary)
                await db.commit()
                logger.info(f"Summary saved to database with ID: {summary_id}")
            else:
                logger.info(f"Mock mode active, skipping DB save for room {room_id}")
                
        except Exception as db_err:
            logger.error(f"Failed to save summary to database: {db_err}")
            # We don't raise here to ensure user still gets the response even if save fails
            # But in a real app, you might want to handle this differently
        
        logger.info(f"Summary generated successfully for room {room_id} with {total_messages} STT messages")
        
        return SummarizationResponse(
            summary=SummarizationData(
                summarization=SummarizationContent(
                    main_point=result.get("main_point", ""),
                    task=result.get("task", ""),
                    decided=result.get("decided", "")
                ),
                meeting_date=meeting_date_str,
                past_time=past_time_str,
                meeting_member=member_count,
                message_count=total_messages
            )
        )
    
    except HTTPException:
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing OpenAI response as JSON: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse summary response")
    except Exception as e:
        logger.error(f"Error getting summarization for room {room_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


