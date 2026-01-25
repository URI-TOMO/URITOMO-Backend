from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from app.infra.db import get_db
from app.models.room import Room, RoomMember
from .logic.meeting_data import fetch_meeting_transcript, format_transcript_for_ai
from .logic.ai_summary import summarize_meeting, save_summary_to_db, get_summary_from_db
import re
import os
import json

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_FILE_PATH = os.path.join(BASE_DIR, "logic", "large_meeting_data.json")

# ============ Schemas ============

class DocumentInfo(BaseModel):
    meeting_date: str
    past_time: str
    meeting_member: str
    meeting_name: str

class SummarizationContent(BaseModel):
    main_point: str
    task: str
    decided: str

class SummarizationData(BaseModel):
    summarization: SummarizationContent
    meeting_date: str
    past_time: str
    meeting_member: int

class TranslationLogItem(BaseModel):
    id: Optional[str] = None
    timestamp: str
    sender_name: str
    text: str

class SummarizationResponse(BaseModel):
    documents: DocumentInfo
    summary: SummarizationData
    translation_log: List[TranslationLogItem]

class MeetingDataInput(BaseModel):
    title: str
    content: str

# ============ Endpoints ============

@router.post("/summarization/{room_id}", response_model=SummarizationResponse, tags=["summary"])
async def get_summarization(
    room_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    会議の要約、会議情報、および全発言ログを取得します（議事録作成）。
    キャッシング機能付き：既に要約がDBに保存されている場合は再利用します。
    """
    # 1. 会議室の存在確認
    room_stmt = select(Room).where(Room.id == room_id)
    room_result = await db.execute(room_stmt)
    room = room_result.scalar_one_or_none()
    if not room:
        raise HTTPException(status_code=404, detail="Meeting room not found")

    # 2. 会議録の取得
    transcript = await fetch_meeting_transcript(db, room_id)
    
    # 3. メンバー情報の取得
    member_stmt = select(RoomMember).where(RoomMember.room_id == room_id)
    member_result = await db.execute(member_stmt)
    members = member_result.scalars().all()
    member_names = [m.display_name for m in members]
    member_count = len(members)

    # 4. 会議時間の計算とログの変換
    past_time_str = "0 min"
    log_items = []
    if transcript:
        try:
            start_time = datetime.strptime(transcript[0]["when"], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(transcript[-1]["when"], "%Y-%m-%d %H:%M:%S")
            duration = int((end_time - start_time).total_seconds() / 60)
            past_time_str = f"{duration} min"
        except (ValueError, IndexError):
            past_time_str = "N/A"
        
        for entry in transcript:
            log_items.append(TranslationLogItem(
                timestamp=entry["when"],
                sender_name=entry["who"],
                text=entry["what"]
            ))

    # 5. 要約の取得またはAIによる生成
    summary_dict = None
    db_summary = await get_summary_from_db(room_id, db)
    
    if db_summary:
        # DBにキャッシュされた要約が存在する場合はそれを使用
        print(f"Using cached summary for room {room_id}")
        summary_data = db_summary.get("meta", {}).get("summary", {})
        if summary_data:
            summary_dict = summary_data
    
    if not summary_dict:
        # キャッシュがない場合、AIで新たに生成
        if transcript:
            formatted_text = format_transcript_for_ai(transcript)
            summary_dict = await summarize_meeting(formatted_text)
            meeting_date_str = room.created_at.strftime("%Y-%m-%d")
            
            # 生成した要約をDBに保存
            summary_data_to_save = {
                "room_title": room.title,
                "processed_at": datetime.utcnow().isoformat(),
                "filtered_message_count": len(transcript),
                "summary": summary_dict
            }
            await save_summary_to_db(room_id, summary_data_to_save, db)
        else:
            summary_dict = {
                "main_point": "No transcript found.",
                "task": "N/A",
                "decided": "N/A"
            }
            meeting_date_str = room.created_at.strftime("%Y-%m-%d")
    else:
        meeting_date_str = room.created_at.strftime("%Y-%m-%d")

    return SummarizationResponse(
        documents=DocumentInfo(
            meeting_date=meeting_date_str,
            past_time=past_time_str,
            meeting_member=", ".join(member_names),
            meeting_name=room.title or "Untitled Meeting"
        ),
        summary=SummarizationData(
            summarization=SummarizationContent(
                main_point=summary_dict.get("main_point", ""),
                task=summary_dict.get("task", ""),
                decided=summary_dict.get("decided", "")
            ),
            meeting_date=meeting_date_str,
            past_time=past_time_str,
            meeting_member=member_count
        ),
        translation_log=log_items
    )

@router.post("/summarization/mock", response_model=SummarizationResponse, tags=["summary"])
async def create_mock_summarization(
    data: MeetingDataInput
):
    """
    large_meeting_data.json の形式のデータを受け取って要約結果を返します。
    """
    # content をパースして transcript 形式にする
    # 形式: [2024-01-23 10:00:00] 田中: おはようございます。
    transcript = []
    lines = data.content.split("\n")
    for line in lines:
        match = re.match(r"\[(.*?)\] (.*?): (.*)", line)
        if match:
            transcript.append({
                "when": match.group(1),
                "who": match.group(2),
                "what": match.group(3)
            })

    # 会議時間の計算とログの変換
    past_time_str = "0 min"
    log_items = []
    member_names_set = set()
    if transcript:
        try:
            start_time = datetime.strptime(transcript[0]["when"], "%Y-%m-%d %H:%M:%S")
            end_time = datetime.strptime(transcript[-1]["when"], "%Y-%m-%d %H:%M:%S")
            duration = int((end_time - start_time).total_seconds() / 60)
            past_time_str = f"{duration} min"
        except (ValueError, IndexError):
            past_time_str = "N/A"
        
        for entry in transcript:
            member_names_set.add(entry["who"])
            log_items.append(TranslationLogItem(
                timestamp=entry["when"],
                sender_name=entry["who"],
                text=entry["what"]
            ))

    # AIによる要約生成
    if transcript:
        formatted_text = format_transcript_for_ai(transcript)
        summary_dict = await summarize_meeting(formatted_text)
    else:
        summary_dict = {
            "main_point": "No transcript found in mock data.",
            "task": "N/A",
            "decided": "N/A"
        }

    meeting_date_str = datetime.now().strftime("%Y-%m-%d")

    return SummarizationResponse(
        documents=DocumentInfo(
            meeting_date=meeting_date_str,
            past_time=past_time_str,
            meeting_member=", ".join(list(member_names_set)),
            meeting_name=data.title
        ),
        summary=SummarizationData(
            summarization=SummarizationContent(
                main_point=summary_dict.get("main_point", ""),
                task=summary_dict.get("task", ""),
                decided=summary_dict.get("decided", "")
            ),
            meeting_date=meeting_date_str,
            past_time=past_time_str,
            meeting_member=len(member_names_set)
        ),
        translation_log=log_items
    )

@router.get("/summarization/mock-data", tags=["summary"])
async def get_mock_summarization_data():
    """
    large_meeting_data.json の内容をそのまま返します。
    """
    if not os.path.exists(JSON_FILE_PATH):
        raise HTTPException(status_code=404, detail=f"JSON file not found at {JSON_FILE_PATH}")

    try:
        with open(JSON_FILE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
