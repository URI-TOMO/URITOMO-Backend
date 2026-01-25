import json
import os
import asyncio
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional
from sqlalchemy import select, desc, func
from pathlib import Path

# プロジェクトルートをPython pathに追加
project_root = str(Path(__file__).resolve().parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from app.core.config import settings
    HAS_SETTINGS = True
except ImportError:
    HAS_SETTINGS = False
    settings = None

try:
    from app.infra.db import AsyncSessionLocal
    from app.models.ai import AIEvent
    HAS_DB = True
except ImportError:
    HAS_DB = False

def load_prompt_template() -> str:
    """
    プロンプトテンプレートを読み込みます。
    """
    prompt_file = os.path.join(os.path.dirname(__file__), "summary_prompt.txt")
    try:
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        # フォールバック：テンプレートが見つからない場合はデフォルトを返す
        return """以下の会議録を分析し、要点、タスク、決定事項を抽出してください。
レスポンスは必ず以下のJSON形式で、日本語で返してください。

{
  "main_point": "会議の主な要点を簡潔にまとめてください",
  "task": "今後やるべきこと（タスク）を箇条書きのテキストで記述してください",
  "decided": "最終的に合意・決定した事項を記述してください"
}

会議録:
{transcript}
"""

async def summarize_meeting_from_file(input_path: str, output_path: str, room_id: Optional[str] = None, db_session: Optional[Any] = None) -> dict:
    """
    JSONファイルから会議録を読み取り、フィルタリングして要約し、結果をDBに保存します。
    """
    if not os.path.exists(input_path):
        return {"error": f"File not found: {input_path}"}

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # room_idがない場合はJSONから取得
    if not room_id:
        room_id = data.get("room_id")

    # メッセージのフィルタリング (人間によるテキスト/翻訳のみ)
    messages = data.get("messages", [])
    filtered_messages = [
        msg for msg in messages 
        if msg.get("sender_type") == "human" and msg.get("message_type") in ["text", "translation"]
    ]

    # トランスクリプトのフォーマット
    formatted_transcript = ""
    for msg in filtered_messages:
        formatted_transcript += f"[{msg['created_at']}] {msg['sender_name']}: {msg['text']}\n"

    # OpenAIによる要約
    summary_result = await summarize_meeting(formatted_transcript)
    
    # 結果の構築
    final_output = {
        "room_id": room_id or data.get("room_id"),
        "room_title": data.get("title"),
        "processed_at": datetime.now().isoformat(),
        "filtered_message_count": len(filtered_messages),
        "summary": summary_result
    }

    # DBに保存 (これが推奨される保存方法)
    db_saved = False
    if room_id and HAS_DB and db_session:
        try:
            await save_summary_to_db(room_id, final_output, db_session)
            db_saved = True
        except Exception as e:
            print(f"Failed to save to DB in summarize_meeting_from_file: {e}")

    # ファイル保存 (バックアップまたはデバッグ用)
    if output_path:
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save to file: {e}")

    if not db_saved and not output_path:
        print("Warning: Summary result was not saved to DB or file.")

    return final_output

async def get_next_seq(room_id: str, db_session: Any) -> int:
    """
    指定された room_id の次のseq番号を取得します。
    """
    if not HAS_DB:
        return 0
    
    try:
        from sqlalchemy import func
        stmt = select(func.max(AIEvent.seq)).where(AIEvent.room_id == room_id)
        result = await db_session.execute(stmt)
        max_seq = result.scalar()
        return (max_seq or 0) + 1
    except Exception as e:
        print(f"Error getting next seq: {str(e)}")
        return 0


async def save_summary_to_db(room_id: str, summary_data: dict, db_session: Any) -> Optional[str]:
    """
    要約結果をデータベースに保存します。
    seq番号を自動決定し、一意制約エラーを防止します。
    
    Returns:
        保存されたAIEventのid、またはエラー時はNone
    """
    if not HAS_DB:
        return None
    
    try:
        # 次のseq番号を自動決定
        next_seq = await get_next_seq(room_id, db_session)
        ai_event_id = str(uuid.uuid4())
        
        # AIEventレコードを作成
        ai_event = AIEvent(
            id=ai_event_id,
            room_id=room_id,
            seq=next_seq,
            event_type="summary",
            text=summary_data.get("summary", {}).get("main_point", ""),
            meta={
                "room_title": summary_data.get("room_title"),
                "processed_at": summary_data.get("processed_at"),
                "filtered_message_count": summary_data.get("filtered_message_count"),
                "summary": summary_data.get("summary", {}),
                "full_data": summary_data
            }
        )
        
        db_session.add(ai_event)
        await db_session.commit()
        return ai_event_id
    except Exception as e:
        await db_session.rollback()
        print(f"Failed to save summary to DB: {str(e)}")
        return None


async def get_summary_from_db(room_id: str, db_session: Any) -> Optional[dict]:
    """
    指定された room_id の最新の要約結果をDBから取得します。
    
    Returns:
        要約データを含むdictionary、またはなければNone
    """
    if not HAS_DB:
        return None
    
    try:
        from sqlalchemy import desc
        stmt = (
            select(AIEvent)
            .where(AIEvent.room_id == room_id)
            .where(AIEvent.event_type == "summary")
            .order_by(desc(AIEvent.created_at))
            .limit(1)
        )
        result = await db_session.execute(stmt)
        ai_event = result.scalar_one_or_none()
        
        if ai_event and ai_event.meta:
            return {
                "id": ai_event.id,
                "seq": ai_event.seq,
                "created_at": ai_event.created_at.isoformat(),
                "text": ai_event.text,
                "meta": ai_event.meta
            }
        return None
    except Exception as e:
        print(f"Error getting summary from DB: {str(e)}")
        return None


async def summarize_meeting(text: str) -> dict:
    """
    OpenAI API を使用して会議録を要約します。
    APIキーがない場合はモック要約を返します。
    """
    openai_api_key = None
    summary_model = "gpt-4o"
    
    if HAS_SETTINGS and hasattr(settings, 'openai_api_key'):
        openai_api_key = settings.openai_api_key
    else:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        # OpenAI APIキーがない場合はモック要約を返す
        return {
            "main_point": "【モック要約】会議ではプロジェクトXの進捗状況、デザイン修正案、テスト環境でのバグ対応などが議論されました。",
            "task": "- フェーズ1の実装完了と検証\n- デザイン修正案の実装\n- テスト環境のバグ修正（来週までに完了予定）\n- 次のマイルストーン確認",
            "decided": "提案された方針で進める。必要に応じてリソース調整を検討する。"
        }

    # OpenAIを使用した実際の要約処理（有効なAPIキーがある場合）
    try:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=openai_api_key)
        
        # 大容量データの場合、トークン制限を考慮して末尾の一定文字数のみを送る
        max_char_limit = 30000 
        if len(text) > max_char_limit:
            text = text[-max_char_limit:]
            text = "[...前略...]\n" + text

        # プロンプトテンプレートを読み込む (外部ファイルを優先)
        prompt_template = load_prompt_template()
        prompt = prompt_template.format(transcript=text)

        response = await client.chat.completions.create(
            model=summary_model,
            messages=[
                {"role": "system", "content": "あなたは優秀な会議進行役、および議事録作成者です。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {
            "main_point": f"Error during summarization: {str(e)}",
            "task": "N/A",
            "decided": "N/A"
        }

if __name__ == "__main__":
    from datetime import datetime
    
    async def main():
        current_dir = os.path.dirname(os.path.abspath(__file__))
        input_file = os.path.join(current_dir, "JSON", "generated_meeting_data.json")
        output_file = os.path.join(current_dir, "JSON", "summary_result.json")
        
        print(f"Summarizing data from {input_file}...")
        result = await summarize_meeting_from_file(input_file, output_file)
        
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Summary saved to {output_file}")
            print("\n--- Summary Preview ---")
            print(json.dumps(result["summary"], ensure_ascii=False, indent=2))

    asyncio.run(main())

