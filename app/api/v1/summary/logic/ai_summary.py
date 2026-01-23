import json
import os
import asyncio
import sys
from datetime import datetime
from typing import List, Dict, Any
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

async def summarize_meeting_from_file(input_path: str, output_path: str) -> dict:
    """
    JSONファイルから会議録を読み取り、フィルタリングして要約し、結果をファイルに保存します。
    """
    if not os.path.exists(input_path):
        return {"error": f"File not found: {input_path}"}

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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
    
    # 部屋情報などを付与
    final_output = {
        "room_id": data.get("room_id"),
        "room_title": data.get("title"),
        "processed_at": datetime.now().isoformat(),
        "filtered_message_count": len(filtered_messages),
        "summary": summary_result
    }

    # 結果をファイルに保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    return final_output

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

        prompt = f"""以下の会議録を分析し、要点、タスク、決定事項を抽出してください。
レスポンスは必ず以下のJSON形式で、日本語で返してください。

{{
  "main_point": "会議の主な要点を簡潔にまとめてください",
  "task": "今後やるべきこと（タスク）を箇条書きのテキストで記述してください",
  "decided": "最終的に合意・決定した事項を記述してください"
}}

会議録:
{text}
"""

        response = await client.chat.completions.create(
            model=summary_model,
            messages=[{"role": "user", "content": prompt}],
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

