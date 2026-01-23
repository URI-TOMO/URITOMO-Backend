import json
import os
import uuid
import random
from datetime import datetime, timedelta

def generate_meeting_data(num_messages=1000):
    """
    RDBスキーマに整合した形式で、大容量の会議メッセージデータを生成します。
    """
    room_id = str(uuid.uuid4())
    room_title = "プロジェクトX 進捗会議 (大容量テストデータ)"
    
    users = [
        {"name": "田中"},
        {"name": "佐藤"},
        {"name": "鈴木"},
        {"name": "伊藤"},
    ]
    
    phrases = [
        "現在の進捗状況を教えてください。",
        "フェーズ1の実装が完了しました。",
        "デザインの修正案を共有します。",
        "テスト環境でいくつかバグが見つかりました。",
        "スケジュールへの影響はありますか？",
        "来週までに修正を完了させる予定です。",
        "了解しました。その方針で進めましょう。",
        "追加のリソースが必要かもしれません。",
        "コスト面で調整が必要ですね。",
        "次のマイルストーンを確認しましょう。",
    ]
    
    messages = []
    base_time = datetime(2026, 1, 23, 10, 0, 0)
    
    for i in range(num_messages):
        # 稀にシステムメッセージやAIメッセージを混ぜる
        rand = random.random()
        if rand < 0.05:
            sender_name = "System"
            sender_type = "system"
            message_type = "notice"
            text = f"{users[random.randint(0, len(users)-1)]['name']}さんが入室しました。"
        elif rand < 0.1:
            sender_name = "AI Assistant"
            sender_type = "ai"
            message_type = "text"
            text = "これまでの議論を要約しましょうか？"
        else:
            user = random.choice(users)
            sender_name = user["name"]
            sender_type = "human"
            # 稀に翻訳メッセージを混ぜる
            message_type = "text" if random.random() > 0.1 else "translation"
            text = random.choice(phrases)
            
        messages.append({
            "seq": i + 1,
            "sender_name": sender_name,
            "sender_type": sender_type,
            "message_type": message_type,
            "text": text,
            "created_at": (base_time + timedelta(seconds=i * 2)).isoformat()
        })
        
    data = {
        "room_id": room_id,
        "title": room_title,
        "messages": messages
    }
    
    return data

if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(output_dir, "JSON")
    os.makedirs(json_dir, exist_ok=True)
    
    output_path = os.path.join(json_dir, "generated_meeting_data.json")
    
    print(f"Generating 1000 messages...")
    data = generate_meeting_data(1000)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"Successfully generated data at: {output_path}")
