
import asyncio
import os
import logging
import io
import wave
import concurrent.futures
from datetime import datetime
import math
from collections import deque
import json
import re

from dotenv import load_dotenv
from livekit import api, rtc
from pydantic import BaseModel
from openai import AsyncOpenAI
import torch
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("LiveKitAgentJP")

# --- Constants ---
VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # 32ms
SAMPLE_RATE_48K = 48000


# フィルタとしきい値（超高速応答・逐次出力設定）
MIN_SPEECH_DURATION_MS = 300
MAX_SPEECH_DURATION_MS = 2500    # 2.5秒ごとに区切って出力。超リアルタイム性を確保 (5000 -> 2500)
MIN_RMS_THRESHOLD = 0.07
VAD_START_THRESHOLD = 2
VAD_END_THRESHOLD = 2          # 約64msの無音で即座に送信 (3 -> 2)
MAX_CONCURRENT_STT = 8

# OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def contains_japanese(text: str) -> bool:
    """ひらがな、またはカタカナが含まれているかチェック（漢字のみやアルファベットのみを除外するための基準）"""
    return bool(re.search(r'[ぁ-んァ-ン]', text))

def is_mostly_alphabet(text: str) -> bool:
    """テキストが主にアルファベットで構成されているかチェック"""
    # 記号を除去
    clean_text = re.sub(r'[\s、。！？?.!,]', '', text)
    if not clean_text:
        return False
    # アルファベットの文字数をカウント
    al_count = len(re.findall(r'[a-zA-Z]', clean_text))
    return (al_count / len(clean_text)) > 0.5

def calculate_rms(data_np: np.ndarray) -> float:
    if len(data_np) == 0:
        return 0.0
    return math.sqrt(np.mean(data_np**2))

class TranscriptionAgentJP:
    def __init__(self):
        self.livekit_url = os.getenv('LIVEKIT_URL')
        self.api_key = os.getenv('LIVEKIT_API_KEY')
        self.api_secret = os.getenv('LIVEKIT_API_SECRET')

        self.room_id = "1"  # デモ用固定値
        self.participant_id = "user_id_jp"
        self.room = None
        self.meeting_start_time = None
        self.speaker_states = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.vad_model = None
        self.stt_queue = asyncio.Queue()
        self.stt_worker_task = None
        self.stt_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STT)
        self._background_tasks = set()
        self.participant_contexts = {} # 各参加者の直前の文字起こし結果を保持

    async def start(self):
        if not all([self.livekit_url, self.api_key, self.api_secret]):
            logger.error("Missing credentials in .env")
            return

        self.stt_worker_task = asyncio.create_task(self.stt_worker())
        self.room = rtc.Room()

        logger.info("Loading Silero VAD model...")
        try:
            self.vad_model, _ = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    trust_repo=True,
                    onnx=False
                )
            )
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            return

        @self.room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                logger.info(f"Subscribed MIC audio from {participant.identity}")
                audio_stream = rtc.AudioStream(track)
                task = asyncio.create_task(self.process_audio_stream(audio_stream, participant))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

        @self.room.on("active_speakers_changed")
        def on_active_speakers_changed(speakers: list[rtc.Participant]):
            for sequence, speaker in enumerate(speakers):
                self.speaker_states[speaker.identity] = {"sequence": sequence}

        logger.info(f"Connecting to {self.livekit_url}, Room: {self.room_id}")

        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity("python-bot-stt-jp")
            .with_name("STT Bot Japanese")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_id))
            .to_jwt()
        )

        try:
            await self.room.connect(self.livekit_url, token, rtc.RoomOptions(auto_subscribe=True))
            self.meeting_start_time = datetime.now()
            logger.info("✅ Connected! Japanese STT System ready.")
            await asyncio.Event().wait()
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if self.room and self.room.isconnected:
                await self.room.disconnect()
            self.executor.shutdown()
            if self.stt_worker_task:
                self.stt_worker_task.cancel()

    async def process_audio_stream(self, stream: rtc.AudioStream, participant: rtc.RemoteParticipant):
        speech_buffer_16k = []
        vad_accumulator = bytearray()
        is_triggered = False
        start_confirm_count = 0
        end_confirm_count = 0
        PRE_BUFFER_SIZE = 3 # さらに削減（5 -> 3）約0.1秒分。
        pre_speech_buffer_16k = deque(maxlen=PRE_BUFFER_SIZE)
        
        try:
            resampler = rtc.AudioResampler(input_rate=SAMPLE_RATE_48K, output_rate=VAD_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"Failed to create resampler for {participant.identity}: {e}")
            return

        loop = asyncio.get_running_loop()

        try:
            async for frame_event in stream:
                frame = frame_event.frame
                
                resampled_frames = list(resampler.push(frame))
                for resampled_frame in resampled_frames:
                    vad_accumulator.extend(resampled_frame.data)

                CHUNK_BYTES = VAD_CHUNK_SIZE * 2
                while len(vad_accumulator) >= CHUNK_BYTES:
                    chunk_data = vad_accumulator[:CHUNK_BYTES]
                    del vad_accumulator[:CHUNK_BYTES]

                    data_np = np.frombuffer(chunk_data, dtype=np.int16).astype(np.float32) / 32768.0
                    tensor = torch.from_numpy(data_np)

                    try:
                        speech_prob = await loop.run_in_executor(
                            self.executor,
                            lambda: self.vad_model(tensor, VAD_SAMPLE_RATE).item()
                        )
                    except Exception:
                        speech_prob = 0.0

                    is_speech_frame = speech_prob > 0.5

                    if is_speech_frame:
                        end_confirm_count = 0
                        start_confirm_count += 1
                        if not is_triggered and start_confirm_count >= VAD_START_THRESHOLD:
                            is_triggered = True
                            logger.info(f"Speech START (JP) for {participant.identity}")
                            speech_buffer_16k.extend(list(pre_speech_buffer_16k))
                            pre_speech_buffer_16k.clear()
                    else:
                        start_confirm_count = 0
                        if is_triggered:
                            end_confirm_count += 1

                    # バッファへの蓄積
                    if is_triggered:
                        speech_buffer_16k.append(chunk_data)
                        
                        # --- 長文リアルタイム出力対策 ---
                        # 一定時間（MAX_SPEECH_DURATION_MS）を超えたら強制的に一度文字起こしに回す
                        total_bytes = sum(len(b) for b in speech_buffer_16k)
                        current_duration_ms = (total_bytes / (VAD_SAMPLE_RATE * 2)) * 1000
                        if current_duration_ms >= MAX_SPEECH_DURATION_MS:
                            logger.info(f"Forcing partial transcription (long speech) for {participant.identity}")
                            # 送信用に現在のバッファをコピー
                            frames_to_stt = list(speech_buffer_16k)
                            await self.stt_queue.put((frames_to_stt, participant))
                            
                            # 次回のバッファに直近の約200ms分（約6フレーム）を残してオーバーラップさせる
                            OVERLAP_FRAMES = 6
                            remaining_frames = list(speech_buffer_16k)[-OVERLAP_FRAMES:] if len(speech_buffer_16k) > OVERLAP_FRAMES else []
                            speech_buffer_16k = remaining_frames # バッファをリセットせずオーバーラップ分を残す
                    else:
                        pre_speech_buffer_16k.append(chunk_data)

                    # 発話終了判定
                    if is_triggered and end_confirm_count >= VAD_END_THRESHOLD:
                        is_triggered = False
                        start_confirm_count = 0
                        end_confirm_count = 0

                        total_bytes = sum(len(b) for b in speech_buffer_16k)
                        duration_ms = (total_bytes / (VAD_SAMPLE_RATE * 2)) * 1000
                        if duration_ms < MIN_SPEECH_DURATION_MS:
                            speech_buffer_16k = []
                            continue

                        # --- 周辺のざわつき対策 (SNRチェック) ---
                        full_audio = b"".join(speech_buffer_16k)
                        data_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        
                        rms = calculate_rms(data_np)
                        if rms < MIN_RMS_THRESHOLD:
                            speech_buffer_16k = []
                            continue
                            
                        # クレストファクター（最大値/実効値）による判定
                        peak = np.max(np.abs(data_np))
                        crest_factor = peak / rms if rms > 0 else 0
                        
                        # ざわつき判定（ノイズ耐性を再度強化）
                        if crest_factor < 3.2: # 2.5 -> 3.2
                            speech_buffer_16k = []
                            continue

                        await self.stt_queue.put((list(speech_buffer_16k), participant))
                        speech_buffer_16k = []

        except Exception as e:
            logger.error(f"Error in audio processing for {participant.identity}: {e}")

    async def stt_worker(self):
        while True:
            try:
                buffer_frames, participant = await self.stt_queue.get()
                task = asyncio.create_task(self.transcribe_and_notify(buffer_frames, participant))
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)
                self.stt_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"STT Worker Error: {e}")

    async def clean_transcript_with_llm(self, raw_text: str):
        """LLM (GPT-4o) を使用してハルシネーションや不自然な繰り返しを除去するガードレール"""
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o", # 高速モデルに変更
                messages=[
                    {"role": "system", "content": (
                        "あなたは日本語STTの高品質浄化エンジンです。ルール：\n"
                        "1. 『ご視聴ありがとうございました』『チャンネル登録』『お疲れ様でした』などの、文脈に合わない定型文やハルシネーションは徹底的に排除し、空文字（\"\"）を返す。\n"
                        "2. 相槌（はい、ええ等）は会話の一部なので保護する。\n"
                        "3. 吃音や不自然な繰り返しを自然な日本語に修正し、校正後テキストのみ出力（解説不要）。"
                    )},
                    {"role": "user", "content": raw_text}
                ],
                temperature=0.0,
                max_tokens=48 # さらに削減して高速化 (64 -> 48)
            )
            cleaned_text = response.choices[0].message.content.strip()
            return cleaned_text
        except Exception as e:
            logger.error(f"LLM Guardrail Error: {e}")
            return raw_text # エラー時は生データを返す

    async def transcribe_and_notify(self, buffer_frames, participant):
        async with self.stt_semaphore:
            if not buffer_frames:
                return
            
            wav_buffer = io.BytesIO()
            wav_buffer.name = "audio.wav"

            try:
                raw_audio = b"".join(buffer_frames)
                data_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
                
                # 音量正規化（小さな声をAIが認識しやすいよう増幅）
                max_amp = np.max(np.abs(data_np))
                if max_amp > 2000: # さらに小さな声から反応 (3000 -> 2000)
                    if max_amp < 18000:
                        # ゲインの上限を引き上げ (3.0 -> 4.5)
                        gain = min(25000.0 / max_amp, 4.5) 
                        data_np = np.clip(data_np * gain, -32768, 32767)
                else:
                    # 音が小さすぎる場合は無音として扱う（AIに渡さない、またはそのまま）
                    pass
                normalized_audio = data_np.astype(np.int16).tobytes()

                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(VAD_SAMPLE_RATE)
                    wf.writeframes(normalized_audio)

                wav_buffer.seek(0)
                
                # 前回のコンテキスト（ある場合）を取得してプロンプトに含める
                prev_text = self.participant_contexts.get(participant.identity, "")
                base_prompt = "日本語の自然な会話です。句読点を適切に使用し、文脈に応じた正しい漢字変換を行ってください。"
                full_prompt = f"{base_prompt} 前回までの内容: {prev_text}" if prev_text else base_prompt

                # Whisper API プロンプト
                transcript = await openai_client.audio.transcriptions.create(
                    model="whisper-1",
                    file=wav_buffer,
                    language="ja",
                    temperature=0.0,
                    response_format="verbose_json",
                    prompt=full_prompt
                )

                # --- セグメント単位での厳格な評価ロジック ---
                segments = getattr(transcript, 'segments', [])
                valid_segments = []

                for seg in segments:
                    seg_text = getattr(seg, 'text', '').strip()
                    seg_logprob = getattr(seg, 'avg_logprob', -10)
                    seg_no_speech = getattr(seg, 'no_speech_prob', 1)

                    # メタ発言（自己解説）やプロンプト漏れを厳格にチェック
                    forbidden = [
                        "書き起こし", "ハルシネーション", "厳禁", "補完", "無視", "不明瞭", 
                        "日常会話", "自然な会話", "句読点", "文脈", "漢字", "かな混じり", "判断",
                        "不自然", "見当たりません", "問題ありません", "そのまま", "ありません", "テキスト"
                    ]
                    if any(k in seg_text for k in forbidden):
                        logger.debug(f"Blocked meta-comment/hallucination: '{seg_text}'")
                        continue

                    # セグメントごとの動的信頼度判定
                    # 1. まずゴミ取り（プロンプト漏れ）を先に行う
                    clean_seg = re.sub(r'^[）)等(等)、。 ]+', '', seg_text).strip()
                    
                    if not clean_seg:
                        continue

                    # 2. クリーンなテキストで判定（ハルシネーション防止のため厳格化）
                    # 短文（6文字未満）は特に捏造が多いため厳しく
                    limit_logprob = -0.6 if len(clean_seg) < 6 else -1.2
                    limit_no_speech = 0.25 if len(clean_seg) < 6 else 0.4

                    if seg_logprob > limit_logprob and seg_no_speech < limit_no_speech:
                        valid_segments.append(clean_seg)
                    else:
                        # ターミナルを汚さないよう、棄却ログは logger.debug に変更
                        logger.debug(f"Filtered segment: '{clean_seg}' (log:{seg_logprob:.2f}, no:{seg_no_speech:.2f})")

                # 有効なセグメントのみを結合
                raw_text = " ".join(valid_segments).strip()
                
                # 句読点などを除いた純粋な文字数をカウント
                pure_text = re.sub(r'[、。！？!.?() ]+', '', raw_text)
                if not pure_text or len(pure_text) < 2:
                    return

                # 4. 繰り返し（エントロピー）チェック
                # 「んんんん」「あわわわ」等の単調な繰り返しを排除
                unique_chars = len(set(pure_text))
                if len(pure_text) >= 4 and unique_chars / len(pure_text) < 0.4:
                    logger.debug(f"Repetitive rejection (low entropy): '{pure_text}'")
                    return

                # --- GPT-4o による LLM ガードレール ---
                text = await self.clean_transcript_with_llm(raw_text)
                if not text:
                    logger.debug(f"LLM Guardrail rejected text: '{raw_text}'")
                    return

                # 【重要】日本語文字チェック（校正後のテキストで実施）
                if not contains_japanese(text) or is_mostly_alphabet(text):
                    return

                captured_at_str = "0:00:00"
                if self.meeting_start_time:
                    elapsed = datetime.now() - self.meeting_start_time
                    captured_at_str = str(elapsed).split('.')[0]

                state = self.speaker_states.get(participant.identity, {"sequence": 0})
                db_json = {
                    "room_id": self.room_id,
                    "participant_id": participant.identity or "unknown",
                    "participant_name": participant.name or participant.identity or "Unknown",
                    "original_text": text,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "sequence": state["sequence"],
                    "language": "Japanese"
                }

                print(json.dumps(db_json, indent=2, ensure_ascii=False))

                # コンテキストの更新（次回の分割分へ文脈を引き継ぐ）
                self.participant_contexts[participant.identity] = text

            except Exception as e:
                logger.error(f"STT Error: {e}")
            finally:
                wav_buffer.close()

if __name__ == "__main__":
    agent = TranscriptionAgentJP()
    async def main():
        await agent.start()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
