
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
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
import torch
import numpy as np

# import mysql.connector
# from mysql.connector import Error

# Try importing alkana, handle if missing

try:
    import alkana
except ImportError:
    alkana = None

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("LiveKitAgent")

# Constants
VAD_SAMPLE_RATE = 16000
VAD_CHUNK_SIZE = 512  # 32ms
SAMPLE_RATE_48K = 48000
MEETING_CONTEXT = "IT開発会議, Python, LiveKit, VAD, API, フロントエンド, バックエンド, 議事録, Docker, AWS"

# Hallucination filters
MIN_SPEECH_DURATION_MS = 100
MIN_RMS_THRESHOLD = 0.005
VAD_START_THRESHOLD = 2
VAD_END_THRESHOLD = 7
MAX_CONCURRENT_STT = 5

HALLUCINATION_PHRASES = [
    "ご清聴ありがとうございました", "視聴ありがとうございました", 
    "日本語と韓国語、英語が話されます", "字幕 視聴ありがとうございました",
    "Thank you for watching", "Amara.org", "MBC", "Subtitles by",
    "ありがとうございました", "Please subscribe", "Copyright"
]

# OpenAI Client
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
'''
def get_mysql_connection():
    try:
        conn = mysql.connector.connect(
            host=os.getenv("MYSQL_HOST"),
            user=os.getenv("MYSQL_USER"),
            password=os.getenv("MYSQL_PASSWORD"),
            database=os.getenv("MYSQL_DATABASE")
        )

        return conn
    except Error as e:
        logger.error(f"MySQL Connection Error: {e}")
        return None

def get_room_id_from_db():
    conn = get_mysql_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    cursor.execute("SELECT room_id FROM rooms LIMIT 1")
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def get_participant_id_from_db(identity):
    conn = get_mysql_connection()
    if not conn:
        return None
    cursor = conn.cursor()
    cursor.execute("SELECT participant_id FROM participants WHERE identity=%s", (identity,))
    row = cursor.fetchone()
    conn.close()
    return row[0] if row else None

def save_json_to_database(data):
    conn = get_mysql_connection()
    if not conn:
        return
    cursor = conn.cursor()

    try:
        cursor.execute("""
            INSERT INTO transcriptions 
            (room_id, participant_id, participant_name, is_speaking, original_text, timestamp, sequence, language)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            data['room_id'],
            data['participant_id'],
            data['participant_name'],
            data['is_speaking'],
            data['Original'],
            data['timestamp'],
            data['sequence'],
            data['Language']
        ))

        conn.commit()
    except Error as e:
        logger.error(f"MySQL Insert Error: {e}")
    finally:
        conn.close()
'''


def is_mostly_english(text: str) -> bool:
    # 日本語の文字（ひらがな、カタカナ、漢字）が含まれているかチェック
    # 含まれていなければ「英語」とみなす
    if re.search(r'[ぁ-んァ-ン一-龥]', text):
        return False
    return True


class CapturedSpeaker(BaseModel):
    user_id: str
    user_name: str
    captured_at: str
    sequence: int          # 👈 同時発話優先順位
    text: str
    language: str

def english_to_katakana(text: str) -> str:
    if not alkana:
        return text
    words = text.split()
    converted_words = []
    for word in words:
        clean_word = word.strip(".,?!")
        kana = alkana.get_kana(clean_word)
        converted_words.append(kana if kana else word)
    return " ".join(converted_words)

def calculate_rms(data_np: np.ndarray) -> float:
    if len(data_np) == 0:
        return 0.0
    return math.sqrt(np.mean(data_np**2))

class TranscriptionAgent:
    def __init__(self):
        self.livekit_url = os.getenv('LIVEKIT_URL')
        self.api_key = os.getenv('LIVEKIT_API_KEY')
        self.api_secret = os.getenv('LIVEKIT_API_SECRET')


        #self.room_id = get_room_id_from_db()
        #self.participant_id = get_participant_id_from_db(participant.identity)

        self.room_id = "1"  # 固定値で使用
        self.participant_id = "user_id"  # デモ用
        self.room = None
        self.meeting_start_time = None
        self.speaker_states = {}  # 同時発話優先順位
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.vad_model = None
        self.stt_queue = asyncio.Queue()
        self.stt_worker_task = None
        self.stt_semaphore = asyncio.Semaphore(MAX_CONCURRENT_STT)

    async def start(self):
        if not all([self.livekit_url, self.api_key, self.api_secret]):
            logger.error("Missing credentials in .env")
            return

        if not alkana:
            logger.warning("Dependency 'alkana' is missing. Katakana conversion disabled.")

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

        @self.room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logger.info(f"Participant connected: {participant.identity}")

        @self.room.on("track_published")

        def on_track_published(
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant
        ):

            if publication.kind != rtc.TrackKind.KIND_AUDIO:
                return

            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                return

            async def subscribe_and_process():
                publication.set_subscribed(True)
                return
                logger.info(f"🎤 Subscribed MIC audio from {participant.identity}")
                audio_stream = rtc.AudioStream(track)
                asyncio.create_task(
                    self.process_audio_stream(audio_stream, participant)
                )

            asyncio.create_task(subscribe_and_process())

        @self.room.on("track_subscribed")
        def on_track_subscribed(
            track: rtc.Track,
            publication: rtc.RemoteTrackPublication,
            participant: rtc.RemoteParticipant
        ):
            if publication.kind != rtc.TrackKind.KIND_AUDIO:
                return

            if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                return

            logger.info(f"🎤 Subscribed MIC audio from {participant.identity}")
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(
                self.process_audio_stream(audio_stream, participant)
            )

        @self.room.on("active_speakers_changed")
        def on_active_speakers_changed(speakers: list[rtc.Participant]):
            if not speakers:
                return
            # 同時発話優先順位を更新（0 が最優先）
            for sequence, speaker in enumerate(speakers):
                self.speaker_states[speaker.identity] = {"sequence": sequence}

        logger.info(f"Connecting to {self.livekit_url}, Room: {self.room_id}")

        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity("python-bot-stt")
            .with_name("STT Bot")
            .with_grants(api.VideoGrants(room_join=True, room=self.room_id))
            .to_jwt()
        )

        try:
            await self.room.connect(
                self.livekit_url,
                token,
                rtc.RoomOptions(auto_subscribe=False)
            )

            local_participant = self.room.local_participant

            for publication in local_participant.track_publications.values():
                if publication.kind != rtc.TrackKind.KIND_AUDIO:
                    continue
                if publication.source != rtc.TrackSource.SOURCE_MICROPHONE:
                    continue
                track = publication.track

                logger.info("🎤 Subscribed MIC audio from LOCAL participant (you)")

                audio_stream = rtc.AudioStream(track)
                asyncio.create_task(
                    self.process_audio_stream(audio_stream, local_participant)
                )
            logger.info(f"Checking existing participants... Count: {len(self.room.remote_participants)}")
            for participant_id, participant in self.room.remote_participants.items():
                logger.info(f"Checking existing participant: {participant.identity}")
                
                for publication in participant.track_publications.values():
                    # マイク音声（Audio）かつ、SourceがMicrophoneの場合のみ
                    if publication.kind == rtc.TrackKind.KIND_AUDIO and publication.source == rtc.TrackSource.SOURCE_MICROPHONE:
                        try:
                            publication.set_subscribed(True)
                            logger.info(f"Requested subscription for existing track from {participant.identity}")
                        except Exception as e:
                            logger.error(f"Failed to subscribe to existing track: {e}")

            self.meeting_start_time = datetime.now()
            logger.info("✅ Connected! System ready.")
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
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
        PRE_BUFFER_SIZE = 15 # 16kHz 32ms frames approx 0.5s
        pre_speech_buffer_16k = deque(maxlen=PRE_BUFFER_SIZE)
        
        # リサンプラーの作成
        try:
            resampler = rtc.AudioResampler(input_rate=SAMPLE_RATE_48K, output_rate=VAD_SAMPLE_RATE)
        except Exception as e:
            logger.error(f"❌ Failed to create resampler: {e}")
            return

        loop = asyncio.get_running_loop()

        try:
            async for frame_event in stream:
                frame = frame_event.frame
                
                try:
                    resampled_frames = []
                    for f in resampler.push(frame):
                        resampled_frames.append(f)
                except Exception as e:
                    logger.error(f"❌ Crash at Resampler: {e}")
                    continue

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
                        # logger.info(f"✅ VAD Success: {speech_prob:.2f}")
                    except Exception as e:
                        logger.error(f"❌ Crash at VAD Model: {e}")
                        speech_prob = 0.0

                    # --- VAD判定ロジック ---
                    is_speech_frame = speech_prob > 0.5

                    if is_speech_frame:
                        end_confirm_count = 0
                        start_confirm_count += 1
                        if not is_triggered and start_confirm_count >= VAD_START_THRESHOLD:
                            is_triggered = True
                            logger.info(f"🗣️ Speech START detected for {participant.identity}")
                            speech_buffer_16k.extend(list(pre_speech_buffer_16k))
                            pre_speech_buffer_16k.clear()
                    else:
                        start_confirm_count = 0
                        if is_triggered:
                            end_confirm_count += 1

                    if is_triggered:
                        speech_buffer_16k.append(chunk_data)
                    else:
                        pre_speech_buffer_16k.append(chunk_data)

                    if is_triggered and end_confirm_count >= VAD_END_THRESHOLD:
                        is_triggered = False
                        start_confirm_count = 0
                        end_confirm_count = 0

                        total_bytes = sum(len(b) for b in speech_buffer_16k)
                        duration_ms = (total_bytes / (VAD_SAMPLE_RATE * 2)) * 1000
                        if duration_ms < MIN_SPEECH_DURATION_MS:
                            speech_buffer_16k = []
                            continue

                        full_audio = b"".join(speech_buffer_16k)
                        data_np = np.frombuffer(full_audio, dtype=np.int16).astype(np.float32) / 32768.0
                        rms = calculate_rms(data_np)
                        if rms < MIN_RMS_THRESHOLD:
                            speech_buffer_16k = []
                            continue

                        await self.stt_queue.put((list(speech_buffer_16k), participant))
                        speech_buffer_16k = []

        except Exception as e:
            logger.error(f"Error in audio processing for {participant.identity}: {e}")
            
    async def stt_worker(self):
        logger.info("STT Worker started.")
        while True:
            try:
                buffer_frames, participant = await self.stt_queue.get()
                asyncio.create_task(self.process_transcription_task(buffer_frames, participant))
                self.stt_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker Error: {e}")



    async def process_transcription_task(self, buffer_frames, participant):
        async with self.stt_semaphore:
            await self.transcribe_and_notify(buffer_frames, participant)

    async def transcribe_and_notify(self, buffer_frames, participant):
        if not buffer_frames:
            return
        wav_buffer = io.BytesIO()
        wav_buffer.name = "audio.wav"

        try:
            raw_audio = b"".join(buffer_frames)
            data_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
            if calculate_rms(data_np / 32768.0) < MIN_RMS_THRESHOLD:
                return

            max_amp = np.max(np.abs(data_np))
            target_level = 25000.0

            if max_amp > 0 and max_amp < 15000:
                gain = min(target_level / max_amp, 5.0)
                data_np = np.clip(data_np * gain, -32768, 32767)
            normalized_audio = data_np.astype(np.int16).tobytes()

            with wave.open(wav_buffer, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(VAD_SAMPLE_RATE)
                wf.writeframes(normalized_audio)

            wav_buffer.seek(0)



            transcript = await openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                temperature=0.0,
                response_format="verbose_json",
                prompt=MEETING_CONTEXT

            )

            text = transcript.text.strip()
            if not text:
                return
            if any(phrase.lower() in text.lower() for phrase in HALLUCINATION_PHRASES):
                return

            words = text.split()
            if len(words) > 5 and len(set(words)) / len(words) < 0.2:
                return

            raw_lang = getattr(transcript, 'language', 'unknown').lower()
            lang_map = {"english": "en", "japanese": "ja", "korean": "ko"}
            language = lang_map.get(raw_lang, raw_lang)

            # 【追加ロジック】
            # Whisperが「日本語」と判定しても、テキストの中身に日本語文字がなく
            # アルファベットだけなら、強制的に "en" に書き換える
            if language == 'ja' and is_mostly_english(text):
                logger.info(f"Language override: Detected 'ja' but text is English ('{text}'). Switched to 'en'.")
                language = 'en'

            # カタカナ変換を実行
            final_text = english_to_katakana(text) if language == 'en' else text
            captured_at_str = "0:00:00"
            if self.meeting_start_time:
                elapsed = datetime.now() - self.meeting_start_time
                captured_at_str = str(elapsed).split('.')[0]

            state = self.speaker_states.get(participant.identity, {"sequence": 0})
            db_json = {
                "room_id": self.room_id,
                "participant_id": self.participant_id,
                "participant_name": participant.identity or "Unknown",
                "is_speaking": True,
                "Original": final_text,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "sequence": state["sequence"],
                "Language": language.capitalize()

            }

            print("📦 DB JSON:\n", json.dumps(db_json, indent=2, ensure_ascii=False))
            # save_json_to_database(db_json)

        except Exception as e:
            logger.error(f"STT Error: {e}")
        finally:
            wav_buffer.close()

if __name__ == "__main__":
    agent = TranscriptionAgent()
    async def main():
        await agent.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass