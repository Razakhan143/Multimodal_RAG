
# import speech_recognition as sr
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

TRANSCRIPT_DIR = "app/ingestion/audio_ingess"


def transcribe_audio(audio_path="app/ingestion/audio_ingess/audio.wav"):
    print("🗣️ Transcribing audio using Whisper...")

    # A missing file means an upstream step produced no audio (e.g. a silent
    # video). Return an empty transcript instead of crashing.
    if not audio_path or not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        return ""

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    client = Groq()
    with open(audio_path, "rb") as file:
        try:
            transcription = client.audio.transcriptions.create(
            file=(os.path.basename(audio_path), file.read()),
            model="whisper-large-v3",
            temperature=0,
            response_format="verbose_json",
            )

            print("📝 Transcription complete.")
            with open(os.path.join(TRANSCRIPT_DIR, "transcript.txt"), "w", encoding="utf-8") as f:
                f.write(transcription.text)
            return transcription.text
        except Exception as e:
            print(f"❌ Whisper request failed; {e}")
            return ""

    

      