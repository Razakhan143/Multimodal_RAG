
# import speech_recognition as sr
import os
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

TRANSCRIPT_DIR = "app/ingestion/audio_ingess"
SEGMENTS_OUT = os.path.join(TRANSCRIPT_DIR, "segments.json")


def transcribe_audio(audio_path="app/ingestion/audio_ingess/audio.wav"):
    """Transcribe audio with Whisper and persist per-segment timestamps.

    Returns the plain transcript text (callers test it for truthiness). The
    timestamped segments are written to ``segments.json`` so the ingestion layer
    can build timestamp-aware chunks — this is what lets the video panel jump to
    the exact moment an answer is spoken rather than to a visually-matched frame.
    """
    print("🗣️ Transcribing audio using Whisper...")

    # A missing file means an upstream step produced no audio (e.g. a silent
    # video). Return an empty transcript instead of crashing.
    if not audio_path or not os.path.exists(audio_path):
        print(f"⚠️ Audio file not found: {audio_path}")
        _write_segments([])
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

            _write_segments(_extract_segments(transcription))
            return transcription.text
        except Exception as e:
            print(f"❌ Whisper request failed; {e}")
            _write_segments([])
            return ""


def _extract_segments(transcription):
    """Normalise Whisper's verbose_json segments to [{start, end, text}, …]."""
    raw = getattr(transcription, "segments", None)
    if raw is None and isinstance(transcription, dict):
        raw = transcription.get("segments")
    segments = []
    for seg in (raw or []):
        get = seg.get if isinstance(seg, dict) else (lambda k, d=None: getattr(seg, k, d))
        text = (get("text", "") or "").strip()
        if not text:
            continue
        segments.append({
            "start": float(get("start", 0.0) or 0.0),
            "end":   float(get("end", 0.0) or 0.0),
            "text":  text,
        })
    return segments


def _write_segments(segments):
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    with open(SEGMENTS_OUT, "w", encoding="utf-8") as f:
        json.dump(segments, f)


def load_segments():
    """Load the most recently transcribed segments, or [] if none."""
    if not os.path.exists(SEGMENTS_OUT):
        return []
    try:
        with open(SEGMENTS_OUT, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []
