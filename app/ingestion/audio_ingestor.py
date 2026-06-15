
import speech_recognition as sr
def transcribe_audio(audio_path):
    print("🗣️ Transcribing audio using Whisper...")
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            # Utilizing the local or API-driven whisper model through speech_recognition
            text = recognizer.recognize_whisper(audio_data)
            print("📝 Transcription complete.")
            return text
        except sr.UnknownValueError:
            print("❌ Whisper could not understand the audio.")
            return ""
        except sr.RequestError as e:
            print(f"❌ Whisper request failed; {e}")
            return ""
    with open(os.path.join("app\\ingestion\\audio_ingess", "transcript.txt"), "w") as f:
        f.write(text_transcript)

# Example execution:
text_transcript = transcribe_audio("/content/video_data/extracted_audio.wav")