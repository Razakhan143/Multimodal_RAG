import speech_recognition as sr
from open_clip import create_model_and_transforms, get_tokenizer
import torch
from PIL import Image
class AudioTranscriber:
    def __init__(self):
        pass
    def transcribe_audio(self, audio_path):
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
            

class EmbeddingsGenerator:
    tokenizer = get_tokenizer("ViT-B-32")
    # Load OpenCLIP model
    model, _, preprocess = create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

    tokenizer = get_tokenizer("ViT-B-32")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    def __init__(self):
        pass
    # Text embedding function
    def embed_text(self, text):
        with torch.no_grad():
            tokens = self.tokenizer([text]).to(torch.device)
            embedding = self.model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0].tolist()
    
    def embed_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(torch.device)

        with torch.no_grad():
            embedding = self.model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)

        return embedding.cpu().numpy()[0].tolist()
    
import base64
import io
from PIL import Image

class encoding:
    def __init__(self):
        pass
    def encode_image_to_base64(image_path, resize=(512, 512)):
        with Image.open(image_path) as img:
            img.thumbnail(resize)
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=80)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        