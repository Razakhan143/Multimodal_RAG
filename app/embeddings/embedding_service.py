import torch
from PIL import Image
from app.embeddings.models import vis_models


class EmbeddingsGenerator:
    """Dual embedder for robust multimodal retrieval.

    Two embedding spaces are used because they serve different jobs:

    * **Text-to-text retrieval** (documents, audio + video transcripts) uses a
      sentence-transformer (``all-MiniLM-L6-v2``, 384-dim). CLIP's text encoder
      truncates at 77 tokens and is tuned for short captions, which made long
      passages effectively invisible to search — the root cause of weak recall.
      MiniLM embeds full passages and is purpose-built for semantic search.

    * **Text-to-image retrieval** (video frames) keeps CLIP (ViT-B-32, 512-dim):
      the query text and the frame images must live in CLIP's joint space.

    Both models are heavy, so each is loaded lazily and shared across every
    instance via a class-level cache. Constructing this class stays cheap.
    """

    _clip_shared = None   # (model, preprocess, tokenizer, device)
    _text_shared = None   # SentenceTransformer

    def __init__(self):
        # No heavy work here; models materialise on first use.
        pass

    # ── CLIP (frame images + cross-modal query) ─────────────────────────────
    def _ensure_clip(self):
        if EmbeddingsGenerator._clip_shared is None:
            model, preprocess, tokenizer = vis_models().clip_model()
            device = next(model.parameters()).device
            EmbeddingsGenerator._clip_shared = (model, preprocess, tokenizer, device)
        return EmbeddingsGenerator._clip_shared

    # ── Sentence-transformer (text-to-text) ─────────────────────────────────
    def _ensure_text(self):
        if EmbeddingsGenerator._text_shared is None:
            from sentence_transformers import SentenceTransformer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            EmbeddingsGenerator._text_shared = SentenceTransformer(
                "all-MiniLM-L6-v2", device=device
            )
        return EmbeddingsGenerator._text_shared

    # ── Public API ──────────────────────────────────────────────────────────
    def embed_text(self, text):
        """Semantic text embedding for text-to-text retrieval (384-dim)."""
        model = self._ensure_text()
        emb = model.encode([text], normalize_embeddings=True)[0]
        return emb.tolist()

    def embed_texts(self, texts):
        """Batch embed many passages at once — far faster than a Python loop."""
        model = self._ensure_text()
        embs = model.encode(list(texts), normalize_embeddings=True, batch_size=32)
        return [e.tolist() for e in embs]

    def embed_text_clip(self, text):
        """CLIP text embedding for querying the frame collection (512-dim)."""
        model, _preprocess, tokenizer, device = self._ensure_clip()
        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            embedding = model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0].tolist()

    def embed_image(self, image_path):
        """CLIP image embedding for storing video frames (512-dim)."""
        model, preprocess, _tokenizer, device = self._ensure_clip()
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0].tolist()
