import torch
from PIL import Image
from app.embeddings.models import vis_models


class EmbeddingsGenerator:
    """CLIP-based text/image embedder.

    The underlying CLIP model is heavy (~hundreds of MB) and slow to load, so
    it is loaded lazily and shared across every ``EmbeddingsGenerator`` instance.
    Constructing this class is therefore cheap; the model is only materialised
    the first time an embedding is actually requested.
    """

    # Shared (model, preprocess, tokenizer, device) tuple — loaded once, reused.
    _shared = None

    def __init__(self):
        # Intentionally does no heavy work; see _ensure_loaded().
        pass

    def _ensure_loaded(self):
        if EmbeddingsGenerator._shared is None:
            model, preprocess, tokenizer = vis_models().clip_model()
            device = next(model.parameters()).device
            EmbeddingsGenerator._shared = (model, preprocess, tokenizer, device)
        return EmbeddingsGenerator._shared

    # Text embedding function
    def embed_text(self, text):
        model, _preprocess, tokenizer, device = self._ensure_loaded()
        with torch.no_grad():
            tokens = tokenizer([text]).to(device)
            embedding = model.encode_text(tokens)
            embedding /= embedding.norm(dim=-1, keepdim=True)
            return embedding.cpu().numpy()[0].tolist()

    def embed_image(self, image_path):
        model, preprocess, _tokenizer, device = self._ensure_loaded()
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
            embedding /= embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy()[0].tolist()
