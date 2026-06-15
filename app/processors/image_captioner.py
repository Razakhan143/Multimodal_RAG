import base64
import io
from PIL import Image

def encode_image_to_base64(image_path, resize=(512, 512)):
    with Image.open(image_path) as img:
        img.thumbnail(resize)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")