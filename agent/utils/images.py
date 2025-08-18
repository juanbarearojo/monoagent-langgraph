import io
import requests
from PIL import Image

def download_to_bytes(url: str) -> bytes:
    """Descarga una imagen desde una URL y devuelve bytes."""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.content

def image_to_bytes(pil_img, format: str = "PNG") -> bytes:
    """Convierte un objeto PIL.Image en bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return buf.getvalue()

def is_valid_image_bytes(data: bytes) -> bool:
    """Verifica que los bytes corresponden a una imagen válida."""
    try:
        Image.open(io.BytesIO(data)).verify()
        return True
    except Exception:
        return False

def is_supported_format(pil_img) -> bool:
    """Comprueba si el formato de la imagen es compatible."""
    return pil_img.format in {"JPEG", "PNG", "WEBP"}
