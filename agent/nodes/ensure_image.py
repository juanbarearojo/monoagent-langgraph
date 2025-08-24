# agent/nodes/ensure_image.py
from __future__ import annotations
from typing import Any, Dict, Optional
from agent.state import ChatVisionState  # type: ignore
from agent.utils.images import download_to_bytes as _download_to_bytes
from agent.utils.images import image_to_bytes as _image_to_bytes



def ensure_image(state: ChatVisionState) -> Dict[str, Any]:
    """
    Normaliza a `image_bytes` para que tanto la inferencia local como ask_gpt41_vision
    tengan una entrada estable.

    Entradas posibles (cualquiera):
      - state['image_bytes']  (bytes ya presentes)
      - state['image']        (PIL.Image)
      - state['image_url']    (str)

    Salida:
      - {"image_bytes": <bytes>, "_tmp.image_ok": True}  si se ha podido normalizar
      - {"_tmp.image_ok": False, "_tmp.image_error": "..."} en caso contrario

    NOTA: Devuelve **un dict parcial** (no muta el state original) para que el
    framework del grafo mezcle las claves en el estado global.
    """
    # 1) Ya hay bytes
    b = state.get("image_bytes")
    if isinstance(b, (bytes, bytearray)) and len(b) > 0:
        return {"_tmp.image_ok": True}

    # 2) PIL.Image -> bytes
    pil = state.get("image")
    if pil is not None:
        try:
            bytes_data = _image_to_bytes(pil)
            if bytes_data:
                return {"image_bytes": bytes_data, "_tmp.image_ok": True}
        except Exception as e:
            return {"_tmp.image_ok": False, "_tmp.image_error": f"image_to_bytes_failed: {e}"}

    # 3) URL -> bytes
    url = state.get("image_url")
    if url:
        try:
            bytes_data = _download_to_bytes(url)
            if bytes_data:
                return {"image_bytes": bytes_data, "_tmp.image_ok": True}
            return {"_tmp.image_ok": False, "_tmp.image_error": "download_empty"}
        except Exception as e:
            return {"_tmp.image_ok": False, "_tmp.image_error": f"download_failed: {e}"}

    # 4) No hay ninguna representación válida
    return {"_tmp.image_ok": False}
