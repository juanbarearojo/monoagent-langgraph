# agent/nodes/ensure_image.py
from __future__ import annotations
from typing import Dict, Any
from PIL import Image
import io, os

from agent.state import ChatVisionState

def ensure_image(state: ChatVisionState) -> Dict[str, Any]:
    pil = None

    # bytes directos
    img_bytes = state.get("image_bytes")
    if isinstance(img_bytes, (bytes, bytearray)) and len(img_bytes) > 0:
        try:
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            pil = None

    # ruta temporal (por si la usas)
    if pil is None:
        img_path = state.get("_tmp", {}).get("image_path")
        if img_path and os.path.exists(img_path):
            try:
                pil = Image.open(img_path).convert("RGB")
            except Exception:
                pil = None

    # URL no la abrimos aquí para no hacer IO; si la usas, ya tendrás bytes.

    if pil is None:
        # Solo devolvemos mensajes y _tmp. NO tocamos image_bytes.
        return {
            "messages": [{
                "type": "ai",
                "content": (
                    "Para continuar necesito **una imagen válida del primate**.\n\n"
                    "Sube una JPG/PNG directa (ideal < 5 MB)."
                )
            }],
            "_tmp": {**state.get("_tmp", {}), "image_ok": False}
        }

    return {
        # Guardamos el PIL para los siguientes nodos
        "pil_image": pil,
        "_tmp": {**state.get("_tmp", {}), "image_ok": True},
    }
