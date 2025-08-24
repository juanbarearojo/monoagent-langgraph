# agent/nodes/ask_gpt41_vision.py
from __future__ import annotations
from typing import Any, Dict

try:
    from agent.state import has_image, valid_binomial
except Exception:
    import re
    _BIN = re.compile(r"^[A-Z][a-z]+ [a-z]+$")
    def has_image(state: Dict[str, Any]) -> bool:
        return bool(
            state.get("image_bytes")
            or state.get("image")
            or state.get("image_url")
            or state.get("_tmp", {}).get("pil_image")
        )
    def valid_binomial(s: str) -> bool:
        return bool(s and _BIN.match(s))

try:
    from agent.utils.images import image_to_bytes as _image_to_bytes
except Exception:
    def _image_to_bytes(pil_img) -> bytes:
        from io import BytesIO
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

from agent.tools.gpt import ask_binomial as _ask_binomial
from agent.prompts import PROMPT_BINOMIAL


def ask_gpt41_vision(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Entradas opcionales:
      - state['image_bytes'] (bytes)
      - state['_tmp']['pil_image'] (PIL.Image)  ← preferible si viene de ensure_image
      - state['image'] (PIL.Image)
      - state['image_url'] (str)
    Salidas (solo delta):
      - _tmp.vision_status = ok|invalid|empty|error
      - _tmp.latin_name    = str (si ok y válido)
    """
    if not has_image(state):
        # Solo delta
        return {"_tmp": {"vision_status": "empty"}}

    # Preferimos la imagen ya normalizada por ensure_image
    pil = state.get("_tmp", {}).get("pil_image") or state.get("image")
    image_bytes = state.get("image_bytes")
    image_url = state.get("image_url")

    if image_bytes is None and pil is not None:
        try:
            image_bytes = _image_to_bytes(pil)
        except Exception:
            image_bytes = None

    try:
        payload = _ask_binomial(
            image_bytes=image_bytes,
            url=image_url,
            prompt=PROMPT_BINOMIAL,
            model=state.get("vision_model", "gpt-4.1-mini"),
            max_output_tokens=int(state.get("vision_max_output_tokens", 16)),
        )
    except Exception:
        return {"_tmp": {"vision_status": "error"}}

    status = payload.get("status", "error")
    latin = payload.get("latin_name", "")

    tmp_out: Dict[str, Any] = {"vision_status": status}
    if status == "ok" and valid_binomial(latin):
        tmp_out["latin_name"] = latin

    # Importante: solo devolvemos DELTA en _tmp
    return {"_tmp": tmp_out}
