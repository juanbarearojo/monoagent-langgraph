# agent/nodes/ask_gpt41_vision.py
from __future__ import annotations
from typing import Any, Dict

try:
    from agent.state import has_image, valid_binomial
except Exception:
    import re
    _BIN = re.compile(r"^[A-Z][a-z]+ [a-z]+$")
    def has_image(state: Dict[str, Any]) -> bool:
        return bool(state.get("image_bytes") or state.get("image") or state.get("image_url"))
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
from agent.prompts import PROMPT_BINOMIAL  # por si quieres inyectar prompt

def ask_gpt41_vision(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invoca modelo multimodal real (OpenAI Responses API) para extraer el binomial.
    Entradas (cualquiera):
      - state['image_bytes']  (bytes)
      - state['image']        (PIL.Image)
      - state['image_url']    (str)
    Salidas:
      - state['_tmp.vision_status'] = ok|invalid|empty|error
      - state['_tmp.latin_name']    = str (si ok)
    """
    if not has_image(state):
        state["_tmp.vision_status"] = "empty"
        return state

    image_bytes = state.get("image_bytes")
    image_url = state.get("image_url")
    if not image_bytes:
        pil = state.get("image")
        if pil is not None:
            try:
                image_bytes = _image_to_bytes(pil)
            except Exception:
                image_bytes = None

    payload = _ask_binomial(
        image_bytes=image_bytes,
        url=image_url,
        prompt=PROMPT_BINOMIAL,
        model=state.get("vision_model", "gpt-4.1-mini"),
        max_output_tokens=int(state.get("vision_max_output_tokens", 16)),
    )

    status = payload.get("status", "error")
    latin = payload.get("latin_name", "")

    state["_tmp.vision_status"] = status
    if status == "ok" and valid_binomial(latin):
        state["_tmp.latin_name"] = latin

    return state
