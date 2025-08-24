# agent/nodes/router.py
from __future__ import annotations
from typing import Any, Dict, Optional
import re

from agent.state import ChatVisionState, has_image, valid_binomial  # type: ignore

_BINOMIAL_RE = re.compile(r"\b([A-Z][a-z]+)\s+([a-z]+)\b")

def _extract_user_binomial(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = _BINOMIAL_RE.search(text)
    if not m:
        return None
    candidate = f"{m.group(1)} {m.group(2)}"
    return candidate if valid_binomial(candidate) else None

def router_input(state: ChatVisionState) -> Dict[str, Any]:
    """
    Política:
      - Imagen obligatoria para iniciar.
      - Si force_route ∈ {"classify","qa","ask_image"} → se respeta.
      - Si hay imagen → "classify"
      - Si NO hay imagen → "ask_image"
    Además:
      - Detecta posible binomio del usuario en state['user_text'] y lo guarda en _tmp.user_binomial (validado).
    """
    forced = state.get("force_route")  # type: ignore
    if forced in ("classify", "qa", "ask_image"):
        route = forced
    else:
        route = "classify" if has_image(state) else "ask_image"

    user_text = state.get("user_text")  # type: ignore
    user_binomial = _extract_user_binomial(user_text)

    tmp = {**state.get("_tmp", {}), "route": route}
    if user_binomial:
        tmp["user_binomial"] = user_binomial

    return {"_tmp": tmp}
