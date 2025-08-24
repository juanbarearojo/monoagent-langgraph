# agent/nodes/router.py
from __future__ import annotations
from typing import Any, Dict
from agent.state import ChatVisionState, has_image  # type: ignore


def router_input(state: ChatVisionState) -> Dict[str, Any]:
    """
    Router mínimo:
      - Si se define state['force_route'] en {"classify","qa","ask_image"}, se respeta.
      - Si hay imagen (image_bytes | image_url) -> "classify"
      - Si no hay imagen pero hay current_taxon -> "qa"
      - En cualquier otro caso -> "ask_image"

    Lee:
      image_bytes | image_url | current_taxon | (opcional) force_route

    Devuelve:
      {"_tmp.route": "classify" | "qa" | "ask_image"}
    """
    forced = state.get("force_route")  # type: ignore
    if forced in ("classify", "qa", "ask_image"):
        return {"_tmp.route": forced}

    if has_image(state):
        route = "classify"
    elif state.get("current_taxon"):
        route = "qa"
    else:
        route = "ask_image"

    return {"_tmp.route": route}
