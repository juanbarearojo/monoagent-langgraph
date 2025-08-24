# agent/state.py
from __future__ import annotations
from typing import TypedDict, Annotated, Optional, Sequence, Literal, Any, Dict
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage

class ChatVisionState(TypedDict, total=False):
    # MENSAJES (¡en plural!)
    messages: Annotated[Sequence[AnyMessage], add_messages]

    # ENTRADAS DE IMAGEN
    image_bytes: Optional[bytes]
    image_url: Optional[str]

    # OPCIONALES / PARÁMETROS
    topk: int
    accept_policy: Literal["entropy", "confidence", "margin"]
    accept_threshold: float
    current_taxon: Optional[str]

    # ⚠️ EPHEMERAL (debe existir para que LangGraph no lo “ignore”)
    _tmp: Dict[str, Any]

def has_image(state: ChatVisionState) -> bool:
    if isinstance(state.get("image_bytes"), (bytes, bytearray)) and state["image_bytes"]:
        return True
    if isinstance(state.get("_tmp", {}).get("image_bytes"), (bytes, bytearray)) and state["_tmp"]["image_bytes"]:
        return True
    if state.get("_tmp", {}).get("image_path"):
        return True
    if state.get("image_url"):
        return True
    return False


def valid_binomial(s: str) -> bool:
    parts = s.strip().split()
    if len(parts) != 2:
        return False
    g, e = parts
    return g[:1].isupper() and g[1:].islower() and e.islower()
