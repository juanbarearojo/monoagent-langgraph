from typing import TypedDict, Annotated, Optional, Sequence, Literal, Any, Dict, List, Tuple
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

class ChatVisionState(TypedDict,total=False):
    message: Annotated[Sequence[AnyMessage], add_messages]
    image_bytes: Optional[bytes]
    image_url: Optional[str]
    topk: int
    accept_policy: Literal["entropy","confidence","margin"]
    accept_threshold: float
    current_taxon: Optional[str]

def has_image(state: ChatVisionState) -> bool:
    return bool(state.get("image_bytes") or state.get("image_url"))

def valid_binomial(s:str) -> bool:
    parts = s.strip().split()
    if len(parts) != 2: return False
    g, e = parts
    return g[:1].isupper() and g[1:].islower() and e.islower()
