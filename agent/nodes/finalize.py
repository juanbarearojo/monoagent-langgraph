# agent/nodes/finalize.py
from typing import Any, Dict
from langchain_core.messages import AIMessage

def finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """Stub temporal mientras se implementa el nodo real."""
    msg = AIMessage(content="(finalize_answer aún no implementado)")
    return {**state, "messages": state.get("messages", []) + [msg]}
