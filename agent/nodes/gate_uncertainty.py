# agent/nodes/gate_uncertainty.py
from __future__ import annotations
from typing import Dict, Any

from agent.state import ChatVisionState

def gate_uncertainty(state: ChatVisionState) -> Dict[str, Any]:
    preds = state.get("_tmp", {}).get("preds") or []
    accept = False
    if preds and preds[0][1] >= state.get("accept_threshold", 0.7):
        accept = True
    return {
        "_tmp": {**state.get("_tmp", {}), "gate": "ACCEPT" if accept else "REVIEW"}
    }
