# agent/nodes/capture_user_taxon.py
from __future__ import annotations
from typing import Any, Dict

def capture_user_taxon(state: Dict[str, Any]) -> Dict[str, Any]:
    user_binomial = state.get("_tmp", {}).get("user_binomial")
    if not user_binomial:
        return state

    return {
        **state,
        "current_taxon": user_binomial,
        "current_taxon_source": "user",
        "_tmp": {**state.get("_tmp", {}), "captured_user_taxon": True},
    }
