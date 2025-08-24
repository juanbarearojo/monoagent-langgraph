# agent/nodes/prompt_for_image.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage
from agent.prompts import PROMPT_NEED_IMAGE, PROMPT_NEED_IMAGE_WITH_TAXON  # type: ignore

def prompt_for_image(state: Dict[str, Any]) -> Dict[str, Any]:
    taxon = state.get("current_taxon")
    if taxon:
        msg = PROMPT_NEED_IMAGE_WITH_TAXON.format(taxon=taxon)
    else:
        msg = PROMPT_NEED_IMAGE

    return {
        **state,
        "messages": state.get("messages", []) + [AIMessage(content=msg)],
        "_tmp": {**state.get("_tmp", {}), "prompted": "image_required"},
    }
