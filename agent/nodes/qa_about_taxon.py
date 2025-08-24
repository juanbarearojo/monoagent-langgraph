# agent/nodes/qa_about_taxon.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage

from agent.prompts import PROMPT_QA_TAXON  # prompt bilingüe
from agent.utils.text import truncate
from agent.tools.gpt import ask_gpt_text


def last_user_utterance(state: Dict[str, Any]) -> str:
    """Extrae el último mensaje humano del historial."""
    msgs = state.get("messages", [])
    for m in reversed(msgs):
        if getattr(m, "type", "") == "human":
            return m.content.strip()
    return ""


def qa_about_taxon(state: Dict[str, Any]) -> Dict[str, Any]:
    latin = state.get("current_taxon")
    if not latin:
        msg = (
            "Necesito primero una imagen para identificar la especie antes de responder preguntas."
        )
        return {
            **state,
            "messages": list(state.get("messages", []))
            + [AIMessage(content=msg)],
        }

    question = last_user_utterance(state)
    context = state.get("context_md", "")
    context = truncate(context, max_chars=4000)

    # Construir prompt completo
    prompt = PROMPT_QA_TAXON.format(
        latin=latin,
        context=context,
        question=question,
    )

    # Llamada al modelo textual
    result = ask_gpt_text(prompt)
    answer = (
        result.get("answer")
        if isinstance(result, dict)
        else str(result)
    )

    return {
        **state,
        "messages": list(state.get("messages", [])) + [AIMessage(content=answer)],
        "_tmp": {**state.get("_tmp", {}), "qa_answered": True},
    }
