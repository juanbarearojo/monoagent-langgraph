# agent/nodes/finalize.py
from __future__ import annotations
from typing import Any, Dict
from langchain_core.messages import AIMessage

from agent.prompts import PROMPT_FINALIZE
from agent.tools.gpt import ask_gpt_text
from agent.utils.text import truncate


def finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    latin = state.get("_tmp", {}).get("latin_name") or state.get("current_taxon") or "—"
    summary = state.get("_tmp", {}).get("summary_snippet") or state.get("context_md", "")
    sources = state.get("_tmp", {}).get("source_attribs") or state.get("sources", [])
    p1 = state.get("_tmp", {}).get("p1")
    entropy = state.get("_tmp", {}).get("entropy")

    # Armar bloque fuentes
    bullet_sources = ""
    if sources:
        bullet_sources = "\n".join(
            f"- [{s['title']}]({s['url']})" for s in sources if s.get("url")
        )

    # Transparencia (opcional, dev info)
    transparency = ""
    if p1 is not None and entropy is not None:
        transparency = f"\n\n(Confianza local: {p1:.2f} · Entropía: {entropy:.2f})"

    # Prompt a GPT con contexto truncado
    context = truncate(summary, max_chars=4000)
    prompt = PROMPT_FINALIZE.format(
        latin=latin,
        context=context + ("\n\n" + bullet_sources if bullet_sources else "")
    )

    result = ask_gpt_text(prompt)
    answer = result.get("answer", "").strip() if isinstance(result, dict) else str(result)

    if not answer:
        # fallback simple si GPT falla
        answer = f"**Especie identificada:** *{latin}*\n\n{summary}{transparency}"

    msg = answer + (transparency if transparency else "")

    # Limpiar efímeros
    cleaned: Dict[str, Any] = {k: v for k, v in state.items() if not str(k).startswith("_tmp")}
    messages = list(cleaned.get("messages", [])) + [AIMessage(content=msg)]
    cleaned["messages"] = messages
    cleaned["current_taxon"] = latin
    cleaned["sources"] = sources

    return cleaned
