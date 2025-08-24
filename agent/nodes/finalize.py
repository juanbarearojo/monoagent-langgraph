# agent/nodes/finalize.py
from __future__ import annotations
from typing import Any, Dict, List
from langchain_core.messages import AIMessage

from agent.prompts import PROMPT_FINALIZE
from agent.tools.gpt import ask_gpt_text

# fallback local si no existe util
try:
    from agent.utils.text import truncate as _truncate
except Exception:
    def _truncate(text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _format_infobox(infobox: Dict[str, Any], max_items: int = 12) -> str:
    if not infobox:
        return ""
    lines: List[str] = []
    for i, (k, v) in enumerate(infobox.items()):
        if i >= max_items:
            break
        lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()


def _build_context_from_wiki(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construye un bloque de contexto Markdown desde state['wiki'].
    Devuelve:
      - context_md: str
      - sources: List[Dict[str, str]]  (title, url)
    """
    wiki: Dict[str, Any] = state.get("wiki") or {}
    title = wiki.get("title") or state.get("current_taxon") or ""
    url   = wiki.get("url") or ""
    plain = wiki.get("plain_text") or ""
    infobox_md = _format_infobox(wiki.get("infobox") or {})

    parts: List[str] = []
    if title or url:
        meta = " | ".join(x for x in [title, url] if x)
        parts.append(meta)
    if infobox_md:
        parts.append("<INFOBOX>\n" + infobox_md + "\n</INFOBOX>")
    if plain:
        parts.append("<WIKIPEDIA>\n" + plain + "\n</WIKIPEDIA>")

    context_md = "\n\n".join(p for p in parts if p).strip()
    context_md = _truncate(context_md, int(state.get("context_max_chars", 8000)))

    sources = []
    if url:
        sources.append({"title": title or "Wikipedia", "url": url})

    return {"context_md": context_md, "sources": sources}


def finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Finaliza la respuesta para el usuario usando SOLO Wikipedia (sin merge_context).
    Espera que:
      - state['_tmp']['latin_name'] o state['current_taxon'] esté definido
      - state['wiki'] ya esté cargado por fetch_wikipedia_fullpage
    """
    latin = state.get("_tmp", {}).get("latin_name") or state.get("current_taxon") or "—"

    # Construir contexto y fuentes desde 'wiki'
    ctx = _build_context_from_wiki(state)
    context_md = ctx["context_md"]
    sources: List[Dict[str, str]] = ctx["sources"]

    # Transparencia del clasificador local si existe
    p1 = state.get("_tmp", {}).get("p1")
    entropy = state.get("_tmp", {}).get("entropy")
    transparency = ""
    if p1 is not None and entropy is not None:
        transparency = f"\n\n(Confianza local: {p1:.2f} · Entropía: {entropy:.2f})"

    # Formatear fuentes
    bullet_sources = ""
    if sources:
        bullet_sources = "\n".join(
            f"- [{s.get('title','Fuente')}]({s.get('url')})" for s in sources if s.get("url")
        )

    # Prompt final a GPT (con contexto + fuentes)
    prompt_context = _truncate(
        context_md + ("\n\n" + bullet_sources if bullet_sources else ""),
        max_chars=4000
    )
    prompt = PROMPT_FINALIZE.format(latin=latin, context=prompt_context)

    result = ask_gpt_text(prompt)
    answer = result.get("answer", "").strip() if isinstance(result, dict) else str(result)

    if not answer:
        # Fallback simple si GPT falla
        answer = f"**Especie identificada:** *{latin}*\n\n{context_md}"

    msg = answer + (transparency if transparency else "")

    # Cerrar conversación: limpiar efímeros y añadir mensaje
    cleaned: Dict[str, Any] = {k: v for k, v in state.items() if not str(k).startswith("_tmp")}
    messages = list(cleaned.get("messages", [])) + [AIMessage(content=msg)]
    cleaned["messages"] = messages
    cleaned["current_taxon"] = latin
    cleaned["sources"] = sources

    return cleaned
