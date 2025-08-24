# agent/nodes/finalize.py
from typing import Any, Dict, List
from langchain_core.messages import AIMessage

from agent.prompts import PROMPT_FINALIZE
from agent.tools.gpt import ask_gpt_text

# Truncador fallback
try:
    from agent.utils.text import truncate as _truncate
except Exception:
    def _truncate(text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _nice_label(s: str) -> str:
    return s.replace("_", " ").strip().title() if s else s


def _format_infobox(infobox: Dict[str, Any], max_items: int = 12) -> str:
    if not infobox:
        return ""
    lines: List[str] = []
    for i, (k, v) in enumerate(infobox.items()):
        if i >= max_items:
            break
        lines.append(f"- {k}: {v}")
    return "\n".join(lines).strip()


def _build_context_from_wiki(state: Dict[str, Any], latin: str) -> Dict[str, Any]:
    """
    Construye:
      - context_md: texto markdown con title/url + infobox + plain_text (truncado)
      - sources: [{'title','url'}]
    """
    wiki: Dict[str, Any] = state.get("wiki") or {}

    title = wiki.get("title") or latin or ""
    url   = wiki.get("url") or ""
    plain = wiki.get("plain_text") or ""
    infobox_md = _format_infobox(wiki.get("infobox") or {})

    parts: List[str] = []
    if title or url:
        parts.append(" | ".join(x for x in [title, url] if x))
    if infobox_md:
        parts.append("<INFOBOX>\n" + infobox_md + "\n</INFOBOX>")
    if plain:
        parts.append("<WIKIPEDIA>\n" + plain + "\n</WIKIPEDIA>")

    context_md = "\n\n".join(p for p in parts if p).strip()
    context_md = _truncate(context_md, int(state.get("context_max_chars", 8000)))

    # Fuentes
    sources: List[Dict[str, str]] = []
    if url:
        sources.append({"title": title or latin or "Wikipedia", "url": url})
    else:
        # Fallback razonable si no hubiera url (no es tu caso, pero por si acaso)
        q = (title or latin).replace(" ", "_")
        sources.append({"title": title or latin or "Wikipedia", "url": f"https://en.wikipedia.org/wiki/{q}"})

    # DEBUG
    print(f"[finalize] wiki.title={title!r} wiki.url={url!r} plain_len={len(plain)}")

    return {"context_md": context_md, "sources": sources}


def finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    w = state.get("wiki")
    print(f"[finalize] has_wiki={isinstance(w, dict)} "
          f"title={w.get('title') if isinstance(w, dict) else None} "
          f"plain_len={(len(w.get('plain_text')) if isinstance(w, dict) and w.get('plain_text') else 0)}")

    # Latin / taxón garantizado (con varios fallbacks amables)
    latin = (
        state.get("_tmp", {}).get("latin_name")
        or state.get("current_taxon")
        or _nice_label(state.get("_tmp", {}).get("pred_label", ""))
        or "—"
    )

    # Construye contexto desde WIKI (ya no dependemos de merge_context)
    ctx = _build_context_from_wiki(state, latin)
    context_md = ctx["context_md"]
    sources: List[Dict[str, str]] = ctx["sources"]

    # Transparencia del clasificador local si existe
    p1 = state.get("_tmp", {}).get("p1")
    entropy = state.get("_tmp", {}).get("entropy")
    transparency = ""
    if p1 is not None and entropy is not None:
        transparency = f"\n\n(Confianza local: {p1:.2f} · Entropía: {entropy:.2f})"

    # Formateo de fuentes (Markdown)
    bullet_sources = ""
    if sources:
        bullet_sources = "\n".join(
            f"- [{s.get('title','Fuente')}]({s.get('url')})"
            for s in sources if s.get("url")
        )

    # Prompt final a GPT (contexto truncado a 4000)
    prompt_context = _truncate(
        context_md + ("\n\n" + bullet_sources if bullet_sources else ""),
        max_chars=8000,
    )
    prompt = PROMPT_FINALIZE.format(latin=latin, context=prompt_context)

    # DEBUG
    print(f"[finalize] latin={latin!r} ctx_len={len(prompt_context)} sources={len(sources)}")

    result = ask_gpt_text(prompt)
    answer = result.get("answer", "").strip() if isinstance(result, dict) else str(result)

    if not answer:
        # Fallback simple si GPT no contesta
        answer = f"**Especie identificada:** *{latin}*\n\n{context_md}"

    msg = answer + (transparency if transparency else "")

    # Cerrar conversación: limpiar efímeros y añadir mensaje
    cleaned: Dict[str, Any] = {k: v for k, v in state.items() if not str(k).startswith("_tmp")}
    messages = list(cleaned.get("messages", [])) + [AIMessage(content=msg)]
    cleaned["messages"] = messages
    cleaned["current_taxon"] = latin
    cleaned["sources"] = sources

    return cleaned
