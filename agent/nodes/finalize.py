# agent/nodes/finalize.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
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
        # Fallback razonable si no hubiera url
        q = (title or latin).replace(" ", "_")
        sources.append({"title": title or latin or "Wikipedia", "url": f"https://en.wikipedia.org/wiki/{q}"})

    # DEBUG
    print(f"[finalize] wiki.title={title!r} wiki.url={url!r} plain_len={len(plain)}")

    return {"context_md": context_md, "sources": sources}


def _build_context_from_rag(
    state: Dict[str, Any],
    max_docs: int = 5,
    max_chars_per_doc: int = 1200
) -> Dict[str, Any]:
    """
    Construye:
      - rag_md: bloques <RAGDOC> con snippets truncados
      - rag_sources: [{'title','url'}]-like; aquí usamos 'file' como 'title' (sin URL)
    """
    docs = state.get("rag_docs") or []
    meta = state.get("rag_meta") or []
    if not docs or not meta:
        return {"rag_md": "", "rag_sources": []}

    blocks: List[str] = []
    rag_sources: List[Dict[str, str]] = []
    used_files = set()

    for i, (txt, m) in enumerate(zip(docs[:max_docs], meta[:max_docs]), start=1):
        sec = m.get("section") or "Body"
        latin = m.get("latin_name") or ""
        year = m.get("year")
        file = m.get("file") or "local-index"

        # Fuente local sin URL
        if file not in used_files:
            label = f"{file} ({sec}{', '+str(year) if year else ''})"
            rag_sources.append({"title": label, "url": ""})
            used_files.add(file)

        snippet = (txt or "").strip()
        if len(snippet) > max_chars_per_doc:
            snippet = snippet[:max_chars_per_doc].rsplit(" ", 1)[0] + "…"

        blocks.append(
            f'<RAGDOC n="{i}" section="{sec}" latin="{latin}" file="{file}" year="{year or ""}">\n'
            f"{snippet}\n"
            f"</RAGDOC>"
        )

    rag_md = "<RAG>\n" + "\n\n".join(blocks) + "\n</RAG>"
    return {"rag_md": rag_md, "rag_sources": rag_sources}


def _format_sources_bullets(sources: List[Dict[str, str]]) -> str:
    """
    Devuelve bullets Markdown. Si hay URL → enlace; si no, texto plano.
    """
    if not sources:
        return ""
    lines: List[str] = []
    for s in sources:
        title = s.get("title") or "Fuente"
        url = s.get("url") or ""
        if url:
            lines.append(f"- [{title}]({url})")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines)


def finalize_answer(state: Dict[str, Any]) -> Dict[str, Any]:
    # DEBUG wiki
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

    # ---- Contexto desde WIKI
    wiki_ctx = _build_context_from_wiki(state, latin)
    wiki_md = wiki_ctx["context_md"]
    wiki_sources = wiki_ctx["sources"]

    # ---- Contexto desde RAG (si existe)
    rag_ctx = _build_context_from_rag(state, max_docs=int(state.get("topk", 3) or 3))
    rag_md = rag_ctx["rag_md"]
    rag_sources = rag_ctx["rag_sources"]

    # ---- Combinar contextos y fuentes
    combined_context = "\n\n".join([p for p in [wiki_md, rag_md] if p]).strip()
    sources: List[Dict[str, str]] = wiki_sources + rag_sources

    # Transparencia del clasificador local si existe
    p1 = state.get("_tmp", {}).get("p1")
    entropy = state.get("_tmp", {}).get("entropy")
    transparency = ""
    if p1 is not None and entropy is not None:
        transparency = f"\n\n(Confianza local: {p1:.2f} · Entropía: {entropy:.2f})"

    # Formateo de fuentes (Markdown)
    bullet_sources = _format_sources_bullets(sources)

    # Prompt final a GPT (contexto truncado)
    prompt_context = _truncate(
        combined_context + ("\n\n" + bullet_sources if bullet_sources else ""),
        max_chars=int(state.get("context_max_chars", 8000)),
    )
    prompt = PROMPT_FINALIZE.format(latin=latin, context=prompt_context)

    # DEBUG
    print(f"[finalize] latin={latin!r} ctx_len={len(prompt_context)} sources={len(sources)} "
          f"rag_present={bool(rag_md)}")

    result = ask_gpt_text(prompt)
    answer = result.get("answer", "").strip() if isinstance(result, dict) else str(result)

    if not answer:
        # Fallback simple si GPT no contesta
        answer = f"**Especie identificada:** *{latin}*\n\n{combined_context}"

    msg = answer + (transparency if transparency else "")

    # ✅ Devolver SOLO updates (no el estado completo)
    updates: Dict[str, Any] = {
        "messages": [AIMessage(content=msg)],
        "current_taxon": latin,
        "sources": sources,   # para interfaz/telemetría
        "_tmp": {},           # limpia efímeros
    }
    return updates
