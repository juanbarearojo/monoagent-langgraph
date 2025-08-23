# agent/nodes/merge_context.py
from typing import Any, Dict, List

# Usa truncate si está disponible; si no, fallback local
try:
    from agent.utils.text import truncate as _truncate
except Exception:
    def _truncate(text: str, max_chars: int) -> str:
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _format_infobox(infobox: Dict[str, str], max_items: int = 12) -> str:
    if not infobox:
        return ""
    lines: List[str] = []
    count = 0
    for k, v in infobox.items():
        lines.append(f"- {k}: {v}")
        count += 1
        if count >= max_items:
            break
    return "\n".join(lines).strip()


def _format_ddg(ddg: Dict[str, Any], max_results: int = 3) -> str:
    if not ddg or ddg.get("status") not in {"ok", "success"}:
        return ""
    parts: List[str] = []
    top = ddg.get("top_snippet")
    if top:
        parts.append(top)
    results = ddg.get("results") or []
    for r in results[:max_results]:
        title = r.get("title") or ""
        snippet = r.get("snippet") or r.get("summary") or ""
        url = r.get("url") or r.get("link") or ""
        chunk = " — ".join([x for x in [title, snippet, url] if x])
        if chunk:
            parts.append(chunk)
    return "\n".join(parts).strip()


def _collect_sources(wiki: Dict[str, Any], ddg: Dict[str, Any], max_sources: int = 10) -> List[str]:
    """
    Devuelve lista de URLs (Wikipedia + resultados DDG) sin duplicados, orden estable.
    """
    urls: List[str] = []

    # Wikipedia
    wurl = wiki.get("url")
    if wurl:
        urls.append(wurl)

    # DDG (si existe)
    results = (ddg or {}).get("results") or []
    for r in results:
        u = r.get("url") or r.get("link")
        if u:
            urls.append(u)

    # Deduplicar preservando orden
    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
        if len(uniq) >= max_sources:
            break
    return uniq


def merge_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Une el contexto de Wikipedia (plain_text + infobox) y, si existe, snippets web (DDG).
    Guarda el resultado en state["_tmp.context"] como un único bloque listo para el LLM.
    Respeta state.get("context_max_chars", 8000) como límite de longitud.
    Además, recopila fuentes en state["_tmp.sources"] (Wikipedia + DDG).
    """
    wiki = state.get("wiki") or {}
    ddg = state.get("_tmp.ddg") or state.get("ddg") or {}

    title = wiki.get("title") or state.get("current_taxon") or ""
    url = wiki.get("url") or ""
    plain = wiki.get("plain_text") or ""
    infobox = wiki.get("infobox") or {}

    infobox_md = _format_infobox(infobox)
    ddg_text = _format_ddg(ddg)

    # Componemos un único bloque de contexto sin encabezados de Wikipedia.
    # Estos separadores son para el LLM, no se muestran al usuario.
    parts: List[str] = []
    if title or url:
        meta_line = " | ".join(x for x in [title, url] if x)
        parts.append(meta_line)
    if infobox_md:
        parts.append("<INFOBOX>\n" + infobox_md + "\n</INFOBOX>")
    if plain:
        parts.append("<WIKIPEDIA>\n" + plain + "\n</WIKIPEDIA>")
    if ddg_text:
        parts.append("<WEB_SNIPPETS>\n" + ddg_text + "\n</WEB_SNIPPETS>")

    context = "\n\n".join(p for p in parts if p).strip()

    max_chars = int(state.get("context_max_chars", 8000))
    context = _truncate(context, max_chars)

    # Salida principal
    state["_tmp.context"] = context
    state["_tmp.context_status"] = "ok" if context else "empty"

    # Fuentes (Wikipedia + DDG)
    state["_tmp.sources"] = _collect_sources(wiki, ddg)

    return state
