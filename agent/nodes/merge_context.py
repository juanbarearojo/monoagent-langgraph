# agent/nodes/merge_context.py
from typing import Any, Dict, List

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
    for i, (k, v) in enumerate(infobox.items()):
        if i >= max_items:
            break
        lines.append(f"- {k}: {v}")
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
        chunk = " ".join(x for x in [title, "—", snippet, url] if x)
        if chunk:
            parts.append(chunk)
    return "\n".join(parts).strip()

def _collect_sources(wiki: Dict[str, Any], ddg: Dict[str, Any], max_sources: int = 10) -> List[str]:
    urls: List[str] = []
    wurl = wiki.get("url")
    if wurl:
        urls.append(wurl)
    results = (ddg or {}).get("results") or []
    for r in results:
        u = r.get("url") or r.get("link")
        if u:
            urls.append(u)
    seen, uniq = set(), []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)
        if len(uniq) >= max_sources:
            break
    return uniq

def merge_context(state: Dict[str, Any]) -> Dict[str, Any]:
    tmp_in = state.get("_tmp", {}) or {}

    # ¿Están listas las dos ramas?
    wiki_status = tmp_in.get("wiki_status")
    ddg_status  = tmp_in.get("ddg_status")

    fanout_ready = bool(wiki_status is not None and ddg_status is not None)

    wiki = state.get("wiki") or {}
    ddg  = tmp_in.get("ddg") or state.get("ddg") or {}

    # Construir (parcial o completo)
    title = wiki.get("title") or state.get("current_taxon") or ""
    url   = wiki.get("url") or ""
    plain = wiki.get("plain_text") or ""
    infobox_md = _format_infobox(wiki.get("infobox") or {})
    ddg_text   = _format_ddg(ddg)

    parts: List[str] = []
    if title or url:
        parts.append(" | ".join(x for x in [title, url] if x))
    if infobox_md:
        parts.append("<INFOBOX>\n" + infobox_md + "\n</INFOBOX>")
    if plain:
        parts.append("<WIKIPEDIA>\n" + plain + "\n</WIKIPEDIA>")
    if ddg_text:
        parts.append("<WEB_SNIPPETS>\n" + ddg_text + "\n</WEB_SNIPPETS>")

    context = _truncate("\n\n".join(p for p in parts if p).strip(),
                        int(state.get("context_max_chars", 8000)))

    out_tmp = {
        **tmp_in,
        "context": context,
        "context_status": "ok" if context else "empty",
        "sources": _collect_sources(wiki, ddg),
        "fanout_ready": fanout_ready,          # 👈 bandera de “ya llegaron wiki y ddg”
    }
    return {"_tmp": out_tmp}
