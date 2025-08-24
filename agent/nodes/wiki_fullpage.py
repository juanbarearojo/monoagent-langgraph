# agent/nodes/wiki_fullpage.py
from __future__ import annotations
from typing import Any, Dict

from agent.tools.wiki import fetch_fullpage  # tu tool síncrona

def fetch_wikipedia_fullpage(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recupera la página completa de Wikipedia para el taxón actual.

    Entrada esperada:
      - state["_tmp"]["latin_name"]  (preferente)
      - o state["current_taxon"]

    Salida (solo deltas):
      - "wiki": Dict[str, Any]            # página completa (title, url, plain_text, infobox, status, ...)
      - "_tmp": { "wiki_status": str, "wiki_error": str? }
    """
    tmp_in = state.get("_tmp", {}) or {}

    # 1) Determinar el taxón objetivo (preferir el latín mapeado)
    latin = tmp_in.get("latin_name") or state.get("current_taxon")
    if not latin:
        # No devolvemos state entero; solo deltas
        out_tmp = {**tmp_in, "wiki_status": "not_found"}
        return {"_tmp": out_tmp}

    # 2) Llamar a la tool con control de errores
    try:
        page: Dict[str, Any] = fetch_fullpage(latin) or {}
        status = page.get("status", "error")
    except Exception as e:
        # No re‑emitir el estado; solo claves nuevas
        out_tmp = {**tmp_in, "wiki_status": "error", "wiki_error": f"{type(e).__name__}: {e}"}
        # Útil para logs locales
        print(f"[wiki_fullpage] ERROR fetching '{latin}': {e}")
        return {"_tmp": out_tmp}

    # 3) Preparar salida (solo deltas)
    out_tmp = {**tmp_in, "wiki_status": status}
    # 👇 DEVOLVEMOS 'wiki' (una sola clave nueva), sin tocar nada más
    return {
        "wiki": page,
        "_tmp": out_tmp,
    }
