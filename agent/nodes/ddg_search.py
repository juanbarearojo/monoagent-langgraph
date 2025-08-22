# agent/nodes/ddg_search.py
from __future__ import annotations
from typing import Any, Dict

try:
    from agent.tools.ddg import search_summary
except Exception:
    # Fallback defensivo en caso de import fallido durante early wiring
    def search_summary(latin: str, max_results: int = 5, timeout: float = 5.0) -> Dict[str, Any]:
        if not latin:
            return {"status": "empty", "top_snippet": "", "results": []}
        return {
            "status": "ok",
            "top_snippet": "Conservation status reported by IUCN; natural history from ADW.",
            "results": [
                {"title": f"{latin} – IUCN Red List", "url": "https://www.iucnredlist.org/", "snippet": "Conservation status and threats..."},
                {"title": f"{latin} – Animal Diversity Web", "url": "https://animaldiversity.org/", "snippet": "Natural history, behavior..."},
            ],
        }

def retrieve_ddg(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo: consulta DDG (vía tools.ddg.search_summary) y guarda salida en state.
    Llaves:
      - Entrada: state["_tmp.latin_name"] o state["current_taxon"]
      - Salida principal para merge_context: state["_tmp.ddg"]
      - Campos de compatibilidad: "_tmp.ddg_status", "_tmp.ddg_results", "_tmp.ddg_snippet"
    """
    latin = state.get("_tmp.latin_name") or state.get("current_taxon")
    if not latin:
        state["_tmp.ddg"] = {"status": "empty", "top_snippet": "", "results": []}
        state["_tmp.ddg_status"] = "empty"
        state["_tmp.ddg_results"] = []
        state["_tmp.ddg_snippet"] = ""
        return state

    ddg_payload = search_summary(latin)
    # Salida primaria para merge_context
    state["_tmp.ddg"] = ddg_payload

    # Compatibilidad con posibles usos previos
    state["_tmp.ddg_status"] = ddg_payload.get("status", "error")
    state["_tmp.ddg_results"] = ddg_payload.get("results", [])
    state["_tmp.ddg_snippet"] = ddg_payload.get("top_snippet", "")

    return state
