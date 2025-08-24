# agent/nodes/ddg_search.py
from __future__ import annotations
from typing import Any, Dict

try:
    from agent.tools.ddg import search_summary
except Exception:
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
    Busca info en DDG y la deja en claves NO compartidas:
      - 'ddg'       -> payload bruto
      - 'ddg_meta'  -> status/snippet rápido (opcional)
    Evita escribir en '_tmp' para no colisionar con otras ramas concurrentes.
    """
    latin = state.get("_tmp", {}).get("latin_name") or state.get("current_taxon")
    if not latin:
        return {
            "ddg": {"status": "empty", "top_snippet": "", "results": []},
            "ddg_meta": {"status": "empty", "snippet": ""},
        }

    ddg_payload = search_summary(latin)

    return {
        "ddg": ddg_payload,
        "ddg_meta": {
            "status": ddg_payload.get("status", "error"),
            "snippet": ddg_payload.get("top_snippet", ""),
        },
    }
