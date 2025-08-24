# agent/nodes/ddg_search.py
from __future__ import annotations
from typing import Any, Dict
from agent.tools.ddg import search_summary  # (latin: str, max_results=5, timeout=5.0) -> Dict[str, Any]


def retrieve_ddg(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Busca info en la web (DDG) para el taxón.
    Entrada:
      - state["_tmp"]["latin_name"]  (preferente)
      - o state["current_taxon"]

    Salida (solo deltas):
      - "_tmp.ddg": payload bruto (status, top_snippet, results[])
      - "_tmp.ddg_status": "ok" | "empty" | "error"
      - "_tmp.ddg_error": str (opcional)
    """
    tmp_in = state.get("_tmp", {}) or {}
    latin = tmp_in.get("latin_name") or state.get("current_taxon")

    if not latin:
        return {
            "_tmp": {
                **tmp_in,
                "ddg": {"status": "empty", "top_snippet": "", "results": []},
                "ddg_status": "empty",
            }
        }

    try:
        ddg_payload: Dict[str, Any] = search_summary(latin)
        status = ddg_payload.get("status", "error")
        # Normaliza status
        if status not in {"ok", "success"}:
            status = "empty" if not (ddg_payload.get("results") or ddg_payload.get("top_snippet")) else "ok"
    except Exception as e:
        print(f"[ddg_search] ERROR '{latin}': {e}")
        return {
            "_tmp": {
                **tmp_in,
                "ddg_status": "error",
                "ddg_error": f"{type(e).__name__}: {e}",
            }
        }

    return {
        "_tmp": {
            **tmp_in,
            "ddg": ddg_payload,
            "ddg_status": status,
        }
    }
