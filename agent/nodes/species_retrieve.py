# agent/nodes/species_retrieve.py
from __future__ import annotations
from typing import Any, Dict, List
from agent.tools.rag_index import retrieve_species

def species_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Nodo LangGraph:
      - Toma el taxón actual del estado
      - Recupera hasta 'topk' trozos de RAG para esa especie
      - Escribe:
          state["rag_docs"]: List[str]
          state["rag_meta"]: List[dict]
    """
    latin = (
        state.get("current_taxon")
        or state.get("_tmp", {}).get("latin_name")
        or ""
    ).strip()

    k = int(state.get("topk", 3)) or 3

    if not latin:
        state["rag_docs"] = []
        state["rag_meta"] = []
        return state

    docs = retrieve_species(latin, top_k_sections=k)
    state["rag_docs"] = [d.get("text","") for d in docs]
    # guarda metadata en raíz (no mezclar con _tmp para que no se limpie)
    state["rag_meta"] = docs

    # DEBUG visible en consola
    print(f"[species_retrieve] latin={latin!r} -> {len(docs)} docs")

    return state
