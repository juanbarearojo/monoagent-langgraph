# agent/nodes/species_retrieve.py
from __future__ import annotations
from typing import Any, Dict, List
from agent.tools.rag_index import retrieve_species

def species_retrieve(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Devuelve SOLO updates:
      - rag_docs: List[str]
      - rag_meta: List[dict]
    """
    latin = (
        state.get("current_taxon")
        or state.get("_tmp", {}).get("latin_name")
        or ""
    ).strip()

    k = int(state.get("topk", 3) or 3)

    if not latin:
        print("[species_retrieve] sin taxón → 0 docs")
        return {"rag_docs": [], "rag_meta": []}

    docs = retrieve_species(latin, top_k_sections=k)
    rag_docs = [d.get("text", "") for d in docs]
    print(f"[species_retrieve] latin={latin!r} -> {len(docs)} docs")

    return {"rag_docs": rag_docs, "rag_meta": docs}
