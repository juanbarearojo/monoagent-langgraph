# agent/nodes/wiki_fullpage.py
from typing import Any, Dict
from agent.tools.wiki import fetch_fullpage

def fetch_wikipedia_fullpage(state) -> Dict[str, Any]:
    """
    NODO: Recupera página completa de Wikipedia para el taxón actual.
    Usa state["_tmp.latin_name"] si existe; si no, state["current_taxon"].
    Inyecta en el estado: state["wiki"] = {...}, y state["_tmp.wiki_status"].
    """
    latin = state.get("_tmp.latin_name") or state.get("current_taxon")
    if not latin:
        state["_tmp.wiki_status"] = "not_found"
        return state

    page = fetch_fullpage(latin)
    state["wiki"] = page
    state["_tmp.wiki_status"] = page.get("status", "error")
    return state
