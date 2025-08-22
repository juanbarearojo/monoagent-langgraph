# agent/tools/ddg.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import os

"""
Interfaz pública:
search_summary(latin) -> { "status": "ok|empty|error",
                           "top_snippet": str,
                           "results": [ {title, url, snippet} ] }

Notas:
- Por defecto NO hace red; devuelve resultados "seed" seguros.
- Si se define DDG_ENABLE_NET=1, intenta una búsqueda muy básica.
- El test unitario mocha _net_search para evitar red real.
"""

def _seed_results(latin: str) -> Dict[str, Any]:
    results = [
        {
            "title": f"{latin} – IUCN Red List",
            "url": "https://www.iucnredlist.org/",
            "snippet": "Conservation status and threats...",
        },
        {
            "title": f"{latin} – Animal Diversity Web",
            "url": "https://animaldiversity.org/",
            "snippet": "Natural history, distribution, behavior...",
        },
    ]
    return {
        "status": "ok",
        "top_snippet": "Conservation status reported by IUCN; natural history from ADW.",
        "results": results,
    }

def _net_search(query: str, max_results: int = 5, timeout: float = 5.0) -> List[Dict[str, str]]:
    """
    Placeholder de red (opcional). No dependemos de librerías externas.
    Por simplicidad, devolvemos vacío; los tests moquean esta función.
    Si activas red real, puedes implementar aquí scraping/cliente propio.
    """
    return []

def search_summary(latin: Optional[str], max_results: int = 5, timeout: float = 5.0) -> Dict[str, Any]:
    if not latin or not str(latin).strip():
        return {"status": "empty", "top_snippet": "", "results": []}

    # Si no se habilita explícitamente la red, devolvemos semillas deterministas
    if os.getenv("DDG_ENABLE_NET", "0") != "1":
        return _seed_results(latin)

    # Camino de red (tests lo moquean)
    try:
        query = f'{latin} site:iucnredlist.org OR site:animaldiversity.org'
        hits = _net_search(query, max_results=max_results, timeout=timeout) or []
        hits = hits[:max_results]
        if not hits:
            return _seed_results(latin)

        # Heurística simple del top_snippet
        top_snippet = ""
        for h in hits:
            u = (h.get("url") or "").lower()
            if "iucnredlist.org" in u:
                top_snippet = "Conservation status summarized from IUCN Red List."
                break
        if not top_snippet:
            top_snippet = "Top web snippets from trusted sources (IUCN/ADW)."

        return {"status": "ok", "top_snippet": top_snippet, "results": hits}
    except Exception:
        # Nunca reventamos el grafo por DDG
        return {"status": "error", "top_snippet": "", "results": []}
