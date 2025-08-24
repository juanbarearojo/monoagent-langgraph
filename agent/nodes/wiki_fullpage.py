# agent/nodes/wiki_fullpage.py
from __future__ import annotations
from typing import Any, Dict
from agent.tools.wiki import fetch_fullpage

def _normalize_binomial(name: str) -> str:
    if not name:
        return name
    name = name.replace("_", " ").strip()
    parts = name.split()
    if len(parts) == 1:
        return parts[0][:1].upper() + parts[0][1:].lower()
    genus = parts[0][:1].upper() + parts[0][1:].lower()
    species = parts[1].lower()
    tail = " ".join(parts[2:])
    return f"{genus} {species}" + (f" {tail}" if tail else "")

def fetch_wikipedia_fullpage(state: Dict[str, Any]) -> Dict[str, Any]:
    tmp_in = state.get("_tmp", {}) or {}
    latin = tmp_in.get("latin_name") or state.get("current_taxon")

    if not latin:
        return {"_tmp": {**tmp_in, "wiki_status": "not_found"}}

    # 1º intento tal cual
    try:
        page = fetch_fullpage(latin) or {}
        status = page.get("status", "error")
        if status == "ok":
            return {"wiki": page, "_tmp": {**tmp_in, "wiki_status": "ok"}}
    except Exception as e:
        print(f"[wiki_fullpage] ERROR '{latin}': {e}")

    # 2º intento con binomial normalizado (Genus species)
    latin2 = _normalize_binomial(latin)
    if latin2 != latin:
        try:
            page = fetch_fullpage(latin2) or {}
            status = page.get("status", "error")
            if status == "ok":
                return {
                    "wiki": page,
                    "_tmp": {**tmp_in, "wiki_status": "ok", "latin_name": latin2},
                    "current_taxon": latin2,
                }
        except Exception as e:
            print(f"[wiki_fullpage] RETRY ERROR '{latin2}': {e}")

    return {"_tmp": {**tmp_in, "wiki_status": "not_found"}}
