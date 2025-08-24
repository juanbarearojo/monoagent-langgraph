# agent/nodes/map_scientific.py
from __future__ import annotations
from typing import Dict, Any

try:
    from agent.labels import LABEL_TO_LATIN  # {"macaca_fuscata": "Macaca fuscata", ... idealmente ya así}
except Exception:
    LABEL_TO_LATIN = {}

def _normalize_binomial(name: str) -> str:
    """
    Normaliza a 'Genus species' (segunda palabra en minúsculas).
    Admite 'macaca_fuscata' o 'Macaca Fuscata' -> 'Macaca fuscata'
    """
    if not name:
        return name
    name = name.replace("_", " ").strip()
    parts = name.split()
    if len(parts) == 1:
        # Si solo hay una palabra, capitaliza 1ª
        return parts[0][:1].upper() + parts[0][1:].lower()
    genus = parts[0][:1].upper() + parts[0][1:].lower()
    species = parts[1].lower()
    tail = " ".join(parts[2:])  # por si hay subespecie (opcional)
    return f"{genus} {species}" + (f" {tail}" if tail else "")

def map_to_scientific_name(state: Dict[str, Any]) -> Dict[str, Any]:
    tmp = dict(state.get("_tmp", {}))
    pred_label: str | None = tmp.get("pred_label")

    if not pred_label:
        tmp["need_fallback"] = True
        print("[map_to_scientific_name] sin pred_label -> need_fallback=True")
        return {"_tmp": tmp}

    latin = LABEL_TO_LATIN.get(pred_label)
    latin_source = "labels" if latin else "normalized"
    if not latin:
        # antes devolvías "Macaca Fuscata"; ahora lo corregimos a 'Macaca fuscata'
        latin = _normalize_binomial(pred_label)

    tmp["latin_name"] = latin
    tmp["latin_source"] = latin_source
    tmp.pop("need_fallback", None)

    out = {
        "_tmp": tmp,
        "current_taxon": latin,   # clave para wiki
    }
    print(f"[map_to_scientific_name] {pred_label} -> {latin} ({latin_source})")
    return out
