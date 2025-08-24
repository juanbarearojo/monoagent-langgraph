# agent/nodes/map_scientific.py
from __future__ import annotations
from typing import Dict, Any

try:
    from agent.labels import LABEL_TO_LATIN  # p.ej. {"macaca_fuscata": "Macaca fuscata", ...}
except Exception:
    LABEL_TO_LATIN = {}

def _nice(s: str) -> str:
    # "macaca_fuscata" -> "Macaca fuscata"
    return s.replace("_", " ").strip().title()

def map_to_scientific_name(state: Dict[str, Any]) -> Dict[str, Any]:
    tmp = dict(state.get("_tmp", {}))
    pred_label: str | None = tmp.get("pred_label")

    if not pred_label:
        # no hay label -> pedir fallback a visión
        tmp["need_fallback"] = True
        print("[map_to_scientific_name] sin pred_label -> need_fallback=True")
        return {"_tmp": tmp}

    latin = LABEL_TO_LATIN.get(pred_label)
    if not latin:
        # si no existe en el mapa, al menos formatea algo legible
        latin = _nice(pred_label)
        latin_source = "normalized"
    else:
        latin_source = "labels"

    tmp["latin_name"] = latin
    tmp["latin_source"] = latin_source
    tmp.pop("need_fallback", None)

    # ⚠️ muy importante para los siguientes nodos:
    # deja también el 'current_taxon' en raíz
    out = {
        "_tmp": tmp,
        "current_taxon": latin,
    }

    print(f"[map_to_scientific_name] {pred_label} -> {latin} ({latin_source})")
    return out
