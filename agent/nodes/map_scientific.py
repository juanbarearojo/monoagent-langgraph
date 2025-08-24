# agent/nodes/map_scientific.py
from __future__ import annotations
from typing import Any, Dict
from agent.labels import LABEL_TO_LATIN  # type: ignore
from agent.state import valid_binomial as _valid_binomial  # type: ignore

def _normalize_label(s: str) -> str:
    """
    Normaliza etiquetas del clasificador para mejorar el matching:
      - quita espacios extremos
      - reemplaza espacios y guiones por '_'
      - colapsa múltiple '_' consecutivos
      - respeta el casing original de las claves (no lower())
    """
    if not s:
        return ""
    s = s.strip().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def map_to_scientific_name(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lee: state['_tmp.pred_label']
    Devuelve:
      - {'_tmp.latin_name': <binomial>, '_tmp.latin_source': 'map'} si hay match
      - {'_tmp.need_fallback': True, '_tmp.latin_source': 'map:miss'} si no hay match
    """
    raw = state.get("_tmp.pred_label")
    if not raw or not isinstance(raw, str):
        return {"_tmp.need_fallback": True, "_tmp.latin_source": "map:miss"}

    # 1) intento exacto
    latin = LABEL_TO_LATIN.get(raw)
    # 2) intento normalizado (espacios/guiones -> '_')
    if not latin:
        latin = LABEL_TO_LATIN.get(_normalize_label(raw))

    if not latin or not _valid_binomial(latin):
        return {"_tmp.need_fallback": True, "_tmp.latin_source": "map:miss"}

    return {"_tmp.latin_name": latin, "_tmp.latin_source": "map"}
