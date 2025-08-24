# agent/nodes/map_scientific.py
from __future__ import annotations
from typing import Any, Dict

# Mapa de etiquetas -> nombre científico (dataset clásico)
try:
    from agent.labels import LABEL_TO_LATIN  # type: ignore
except Exception:
    LABEL_TO_LATIN = {
        "Mantled_howler": "Alouatta palliata",
        "Patas_monkey": "Erythrocebus patas",
        "Bald_uakari": "Cacajao calvus",
        "Japanese_macaque": "Macaca fuscata",
        "Pygmy_marmoset": "Cebuella pygmaea",
        "White_headed_capuchin": "Cebus capucinus",
        "Silvery_marmoset": "Mico argentatus",
        "Common_squirrel_monkey": "Saimiri sciureus",
        "Black_headed_night_monkey": "Aotus nigriceps",
        "Nilgiri_langur": "Semnopithecus johnii",
    }

# Validador binomial
try:
    from agent.state import valid_binomial as _valid_binomial  # type: ignore
except Exception:
    import re
    _RE = re.compile(r"^[A-Z][a-z]+ [a-z]+$")
    def _valid_binomial(s: str) -> bool:
        return bool(s and _RE.match(s))

# Whitelist de géneros conocidos (derivada de LABEL_TO_LATIN)
_GENUS_WHITELIST = {
    (v.split()[0].strip().lower())
    for v in LABEL_TO_LATIN.values()
    if isinstance(v, str) and len(v.split()) >= 2
}

def _normalize_label_key(s: str) -> str:
    if not s:
        return ""
    s = s.strip().replace("-", "_").replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    return s

def _slug_to_binomial(label: str) -> str:
    """
    Convierte 'macaca_fuscata' / 'macaca-fuscata' a 'Macaca fuscata'
    SOLO si el género está en la whitelist (_GENUS_WHITELIST).
    """
    if not label or not isinstance(label, str):
        return ""
    s = label.replace("_", " ").replace("-", " ").strip()
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        return ""
    g, e = parts[0], parts[1]
    if not g.isalpha() or not e.isalpha():
        return ""
    if g.lower() not in _GENUS_WHITELIST:
        return ""  # evita falsos positivos tipo "Unknown monkey"
    binom = g[:1].upper() + g[1:].lower() + " " + e.lower()
    return binom if _valid_binomial(binom) else ""

def map_to_scientific_name(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lee: state['_tmp.pred_label']
    Devuelve:
      - {'_tmp.latin_name': <binomial>, '_tmp.latin_source': 'map'|'normalized'}
      - o {'_tmp.need_fallback': True, '_tmp.latin_source': 'map:miss'}
    """
    raw = state.get("_tmp.pred_label")
    if not raw or not isinstance(raw, str):
        return {"_tmp.need_fallback": True, "_tmp.latin_source": "map:miss"}

    # 1) Intento por mapping (label estilo dataset)
    latin = LABEL_TO_LATIN.get(raw)
    if not latin:
        latin = LABEL_TO_LATIN.get(_normalize_label_key(raw))

    if latin and _valid_binomial(latin):
        return {"_tmp.latin_name": latin, "_tmp.latin_source": "map"}

    # 2) Intento por normalización de slug si el género es conocido
    latin_norm = _slug_to_binomial(raw) or _slug_to_binomial(_normalize_label_key(raw))
    if latin_norm and _valid_binomial(latin_norm):
        return {"_tmp.latin_name": latin_norm, "_tmp.latin_source": "normalized"}

    # 3) Sin match
    return {"_tmp.need_fallback": True, "_tmp.latin_source": "map:miss"}
