# agent/nodes/infer_local.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
import io
from PIL import Image

# Tool de visión (TorchScript real)
try:
    from agent.tools.vision import infer as _vision_infer  # type: ignore
except Exception as e:
    raise RuntimeError(f"agent.tools.vision.infer no disponible: {e}")

# Mapa label <-> latín para invertir si la tool devuelve latín
try:
    from agent.labels import LABEL_TO_LATIN  # type: ignore
except Exception:
    LABEL_TO_LATIN = {}

_LATIN_TO_LABEL: Dict[str, str] = {v: k for k, v in LABEL_TO_LATIN.items()}


def _to_label(item: Dict[str, Any]) -> Optional[str]:
    """
    Acepta item top-k con 'label' o con 'latin'.
    Devuelve el 'label' si existe o si se puede invertir desde 'latin'.
    """
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("label"), str):
        return item["label"]
    latin = item.get("latin")
    if isinstance(latin, str):
        return _LATIN_TO_LABEL.get(latin)
    return None


def infer_local(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ejecuta clasificador TorchScript y devuelve SOLO labels + métricas.

    Entrada (state):
      - image_bytes: bytes (requerido)

    Salida (mergea en state['_tmp']):
      - p1, p2, margin, entropy: float
      - topk_list: [{"label": str, "prob": float}, ...]
      - preds: [(label: str, prob: float), ...]   # equivalente a topk_list pero en tupla
      - pred_label: str                            # label del top-1 normalizado
      - need_fallback: True                        # si no hay label utilizable
      - error: str                                 # si hay error
    """
    img_bytes = state.get("image_bytes")
    if not img_bytes:
        # Devolver en _tmp anidado para coherencia
        tmp = dict(state.get("_tmp", {}))
        tmp["error"] = "no_image"
        print("[infer_local] ERROR: no_image")
        return {"_tmp": tmp}

    # Bytes -> PIL
    try:
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        tmp = dict(state.get("_tmp", {}))
        tmp["error"] = f"bad_image: {e}"
        print(f"[infer_local] ERROR: bad_image: {e}")
        return {"_tmp": tmp}

    topk = int(state.get("topk", 5))
    try:
        result = _vision_infer(pil, topk=topk)  # <- llama a la TOOL real
    except Exception as e:
        tmp = dict(state.get("_tmp", {}))
        tmp["error"] = f"infer_failed: {e}"
        print(f"[infer_local] ERROR: infer_failed: {e}")
        return {"_tmp": tmp}

    raw_topk = result.get("topk") or []
    metrics = result.get("metrics") or {}

    # DEBUG crudo
    print("\n[infer_local] raw_topk:", raw_topk)
    print("[infer_local] metrics:", metrics)

    # Métricas tolerantes
    p1 = float(metrics.get("p1", 0.0))
    p2 = float(metrics.get("p2", 0.0))
    entropy = float(metrics.get("entropy", 0.0))

    # Normaliza top-k a SOLO labels
    norm_topk: List[Dict[str, Any]] = []
    for it in raw_topk:
        label = _to_label(it)
        prob = float(it.get("prob", 0.0)) if isinstance(it, dict) else 0.0
        if label:
            norm_topk.append({"label": label, "prob": prob})
        else:
            print("[infer_local] Ignorado (no mapea a label):", it)

    pred_label: Optional[str] = norm_topk[0]["label"] if norm_topk else None

    # Construir salida anidada coherente con el resto de nodos
    tmp = dict(state.get("_tmp", {}))
    tmp.update({
        "p1": p1,
        "p2": p2,
        "margin": p1 - p2,
        "entropy": entropy,
        "topk_list": norm_topk,
    })

    if pred_label:
        tmp["pred_label"] = pred_label
        # También generamos 'preds' para el gate (tupla equivalente)
        tmp["preds"] = [(d["label"], float(d["prob"])) for d in norm_topk]
    else:
        tmp["need_fallback"] = True

    # DEBUG final
    print("[infer_local] norm_topk:", norm_topk)
    print("[infer_local] pred_label:", pred_label)
    print(f"[infer_local] p1={tmp['p1']:.6f} margin={tmp['margin']:.6f} entropy={tmp['entropy']:.6f}")

    return {"_tmp": tmp}
