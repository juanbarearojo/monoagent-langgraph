# agent/nodes/gate_uncertainty.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
from agent.state import ChatVisionState

def gate_uncertainty(state: ChatVisionState) -> Dict[str, Any]:
    # 1) Cargar _tmp actual
    tmp = dict(state.get("_tmp", {}))

    # 2) COMPAT: mover cualquier clave plana "_tmp.xxx" al dict anidado
    for k, v in list(state.items()):
        if isinstance(k, str) and k.startswith("_tmp."):
            tmp[k.split(".", 1)[1]] = v

    # 3) Asegurar preds (acepta topk_list o preds)
    topk_list: List[Dict[str, Any]] = tmp.get("topk_list") or []
    preds: List[Tuple[str, float]] = tmp.get("preds") or [
        (d["label"], float(d.get("prob", 0.0)))
        for d in topk_list if isinstance(d, dict) and "label" in d
    ]

    # 4) Métricas
    p1 = float(tmp.get("p1", preds[0][1] if preds else 0.0))
    margin = float(tmp.get("margin", 0.0))

    accept_thr = float(state.get("accept_threshold", 0.7))
    margin_thr = float(state.get("min_margin", 0.1))

    accept = bool(preds) and (p1 >= accept_thr) and (margin >= margin_thr)

    # DEBUG
    if preds:
        print(f"[GATE] pred={preds[0][0]} p1={p1:.4f} margin={margin:.4f} "
              f"thr={accept_thr} mthr={margin_thr} -> {'ACCEPT' if accept else 'REVIEW'}")
    else:
        print("[GATE] sin preds -> REVIEW")

    tmp["preds"] = preds
    tmp["gate"] = "ACCEPT" if accept else "REVIEW"
    return {"_tmp": tmp}
