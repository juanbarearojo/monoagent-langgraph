# agent/nodes/gate_uncertainty.py
from agent.tools.vision import infer
from agent.policies import gate_decision

DEFAULTS = {
    "entropy":   1.5,   # más bajo = más estricto
    "confidence": 0.75, # aceptar si p1 >= 0.75
    "margin":     0.25, # aceptar si (p1 - p2) >= 0.25
}

def gate_uncertainty(state):
    pil = state["pil_image"]          # asegura que ensure_image la dejó lista
    k   = state.get("topk", 5)
    res = infer(pil, topk=k)

    p1 = res["metrics"]["p1"]
    p2 = res["metrics"]["p2"]
    ent = res["metrics"]["entropy"]

    policy = state.get("accept_policy", "entropy")
    threshold = state.get("accept_threshold", DEFAULTS[policy])

    decision = gate_decision(policy, threshold, p1, p2, ent)

    # Guarda para siguientes nodos
    state["local_topk"] = res["topk"]          # lista de {latin, prob}
    state["local_metrics"] = res["metrics"]    # {p1,p2,entropy}
    state["local_decision"] = decision         # "ACCEPT" | "REVIEW"
    return state
