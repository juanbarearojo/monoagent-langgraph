from typing import Literal

def gate_decision(policy: str, threshold: float, p1: float, p2: float, entropy: float) -> Literal["ACCEPT","REVIEW"]:
    if policy == "confidence":
        return "ACCEPT" if p1 >= threshold else "REVIEW"
    if policy == "margin":
        return "ACCEPT" if (p1 - p2) >= threshold else "REVIEW"
    # default entropy
    return "ACCEPT" if entropy <= threshold else "REVIEW"