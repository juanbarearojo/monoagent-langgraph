# agent/tools/vision.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torchvision.transforms as T
from PIL import Image

# -----------------------
# Config / rutas
# -----------------------
MODEL_FILE = os.getenv("MONKEY_MODEL_FILE", "monkey_classifier_ts-v0.1.pt")
LABELS_FILE = os.getenv("MONKEY_LABELS_FILE", "labels.json")

_model = None
_classes: Optional[List[str]] = None
_transform = None

def _candidate_model_dirs() -> List[Path]:
    cands: List[Path] = []

    # 1) Override por entorno
    env_dir = os.getenv("MONKEY_MODEL_DIR")
    if env_dir:
        cands.append(Path(env_dir))

    # 2) agent/model (layout original)
    here = Path(__file__).resolve()
    agent_root = here.parent.parent         # .../agent
    cands.append(agent_root / "model")

    # 3) ./model (raíz del repo — tu layout actual)
    project_root = agent_root.parent
    cands.append(project_root / "model")

    # 4) cwd/model
    cands.append(Path.cwd() / "model")

    # dedupe preservando orden
    seen = set()
    uniq: List[Path] = []
    for d in cands:
        if d not in seen:
            uniq.append(d)
            seen.add(d)
    return uniq

def _resolve_paths() -> Tuple[Path, Path]:
    tried_model, tried_labels = [], []
    for d in _candidate_model_dirs():
        mp = d / MODEL_FILE
        lp = d / LABELS_FILE
        tried_model.append(str(mp))
        tried_labels.append(str(lp))
        if mp.exists() and lp.exists():
            return mp, lp

    msg = (
        "Modelo o labels no encontrados.\n"
        f"Busqué (en orden):\n  modelos: {', '.join(tried_model)}\n"
        f"  labels: {', '.join(tried_labels)}\n"
        "Soluciones:\n"
        "  1) Mueve 'model/' a 'agent/model/'.\n"
        "  2) O define MONKEY_MODEL_DIR apuntando a la carpeta con los ficheros.\n"
        "  3) O ajusta MONKEY_MODEL_FILE / MONKEY_LABELS_FILE.\n"
    )
    raise FileNotFoundError(msg)

def _softmax_1d(logits: torch.Tensor) -> torch.Tensor:
    # logits: [1, C] o [C]
    if logits.dim() == 2:
        return torch.softmax(logits, dim=1).squeeze(0)  # [C]
    return torch.softmax(logits, dim=0)  # [C]

def load_model():
    global _model, _classes, _transform
    if _model is not None:
        return _model

    model_path, labels_path = _resolve_paths()

    _model = torch.jit.load(str(model_path), map_location="cpu").eval()

    with open(labels_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Clases
    if "id2label" in meta:
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        _classes = [id2label[i] for i in range(len(id2label))]
    elif "classes" in meta:
        _classes = list(meta["classes"])
    else:
        raise ValueError("labels.json debe contener 'id2label' o 'classes'")

    # Normalización + tamaño
    mean = meta.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std  = meta.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])
    input_size = meta.get("input_size", [1, 3, 224, 224])
    image_hw = (int(input_size[2]), int(input_size[3])) if len(input_size) >= 4 else (224, 224)

    _transform = T.Compose([
        T.Resize(image_hw),
        T.ToTensor(),
        T.Normalize(mean, std),   # listas/tuplas OK en TorchVision
    ])
    return _model

@torch.inference_mode()
def infer(pil_img: Image.Image, topk: int = 5):
    """
    Devuelve:
      {
        "topk": [{"label": str, "prob": float}, ...],
        "metrics": {"p1": float, "p2": float, "entropy": float},
      }
    """
    model = load_model()
    assert _classes is not None and _transform is not None

    x = _transform(pil_img.convert("RGB")).unsqueeze(0)  # [1,C,H,W]
    logits = model(x)
    probs = _softmax_1d(logits)  # [C]

    k = min(int(topk), len(_classes))
    vals, idxs = torch.topk(probs, k=k)  # [k]

    p1 = float(vals[0].item())
    p2 = float(vals[1].item()) if k >= 2 else 0.0

    # entropía (base e)
    eps = 1e-12
    entropy = -float((probs * (probs + eps).log()).sum().item())

    topk_list = [
        {"label": _classes[i.item()], "prob": float(v.item())}
        for v, i in zip(vals, idxs)
    ]

    return {
        "topk": topk_list,
        "metrics": {"p1": p1, "p2": p2, "entropy": entropy},
    }
