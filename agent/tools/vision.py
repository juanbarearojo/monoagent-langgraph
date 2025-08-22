# agent/tools/vision.py
import json, time
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "monkey_classifier_ts-v0.1.pt"
LABELS_PATH = MODEL_DIR / "labels.json"

_model = None
_classes = None
_transform = None

def _softmax_1d(logits: torch.Tensor) -> torch.Tensor:
    # logits: [1, C] o [C]
    if logits.dim() == 2:
        return torch.softmax(logits, dim=1).squeeze(0)  # [C]
    return torch.softmax(logits, dim=0)  # [C]

def load_model():
    global _model, _classes, _transform
    if _model is not None:
        return _model

    assert MODEL_PATH.exists(), f"Modelo no encontrado: {MODEL_PATH}"
    assert LABELS_PATH.exists(), f"labels.json no encontrado: {LABELS_PATH}"

    _model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "id2label" in meta:
        id2label = {int(k): v for k, v in meta["id2label"].items()}
        _classes = [id2label[i] for i in range(len(id2label))]
    elif "classes" in meta:
        _classes = meta["classes"]
    else:
        raise ValueError("labels.json debe contener 'id2label' o 'classes'")

    mean = meta.get("normalize", {}).get("mean", [0.485, 0.456, 0.406])
    std  = meta.get("normalize", {}).get("std",  [0.229, 0.224, 0.225])
    input_size = meta.get("input_size", [1, 3, 224, 224])
    image_hw = tuple(input_size[2:4])

    _transform = T.Compose([
        T.Resize(image_hw),
        T.ToTensor(),
        T.Normalize(torch.tensor(mean), torch.tensor(std))
    ])
    return _model

@torch.inference_mode()
def infer(pil_img: Image.Image, topk: int = 5):
    """
    Devuelve:
      {
        "topk": [{"latin": str, "prob": float}, ...],
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

    # p1, p2 seguros aunque haya 1 clase
    p1 = float(vals[0].item())
    p2 = float(vals[1].item()) if k >= 2 else 0.0

    # entropía (base e)
    eps = 1e-12
    entropy = -float((probs * (probs + eps).log()).sum().item())

    topk_list = [
        {"latin": _classes[i.item()], "prob": float(v.item())}
        for v, i in zip(vals, idxs)
    ]

    return {
        "topk": topk_list,
        "metrics": {"p1": p1, "p2": p2, "entropy": entropy},
    }
