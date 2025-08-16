import json
import time
from pathlib import Path

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

# ------------------ Rutas y carga de artefactos ------------------
ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "monkey_classifier_ts-v0.1.pt"
LABELS_PATH = MODEL_DIR / "labels.json"

assert MODEL_PATH.exists(), f"Modelo no encontrado: {MODEL_PATH}"
assert LABELS_PATH.exists(), f"labels.json no encontrado: {LABELS_PATH}"

# Cargar modelo TorchScript (CPU)
model = torch.jit.load(str(MODEL_PATH), map_location="cpu").eval()

# Cargar metadatos / labels
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

# Soporta ambos formatos: id2label o lista "classes"
if "id2label" in meta:
    id2label = {int(k): v for k, v in meta["id2label"].items()}
    classes = [id2label[i] for i in range(len(id2label))]
elif "classes" in meta:
    classes = meta["classes"]
    id2label = {i: classes[i] for i in range(len(classes))}
else:
    raise ValueError("labels.json debe contener 'id2label' o 'classes'")

mean = meta.get("normalize", {}).get("mean", [0.4363, 0.4328, 0.3291])
std  = meta.get("normalize", {}).get("std",  [0.2129, 0.2075, 0.2038])
input_size = meta.get("input_size", [1, 3, 224, 224])
image_hw = tuple(input_size[2:4])  # [H, W]
NUM_CLASSES = len(classes)

transform = T.Compose([
    T.Resize(image_hw),
    T.ToTensor(),
    T.Normalize(torch.tensor(mean), torch.tensor(std))
])

# ------------------ Utilidades ------------------
def softmax_logits(logits: torch.Tensor) -> torch.Tensor:
    """Garantiza softmax sobre dimensión de clases y devuelve [C]."""
    if logits.dim() == 2:  # [1, C]
        probs = torch.softmax(logits, dim=1).squeeze(0)
    else:  # [C]
        probs = torch.softmax(logits, dim=0)
    return probs

def make_summary_md(result: dict) -> str:
    """Resumen en Markdown legible."""
    pred = result["prediction"]
    tm = result["timing_ms"]
    ts = result["tensor_stats"]
    topk = result["topk"]

    lines = []
    lines.append("### Resumen de inferencia")
    lines.append(f"- **Predicción (top-1):** `{pred['label']}`  — confianza: **{pred['confidence']:.4f}**")
    lines.append(f"- **Entropía:** {result['entropy']:.4f}")
    lines.append(f"- **Imagen:** {result['image_size']['width']}×{result['image_size']['height']} px")
    lines.append(f"- **Tensor:** min={ts['min']:.4f} · max={ts['max']:.4f} · mean={ts['mean']:.4f} · std={ts['std']:.4f}")
    lines.append(f"- **Tiempos (ms):** preprocess={tm['preprocess']:.2f} · inference={tm['inference']:.2f} · total={tm['total']:.2f}")
    lines.append("")
    lines.append("**Top-k:**")
    for i, item in enumerate(topk, 1):
        lines.append(f"{i:>2}. `{item['label']}` — {item['prob']:.4f}")
    return "\n".join(lines)

# ------------------ Inferencia ------------------
@torch.inference_mode()
def infer(image: Image.Image, topk: int = 5):
    """
    Devuelve:
      1) Label (Gradio): dict top-k {label: prob}
      2) JSON con toda la info: predicción, top-k, probs y logits completos, entropía, tiempos, stats
      3) Markdown con resumen formateado
    """
    t0 = time.perf_counter()

    # Preprocesado
    img = image.convert("RGB")
    orig_w, orig_h = img.size
    x = transform(img).unsqueeze(0)  # [1, C, H, W]
    x_min, x_max = float(x.min().item()), float(x.max().item())
    x_mean = float(x.mean().item())
    x_std = float(x.std().item())

    t1 = time.perf_counter()
    logits = model(x)
    t2 = time.perf_counter()

    # Probabilidades
    probs = softmax_logits(logits)  # [C]

    # Top-k
    k = min(int(topk), NUM_CLASSES)
    vals, idxs = torch.topk(probs, k=k)
    topk_labels = [classes[int(i)] for i in idxs.tolist()]
    topk_scores = [float(v) for v in vals.tolist()]
    topk_label_dict = {lbl: sc for lbl, sc in zip(topk_labels, topk_scores)}

    # Top-1
    top1_idx = int(torch.argmax(probs).item())
    top1_label = classes[top1_idx]
    top1_conf = float(probs[top1_idx].item())

    # Entropía (base e)
    eps = 1e-12
    entropy = -float((probs * (probs + eps).log()).sum().item())

    # Logits/Probs completos (en diccionario por claridad)
    if logits.dim() == 2:
        logits_list = logits.squeeze(0).tolist()
    else:
        logits_list = logits.tolist()

    result = {
        "image_size": {"width": orig_w, "height": orig_h},
        "tensor_stats": {"min": x_min, "max": x_max, "mean": x_mean, "std": x_std},
        "prediction": {"label": top1_label, "index": top1_idx, "confidence": top1_conf},
        "topk": [{"label": l, "prob": s} for l, s in zip(topk_labels, topk_scores)],
        "probs": {classes[i]: float(probs[i].item()) for i in range(NUM_CLASSES)},
        "logits": {classes[i]: float(logits_list[i]) for i in range(NUM_CLASSES)},
        "entropy": entropy,
        "timing_ms": {
            "preprocess": (t1 - t0) * 1000.0,
            "inference":  (t2 - t1) * 1000.0,
            "total":      (t2 - t0) * 1000.0
        }
    }

    summary_md = make_summary_md(result)
    return topk_label_dict, result, summary_md

# ------------------ UI ------------------
title = "Monkey Classifier 🐒 — Info detallada (sin gráficas)"
description = (
    "Sube una imagen. Se muestra Top-k, predicción top-1, entropía, "
    "distribución completa de probabilidades y logits, además de tiempos y stats del tensor."
)

demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Imagen"),
        gr.Slider(1, 10, value=5, step=1, label="Top-k"),
    ],
    outputs=[
        gr.Label(num_top_classes=5, label="Top-k (Label)"),
        gr.JSON(label="Detalles (JSON)"),
        gr.Markdown(label="Resumen"),
    ],
    title=title,
    description=description,
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
