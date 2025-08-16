# Monkey Classifier (TorchScript) — v0.1

Clasificador de 10 especies de monos exportado a **TorchScript** para despliegue sencillo en CPU (Hugging Face Spaces o entornos locales).

* **Artefactos**: `model/monkey_classifier_ts-v0.1.pt` (TorchScript) y `model/labels.json` (metadatos y clases).
* **Space de demo**: [https://huggingface.co/spaces/Barearojojuan/MonoAgent](https://huggingface.co/spaces/Barearojojuan/MonoAgent)
+
---

## 🧠 Detalles del modelo

* **Arquitectura**: ResNet (cargado y exportado como TorchScript).
* **Entrada**: imagen RGB, tamaño `224×224`.
* **Salida**: vector de logits de tamaño 10 → *softmax* para probabilidades.
* **Normalización**:

  * mean = `[0.4363, 0.4328, 0.3291]`
  * std  = `[0.2129, 0.2075, 0.2038]`

### Clases (id2label)

```
0: Mantled_howler
1: Patas_monkey
2: Bald_uakari
3: Japanese_macaque
4: Pygmy_marmoset
5: White_headed_capuchin
6: Silvery_marmoset
7: Common squirrel_monkey
8: Black_headed_night_monkey
9: Nilgiri_langur
```

> El archivo `labels.json` contiene `id2label`, `label2id`, `input_size` y `normalize`.

---

## ✅ Uso recomendado

### Requisitos

```
pip install torch torchvision pillow
```

### Carga (TorchScript) e inferencia mínima

```python
import json, torch
from PIL import Image
import torchvision.transforms as T

# Cargar modelo y labels
model = torch.jit.load("model/monkey_classifier_ts-v0.1.pt", map_location="cpu").eval()
meta = json.load(open("model/labels.json","r",encoding="utf-8"))

id2label = {int(k): v for k, v in meta["id2label"].items()}
mean = meta["normalize"]["mean"]; std = meta["normalize"]["std"]
H, W = meta["input_size"][2], meta["input_size"][3]

transform = T.Compose([
    T.Resize((H, W)),
    T.ToTensor(),
    T.Normalize(torch.tensor(mean), torch.tensor(std))
])

@torch.inference_mode()
def predict(path):
    x = transform(Image.open(path).convert("RGB")).unsqueeze(0)
    probs = torch.softmax(model(x), dim=1).squeeze(0)
    top_prob, top_idx = probs.max(dim=0)
    return id2label[int(top_idx)], float(top_prob)

label, p = predict("test.jpg")
print(label, p)
```

---

## 📦 Exportación y compatibilidad

El modelo se exportó desde PyTorch a **TorchScript** (`torch.jit.trace`), lo que:

* evita *pickle* (más seguro en servidores públicos),
* reduce dependencias de código,
* mejora la portabilidad entre versiones de PyTorch.

> Entrada esperada: `1×3×224×224`. Ajusta el *trace* si cambias tamaño de entrada o *preprocessing*.

---

## 🧪 Datos y *preprocessing*

* **Origen de datos**: conjunto propio (10 clases).
* **Aumento de datos** (recomendado durante entrenamiento): *random resize/crop*, *horizontal flip*, *color jitter* (si procede).
* **Preprocesado en inferencia**: `Resize(224,224) → ToTensor → Normalize(mean,std)` (ver arriba).

---

## 👀 Limitaciones y sesgos

* Dataset de tamaño moderado (10 clases): posible **sobreajuste** y **sesgo** hacia condiciones de captura similares al *train*.
* Sensible a **oclusiones**, **iluminación extrema** y **ángulos inusuales**.
* No substituye validación experta para tareas críticas.

**Mitigación sugerida**: aumentar diversidad del dataset, *augmentation* más fuerte, evaluación en *out-of-distribution*, calibración de probabilidades y *thresholding* según aplicación.

---

## 🔒 Consideraciones de seguridad

* TorchScript minimiza riesgos de deserialización comparado con `.pth` pickled.
* Verifica integridad con **SHA256** del `.pt` (añade aquí tu suma si la calculas):
  `SHA256(monkey_classifier_ts-v0.1.pt) = <TU_HASH>`

---

## 🔁 Versionado

* **v0.1**: primera versión pública (10 clases, normalización indicada, entrada 224×224).
* Cambios en el orden de clases o en el *preprocessing* deben ir acompañados de un **bump de versión mayor**.

---

## 📄 Licencia

MIT. Consulta `LICENSE` en el repositorio.

---

## ✍️ Cita

Si este modelo o demo te resulta útil, por favor cita el Space o el repositorio:

```
@software{monoagent_monkey_classifier,
  author  = {Barea Rojo, Juan},
  title   = {Monkey Classifier (TorchScript) – MonoAgent Space},
  year    = {2025},
}
```
