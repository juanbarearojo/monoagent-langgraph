---

title: MonoAgent (LangGraph + Gradio)
emoji: 🐒
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk\_version: "latest"
app\_file: app.py
pinned: false
-------------

# MonoAgent — LangGraph + Gradio

Agente y demo con **Gradio**.

* **Código (GitHub):** [https://github.com/juanbarearojo/monoagent-langgraph](https://github.com/juanbarearojo/monoagent-langgraph)
* **Space (Hugging Face):** [https://huggingface.co/spaces/Barearojojuan/MonoAgent](https://huggingface.co/spaces/Barearojojuan/MonoAgent)

> Este Space carga el modelo desde `model/` (subido manualmente) y muestra inferencia detallada sin gráficas.

## Estructura mínima

```
app.py
requirements.txt
model/
 ├─ monkey_classifier_ts-v0.1.pt
 └─ labels.json
```

## Requisitos (Space)

Los paquetes se instalan automáticamente desde `requirements.txt`.
Para ejecución local:

```
pip install -r requirements.txt
python app.py
```

## Notas de despliegue

* El CI **no** sube archivos grandes: `model/monkey_classifier_ts-v0.1.pt` y `model/labels.json` se gestionan **a mano** en la UI del Space.
* Si actualizas el modelo, vuelve a subir esos dos ficheros a `model/`.

## Licencia

MIT

