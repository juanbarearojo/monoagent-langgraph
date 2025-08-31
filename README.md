---
title: MonoAgent
emoji: 🐵
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk\_version: "5.42.0"
app\_file: app.py
pinned: false
-------------

# MonoAgent — Agente con LangGraph + Gradio

**MonoAgent** es un agente construido con **LangGraph** que integra un clasificador de primates, recuperación de contexto externo y capacidades de razonamiento con LLMs multimodales. Se ejecuta en **Gradio** y está pensado como un asistente zoológico capaz de identificar especies y responder preguntas con información científica adicional.

* **Código (GitHub):** [https://github.com/juanbarearojo/monoagent-langgraph](https://github.com/juanbarearojo/monoagent-langgraph)
* **Space (Hugging Face):** [https://huggingface.co/spaces/Barearojojuan/MonoAgent](https://huggingface.co/spaces/Barearojojuan/MonoAgent)

## Arquitectura del agente

El agente se organiza en torno a un **grafo de nodos** definido en `agent/graph.py`. Cada nodo encapsula una función específica, y el estado fluye entre ellos según las condiciones y políticas definidas en `agent/state.py` y `agent/policies.py`.

<p align="center">
  <img src="grafo.png" alt="grafo del agente" width="600"/>
</p>

### Nodos principales

* **router\_input (`agent/nodes/router.py`)** → Clasifica la entrada inicial (imagen o texto).
* **ensure\_image (`agent/nodes/ensure_image.py`)** → Comprueba si hay imagen disponible o solicita una.
* **infer\_local (`agent/nodes/infer_local.py`)** → Ejecuta el clasificador TorchScript (`model/monkey_classifier_ts-v0.1.pt`).
* **gate\_uncertainty (`agent/nodes/gate_uncertainty.py`)** → Evalúa la confianza de la predicción; si es baja, deriva a GPT‑4 Vision.
* **map\_to\_scientific\_name (`agent/nodes/map_scientific.py`)** → Traduce etiquetas internas al nombre científico usando `agent/labels.py`.
* **ask\_gpt41\_vision (`agent/nodes/ask_gpt41_vision.py`)** → LLM multimodal para casos fuera de distribución.
* **species\_retrieve (`agent/nodes/species_retrieve.py`)** → Recuperación semántica (RAG) con índices FAISS (`indices/global/`).
* **fetch\_wikipedia\_fullpage (`agent/nodes/wiki_fullpage.py`)** → Descarga y procesa páginas completas de Wikipedia.
* **merge\_context (`agent/nodes/merge_context.py`)** → Combina información de Wikipedia y RAG.
* **finalize (`agent/nodes/finalize.py`)** → Construye la respuesta final para el usuario.
* **qa\_about\_taxon (`agent/nodes/qa_about_taxon.py`)** → Responde preguntas directas sobre un taxón ya identificado.
* **clarify (`agent/nodes/clarify.py`)** y **prompt\_for\_image (`agent/nodes/prompt_for_image.py`)** → Manejo de casos en los que falta información.
* **capture\_user\_taxon (`agent/nodes/capture_user_taxon.py`)** → Permite al usuario proporcionar manualmente el nombre del taxón.

### Herramientas (`agent/tools/`)

* **ddg.py** → Búsqueda DuckDuckGo.
* **wiki.py** → Descarga de artículos de Wikipedia.
* **vision.py** → Inferencia con modelos multimodales.
* **gpt.py** → Conexión con LLMs (GPT‑4/3.5).
* **rag\_index.py** → Construcción y búsqueda en índices FAISS.

### Utilidades (`agent/utils/`)

* **images.py** → Procesamiento de imágenes y conversión a bytes.
* **text.py** → Normalización y validaciones de texto.

## Flujo general

1. El usuario envía una imagen o pregunta.
2. El sistema valida la entrada (router/ensure\_image).
3. Si hay imagen → se ejecuta el clasificador local.
4. Se evalúa la confianza → si es baja, se consulta a GPT‑4 Vision.
5. El resultado se mapea a nombre científico.
6. Se buscan artículos completos de Wikipedia y contexto adicional en el índice RAG.
7. Se combinan las fuentes en una respuesta final.
8. El agente responde o pide aclaraciones si falta información.

## Ejecución local

```bash
pip install -r requirements.txt
python app.py
```

## Notas de despliegue

* El flujo de CI/CD prepara automáticamente la versión para Hugging Face Space.
* Los binarios grandes (`.pt`, `.faiss`, PDFs de `data/corpus/`) pueden gestionarse con **Git LFS**.

## Licencia

MIT
