# app.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import io, os

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ───────────────────────── Proyecto (ajusta si difiere) ─────────────────
try:
    from agent.graph import build_graph                 # grafo determinista
except Exception:
    def build_graph():
        raise NotImplementedError("Conecta tu build_graph() real desde agent.graph")

try:
    from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node
except Exception:
    def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = list(state.get("messages", [])) + [AIMessage(content="[QA placeholder]")]
        return {**state, "messages": msgs, "_tmp": {**state.get("_tmp", {}), "qa_answered": True}}

# ───────────────────────── Langfuse (exacto como tu notebook) ─────────────────
def get_langfuse_handler():
    """
    Intenta crear CallbackHandler() de langfuse.langchain.
    Si no está instalado o faltan claves válidas, devuelve None sin romper.
    """
    try:
        # MISMO IMPORT QUE EN TU NOTEBOOK
        from langfuse.langchain import CallbackHandler as LangfuseHandler  # type: ignore
        # en tu ejemplo: handler = CallbackHandler()  (sin argumentos)
        handler = LangfuseHandler()
        # si faltan claves, la lib puede loggear warning pero no queremos romper
        return handler
    except Exception:
        return None

# ───────────────────────── Helpers ─────────────────────────
def normalize_id_output(state_out: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    latin = (
        (state_out.get("_tmp", {}) or {}).get("latin_name")
        or state_out.get("latin_name")
        or state_out.get("current_taxon")
    )
    topk = (state_out.get("_tmp", {}) or {}).get("topk") or state_out.get("topk") or []
    entropy = (state_out.get("_tmp", {}) or {}).get("entropy") or state_out.get("entropy")
    pred_label = (state_out.get("_tmp", {}) or {}).get("pred_label") or state_out.get("pred_label")
    source = (state_out.get("_tmp", {}) or {}).get("latin_source") or state_out.get("latin_source")

    id_report = {
        "pred_label": pred_label,
        "latin_name": latin,
        "topk": topk,
        "entropy": entropy,
        "source": source,
    }

    extra_ctx = {
        "wikipedia_fullpage": state_out.get("wikipedia_fullpage"),
        "context_md": state_out.get("context_md"),
    }
    return latin, id_report, extra_ctx

def build_context_md(extra_ctx: Dict[str, Any]) -> str:
    if extra_ctx.get("context_md"):
        return str(extra_ctx["context_md"])
    wiki = extra_ctx.get("wikipedia_fullpage")
    if not wiki:
        return ""
    return f"## Wikipedia (página completa)\n{wiki}"

def lc_to_ui_messages(msgs: List[BaseMessage]) -> List[Dict[str, str]]:
    out = []
    for m in msgs:
        role = "assistant" if isinstance(m, AIMessage) else "user"
        out.append({"role": role, "content": m.content})
    return out

# Estado del grafo (lazy)
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ───────────────────────── Callbacks ─────────────────────────
def do_identify(image) -> Tuple[str, Dict[str, Any], List[Dict[str, str]], bytes, Dict[str, Any], List[Any], str]:
    """Ejecuta el grafo (bloqueante), pinta el mensaje de finalize y devuelve estado de chat."""
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1"

    # PIL → bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in: Dict[str, Any] = {"messages": [], "image_bytes": img_bytes}

    # Langfuse handler (como tu notebook)
    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    thread_id = "test1"  # EXACTO como pides

    # INVOKE BLOQUEANTE con callbacks y thread_id fijo
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": callbacks, "configurable": {"thread_id": thread_id}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)

    # Mensaje de finalize (último AIMessage del grafo)
    finalize_ai = None
    for m in reversed(state_out.get("messages", [])):
        if isinstance(m, AIMessage):
            finalize_ai = m
            break

    if not latin:
        ui_msgs = [{"role": "assistant", "content": "No he podido identificar la especie. Sube otra imagen y pulsa **Identificar**."}]
        lc_messages: List[Any] = state_out.get("messages", [])
        return "No identificado (repite o sube otra imagen)", id_report, ui_msgs, img_bytes, extra_ctx, lc_messages, thread_id

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**. Pregunta sobre hábitat, dieta, distribución, conservación…"
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    lc_messages: List[Any] = state_out.get("messages", []) or [AIMessage(content=first_msg)]

    return latin, id_report, ui_msgs, img_bytes, extra_ctx, lc_messages, thread_id

def redo_identify(last_image: Optional[bytes], prev_thread_id: str) -> Tuple[str, Dict[str, Any], List[Dict[str, str]], bytes, Dict[str, Any], List[Any], str]:
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1"

    state_in = {"messages": [], "image_bytes": last_image}

    handler = get_langfuse_handler()
    callbacks = [handler] if handler else []
    thread_id = "test1"  # fijo

    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": callbacks, "configurable": {"thread_id": thread_id}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)

    finalize_ai = None
    for m in reversed(state_out.get("messages", [])):
        if isinstance(m, AIMessage):
            finalize_ai = m
            break

    if not latin:
        ui_msgs = [{"role": "assistant", "content": "No he podido identificar la especie. Sube otra imagen distinta y pulsa **Identificar**."}]
        lc_messages: List[Any] = state_out.get("messages", [])
        return "No identificado (repite o sube otra imagen)", id_report, ui_msgs, last_image, extra_ctx, lc_messages, thread_id

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**. ¡Pregunta lo que quieras!"
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    lc_messages: List[Any] = state_out.get("messages", []) or [AIMessage(content=first_msg)]

    return latin, id_report, ui_msgs, last_image, extra_ctx, lc_messages, thread_id

def do_chat(user_msg: str,
            current_taxon: str,
            ui_messages: List[Dict[str, str]],
            id_report: Dict[str, Any],
            extra_ctx: Dict[str, Any],
            lc_messages: List[Any],
            thread_id: str) -> Tuple[List[Dict[str, str]], str, List[Any]]:
    """
    - Añade HumanMessage al historial LC.
    - Llama a QA (bloqueante) con context_md (Wikipedia).
    - Si falta OPENAI_API_KEY, responde amable sin colgarse.
    """
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages

    if not current_taxon or current_taxon.startswith("(") or current_taxon.startswith("No identificado"):
        tip = "Primero identifica una especie: sube imagen y pulsa **Identificar**."
        return ui_messages + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": tip}], "", lc_messages

    if not os.getenv("OPENAI_API_KEY"):
        msg = ("QA deshabilitado: falta `OPENAI_API_KEY` en el entorno.\n"
               "Ve a *Settings → Secrets* y añade tu clave para habilitar preguntas.")
        new_ui = ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": msg},
        ]
        return new_ui, "", lc_messages

    lc_hist = list(lc_messages) + [HumanMessage(content=user_msg)]
    state_in: Dict[str, Any] = {
        "messages": lc_hist,
        "current_taxon": current_taxon,
        "context_md": build_context_md(extra_ctx),
    }

    # El QA no usa Langfuse aquí (es un nodo aparte). Si quieres trazar también,
    # puedes adaptarlo dentro del nodo usando langfuse.langchain.
    try:
        state_out = qa_node(state_in)
    except Exception as e:
        answer = f"Ocurrió un error en QA: {type(e).__name__}: {e}"
        return ui_messages + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": answer}], "", lc_hist

    out_msgs = state_out.get("messages", [])
    last_ai = next((m for m in reversed(out_msgs) if isinstance(m, AIMessage)), None)
    answer = (last_ai.content if last_ai else "").strip()

    if not answer:
        friendly = "No he podido generar respuesta de QA ahora mismo. ¿Tienes configurada la variable `OPENAI_API_KEY`?"
        new_ui = ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": friendly},
        ]
        return new_ui, "", out_msgs or lc_hist

    new_ui = list(ui_messages) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", out_msgs

def reset_all():
    return "(Sube una imagen)", {}, [], None, {}, [], "test1",

# ───────────────────────── UI (Gradio) ─────────────────────────
with gr.Blocks(title="MonoAgent · Identificación + QA", fill_height=True, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐒 MonoAgent — Identificación (grafo) + QA (Wikipedia)")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(label="Sube una imagen del animal", type="pil")
            btn_identify = gr.Button("🔍 Identificar", variant="primary")
            btn_reidentify = gr.Button("↻ Re-identificar")
            btn_reset = gr.Button("🧹 Reiniciar")
        with gr.Column(scale=1):
            current_taxon = gr.Label(value="(Sube una imagen)", label="Especie identificada")
            id_report = gr.JSON(label="Detalle de predicción")

    gr.Markdown("---")
    with gr.Row():
        with gr.Column():
            chat = gr.Chatbot(label="Preguntas sobre la especie", type="messages", height=420)
            user_box = gr.Textbox(placeholder="Escribe tu pregunta…", label="Tu pregunta")
            btn_ask = gr.Button("Enviar")

    # Estados
    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_chat_msgs = gr.State([])
    st_lc = gr.State([])
    st_thread = gr.State("test1")  # mantenemos el thread_id fijo

    # Identificar
    btn_identify.click(
        fn=do_identify,
        inputs=[image_in],
        outputs=[current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread],
        show_progress="minimal",
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

    # Re-identificar (misma imagen)
    btn_reidentify.click(
        fn=redo_identify,
        inputs=[st_last_image, st_thread],
        outputs=[current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread],
        show_progress="minimal",
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

    # Chat (botón)
    btn_ask.click(
        fn=do_chat,
        inputs=[user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread],
        outputs=[st_chat_msgs, user_box, st_lc],
        show_progress="minimal",
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

    # Chat (Enter)
    user_box.submit(
        fn=do_chat,
        inputs=[user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread],
        outputs=[st_chat_msgs, user_box, st_lc],
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

    # Reset
    btn_reset.click(
        fn=reset_all,
        inputs=[],
        outputs=[current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, user_box],
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

if __name__ == "__main__":
    demo.launch()
