# app.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import io, os, uuid

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

# ───────────────────────── Langfuse (opcional) ─────────────────────────
def get_callbacks():
    """
    Devuelve callbacks para Langfuse si está instalado y con credenciales.
    Prioriza variable de entorno LANGFUSE_SECRET_KEY/ LANGFUSE_PUBLIC_KEY / LANGFUSE_HOST.
    Si no están, no añade callbacks.
    """
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # type: ignore
        # Solo instanciar si hay credenciales
        if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
            return [LangfuseCallbackHandler(
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
                host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            )]
    except Exception:
        pass
    return []

def make_thread_id(prefix: str = "session") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

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
    """Convierte mensajes LangChain → formato Chatbot(type='messages')."""
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
    """
    1) Ejecuta el grafo (bloqueante) con Langfuse callbacks y thread_id.
    2) Muestra como primer mensaje el AIMessage de 'finalize' si existe.
    3) Prepara historial LangChain para QA posterior.
    """
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, [], make_thread_id("noimg")

    # PIL → bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    # Entrada mínima para el grafo
    state_in: Dict[str, Any] = {"messages": [], "image_bytes": img_bytes}

    # Config Langfuse + thread_id
    callbacks = get_callbacks()
    thread_id = make_thread_id("identify")

    # ⚠️ INVOKE BLOQUEANTE: no actualizamos UI hasta acabar
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": callbacks, "configurable": {"thread_id": thread_id}},
    )

    # Extraer datos
    latin, id_report, extra_ctx = normalize_id_output(state_out)

    # Primer mensaje al usuario: el que haya puesto el nodo 'finalize'
    # (buscamos último AIMessage)
    finalize_ai = None
    for m in reversed(state_out.get("messages", [])):
        if isinstance(m, AIMessage):
            finalize_ai = m
            break

    if not latin:
        ui_msgs = [{"role": "assistant", "content": "No he podido identificar la especie. Sube otra imagen y pulsa **Identificar**."}]
        lc_messages: List[Any] = state_out.get("messages", [])
        return "No identificado (repite o sube otra imagen)", id_report, ui_msgs, img_bytes, extra_ctx, lc_messages, thread_id

    if finalize_ai:
        first_msg = finalize_ai.content
    else:
        first_msg = f"Identificado: **{latin}**. Pregunta sobre hábitat, dieta, distribución, conservación…"

    ui_msgs = [{"role": "assistant", "content": first_msg}]
    # Historial LC: arrancamos con lo que dejó el grafo (para mantener trazabilidad)
    lc_messages: List[Any] = state_out.get("messages", []) or [AIMessage(content=first_msg)]

    return latin, id_report, ui_msgs, img_bytes, extra_ctx, lc_messages, thread_id

def redo_identify(last_image: Optional[bytes], prev_thread_id: str) -> Tuple[str, Dict[str, Any], List[Dict[str, str]], bytes, Dict[str, Any], List[Any], str]:
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, [], make_thread_id("noimg")

    state_in = {"messages": [], "image_bytes": last_image}
    callbacks = get_callbacks()
    # Reutilizamos o regeneramos thread id
    thread_id = prev_thread_id or make_thread_id("identify")

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
    - Llama al nodo QA (bloqueante) con context_md (Wikipedia).
    - Añade la respuesta del LLM al UI y devuelve historial LC completo.
    """
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages

    if not current_taxon or current_taxon.startswith("(") or current_taxon.startswith("No identificado"):
        tip = "Primero identifica una especie: sube imagen y pulsa **Identificar**."
        return ui_messages + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": tip}], "", lc_messages

    # Historial LC → + humano
    lc_hist = list(lc_messages) + [HumanMessage(content=user_msg)]

    # Estado para el nodo QA
    state_in: Dict[str, Any] = {
        "messages": lc_hist,
        "current_taxon": current_taxon,
        "context_md": build_context_md(extra_ctx),
    }

    # Llamada bloqueante al nodo QA
    try:
        state_out = qa_node(state_in)
    except Exception as e:
        answer = f"Ocurrió un error en QA: {type(e).__name__}: {e}"
        return ui_messages + [{"role": "user", "content": user_msg},
                              {"role": "assistant", "content": answer}], "", lc_hist

    out_msgs = state_out.get("messages", [])
    last_ai = next((m for m in reversed(out_msgs) if isinstance(m, AIMessage)), None)
    answer = last_ai.content if last_ai else "(QA no devolvió respuesta)"

    new_ui = list(ui_messages) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", out_msgs  # limpiamos textbox; devolvemos historial LC completo

def reset_all():
    return "(Sube una imagen)", {}, [], None, {}, [], "",

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
            # ✅ formato estable (openai-style)
            chat = gr.Chatbot(label="Preguntas sobre la especie", type="messages", height=420)
            user_box = gr.Textbox(placeholder="Escribe tu pregunta…", label="Tu pregunta")
            btn_ask = gr.Button("Enviar")

    # Estados
    st_last_image = gr.State(None)      # bytes
    st_extra_ctx = gr.State({})         # wiki/context_md
    st_chat_msgs = gr.State([])         # mensajes UI [{role, content}, ...]
    st_lc = gr.State([])                # LangChain messages [Human/AI...]
    st_thread = gr.State("")            # thread_id para Langfuse/configurable

    # Identificar (bloqueante hasta terminar invoke)
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

    # Reset total
    btn_reset.click(
        fn=reset_all,
        inputs=[],
        outputs=[current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, user_box],
    ).then(lambda msgs: msgs, inputs=[st_chat_msgs], outputs=[chat])

if __name__ == "__main__":
    # Si usas HF Space, puedes dejarlo por defecto; si lo expones en contenedor, usa server_name="0.0.0.0"
    demo.launch()
