# app.py
from typing import Any, Dict, List, Optional, Tuple
import io, os

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ───────────────────────── Proyecto (ajusta si difiere) ─────────────────
from agent.graph import build_graph                          # grafo determinista
from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node

# ───────────────────────── Langfuse (estricto, sin try/except) ─────────
# ⛔️ NOTA: en producción, mejor mover estas claves a Secrets/ENV.
from langfuse.langchain import CallbackHandler  # si no está instalado, fallará aquí (bien para depurar)


# valida presencia explícita
_missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"[Langfuse] Faltan variables de entorno: {_missing}")
handler = CallbackHandler()  # si las claves son inválidas, verás el error en logs

# ───────────────────────── Helpers ─────────────────────────
def normalize_id_output(state_out: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
    latin = (
        (state_out.get("_tmp", {}) or {}).get("latin_name")
        or state_out.get("latin_name")
        or state_out.get("current_taxon")
    )
    id_report = {
        "pred_label": (state_out.get("_tmp", {}) or {}).get("pred_label") or state_out.get("pred_label"),
        "latin_name": latin,
        "topk": (state_out.get("_tmp", {}) or {}).get("topk") or state_out.get("topk") or [],
        "entropy": (state_out.get("_tmp", {}) or {}).get("entropy") or state_out.get("entropy"),
        "source": (state_out.get("_tmp", {}) or {}).get("latin_source") or state_out.get("latin_source"),
    }
    extra_ctx = {
        "wikipedia_fullpage": state_out.get("wikipedia_fullpage"),
        "context_md": state_out.get("context_md"),
    }
    return latin, id_report, extra_ctx

def build_context_md(extra_ctx: Dict[str, Any]) -> str:
    if extra_ctx.get("context_md"):
        return str(extra_ctx["context_md"])
    if extra_ctx.get("wikipedia_fullpage"):
        return f"## Wikipedia\n{extra_ctx['wikipedia_fullpage']}"
    return ""

# Estado grafo lazy
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ───────────────────────── Callbacks ─────────────────────────
def do_identify(image):
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1"

    buf = io.BytesIO(); image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in = {"messages": [], "image_bytes": img_bytes}

    # Langfuse EXACTO como pediste (handler creado arriba, thread_id fijo)
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    # primer mensaje: el AIMessage de finalize
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)

    if not latin:
        return (
            "No identificado",
            id_report,
            [{"role": "assistant", "content": "No se pudo identificar. Sube otra imagen y pulsa **Identificar**."}],
            img_bytes,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**."
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    return latin, id_report, ui_msgs, img_bytes, extra_ctx, state_out.get("messages", []), "test1"

def redo_identify(last_image, prev_thread_id):
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1"

    state_in = {"messages": [], "image_bytes": last_image}
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    if not latin:
        return (
            "No identificado",
            id_report,
            [{"role": "assistant", "content": "No se pudo identificar. Prueba con otra imagen."}],
            last_image,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**."
    return latin, id_report, [{"role": "assistant", "content": first_msg}], last_image, extra_ctx, state_out.get("messages", []), "test1"

def do_chat(user_msg, current_taxon, ui_messages, id_report, extra_ctx, lc_messages, thread_id):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages
    if not current_taxon or current_taxon.startswith("("):
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Primero identifica una especie."}
        ], "", lc_messages

    # guard si no hay clave de OpenAI → no colgamos
    if not os.getenv("OPENAI_API_KEY"):
        msg = "QA deshabilitado: falta `OPENAI_API_KEY` en el entorno."
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": msg}
        ], "", lc_messages

    lc_hist = list(lc_messages) + [HumanMessage(content=user_msg)]
    state_in = {"messages": lc_hist, "current_taxon": current_taxon, "context_md": build_context_md(extra_ctx)}

    try:
        state_out = qa_node(state_in)
    except Exception as e:
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"Error en QA: {e}"}
        ], "", lc_hist

    last_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    answer = (last_ai.content if last_ai else "").strip() or "No he podido generar respuesta de QA ahora mismo."
    new_ui = ui_messages + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", state_out.get("messages", [])

def reset_all():
    return "(Sube una imagen)", {}, [], None, {}, [], "test1"

# ───────────────────────── UI (Gradio) ─────────────────────────
with gr.Blocks(title="MonoAgent · Identificación + QA") as demo:
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Sube imagen", type="pil")
            btn_identify = gr.Button("🔍 Identificar")
            btn_reidentify = gr.Button("↻ Re-identificar")
            btn_reset = gr.Button("🧹 Reiniciar")
        with gr.Column():
            current_taxon = gr.Label(value="(Sube una imagen)", label="Especie identificada")
            id_report = gr.JSON(label="Detalle de predicción")

    chat = gr.Chatbot(label="Preguntas sobre la especie", type="messages", height=420)
    user_box = gr.Textbox(placeholder="Tu pregunta…", label="Tu pregunta")
    btn_ask = gr.Button("Enviar")

    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_chat_msgs = gr.State([])
    st_lc = gr.State([])
    st_thread = gr.State("test1")

    btn_identify.click(
        do_identify, [image_in],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread]
    ).then(lambda m: m, st_chat_msgs, chat)

    btn_reidentify.click(
        redo_identify, [st_last_image, st_thread],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread]
    ).then(lambda m: m, st_chat_msgs, chat)

    btn_ask.click(
        do_chat, [user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread],
        [st_chat_msgs, user_box, st_lc]
    ).then(lambda m: m, st_chat_msgs, chat)

    user_box.submit(
        do_chat, [user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread],
        [st_chat_msgs, user_box, st_lc]
    ).then(lambda m: m, st_chat_msgs, chat)

    btn_reset.click(
        reset_all, [],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread]
    ).then(lambda m: m, st_chat_msgs, chat)

if __name__ == "__main__":
    demo.launch()
