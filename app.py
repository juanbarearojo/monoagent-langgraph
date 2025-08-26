# app.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import io
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage

# ───────────────────────── Proyecto (ajusta rutas si difieren) ─────────────────
try:
    from agent.graph import build_graph                 # tu grafo determinista
except Exception:
    def build_graph():
        raise NotImplementedError("Conecta tu build_graph() real desde agent.graph")

try:
    from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node
except Exception:
    def qa_node(state: Dict[str, Any]) -> Dict[str, Any]:
        msgs = list(state.get("messages", [])) + [AIMessage(content="[QA placeholder]")]
        return {**state, "messages": msgs, "_tmp": {**state.get("_tmp", {}), "qa_answered": True}}

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

_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ───────────────────────── Callbacks ─────────────────────────
def do_identify(image) -> Tuple[str, Dict[str, Any], List[tuple], bytes, Dict[str, Any], List[Any]]:
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, []

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in: Dict[str, Any] = {"messages": [], "image_bytes": img_bytes}
    state_out = get_graph().invoke(state_in)

    latin, id_report, extra_ctx = normalize_id_output(state_out)

    if not latin:
        return "No identificado (repite o sube otra imagen)", {}, [], img_bytes, extra_ctx, []

    chat_pairs = [(None, f"Identificado: **{latin}**. Pregunta sobre hábitat, dieta, distribución, conservación…")]
    lc_messages: List[Any] = [AIMessage(content=f"Especie identificada: {latin}. ¿En qué puedo ayudarte?")]
    return latin, id_report, chat_pairs, img_bytes, extra_ctx, lc_messages

def redo_identify(last_image: Optional[bytes]) -> Tuple[str, Dict[str, Any], List[tuple], bytes, Dict[str, Any], List[Any]]:
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, []
    state_in = {"messages": [], "image_bytes": last_image}
    state_out = get_graph().invoke(state_in)
    latin, id_report, extra_ctx = normalize_id_output(state_out)
    if not latin:
        return "No identificado (repite o sube otra imagen)", {}, [], last_image, extra_ctx, []
    chat_pairs = [(None, f"Identificado: **{latin}**. ¡Pregunta lo que quieras!")]
    lc_messages = [AIMessage(content=f"Especie identificada: {latin}. ¿En qué puedo ayudarte?")]
    return latin, id_report, chat_pairs, last_image, extra_ctx, lc_messages

def do_chat(user_msg: str,
            current_taxon: str,
            chat_pairs: List[tuple],
            id_report: Dict[str, Any],
            extra_ctx: Dict[str, Any],
            lc_messages: List[Any]) -> Tuple[List[tuple], str, List[Any]]:
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return chat_pairs, "", lc_messages

    if not current_taxon or current_taxon.startswith("(") or current_taxon.startswith("No identificado"):
        tip = "Primero identifica una especie: sube imagen y pulsa **Identificar**."
        return chat_pairs + [(user_msg, tip)], "", lc_messages

    lc_hist = list(lc_messages) + [HumanMessage(content=user_msg)]
    state_in: Dict[str, Any] = {
        "messages": lc_hist,
        "current_taxon": current_taxon,
        "context_md": build_context_md(extra_ctx),
    }

    try:
        state_out = qa_node(state_in)
    except Exception as e:
        answer = f"Ocurrió un error en QA: {type(e).__name__}: {e}"
        return chat_pairs + [(user_msg, answer)], "", lc_hist

    out_msgs = state_out.get("messages", [])
    last_ai = next((m for m in reversed(out_msgs) if isinstance(m, AIMessage)), None)
    answer = last_ai.content if last_ai else "(QA no devolvió respuesta)"

    new_pairs = chat_pairs + [(user_msg, answer)]
    return new_pairs, "", out_msgs

def reset_all():
    return "(Sube una imagen)", {}, [], None, {}, [], ""

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
            # ✅ aquí el cambio importante
            chat = gr.Chatbot(label="Preguntas sobre la especie", type="tuple", height=420)
            user_box = gr.Textbox(placeholder="Escribe tu pregunta…", label="Tu pregunta")
            btn_ask = gr.Button("Enviar")

    # Estados
    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_pairs = gr.State([])
    st_lc = gr.State([])

    btn_identify.click(
        fn=do_identify,
        inputs=[image_in],
        outputs=[current_taxon, id_report, st_pairs, st_last_image, st_extra_ctx, st_lc],
        show_progress="minimal",
    ).then(lambda p: p, inputs=[st_pairs], outputs=[chat])

    btn_reidentify.click(
        fn=redo_identify,
        inputs=[st_last_image],
        outputs=[current_taxon, id_report, st_pairs, st_last_image, st_extra_ctx, st_lc],
        show_progress="minimal",
    ).then(lambda p: p, inputs=[st_pairs], outputs=[chat])

    btn_ask.click(
        fn=do_chat,
        inputs=[user_box, current_taxon, st_pairs, id_report, st_extra_ctx, st_lc],
        outputs=[st_pairs, user_box, st_lc],
        show_progress="minimal",
    ).then(lambda p: p, inputs=[st_pairs], outputs=[chat])

    user_box.submit(
        fn=do_chat,
        inputs=[user_box, current_taxon, st_pairs, id_report, st_extra_ctx, st_lc],
        outputs=[st_pairs, user_box, st_lc],
    ).then(lambda p: p, inputs=[st_pairs], outputs=[chat])

    btn_reset.click(
        fn=reset_all,
        inputs=[],
        outputs=[current_taxon, id_report, st_pairs, st_last_image, st_extra_ctx, st_lc, user_box],
    ).then(lambda p: p, inputs=[st_pairs], outputs=[chat])

if __name__ == "__main__":
    demo.launch()
