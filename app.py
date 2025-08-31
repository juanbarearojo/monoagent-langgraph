# app.py
from typing import Any, Dict, List, Optional, Tuple
import io, os

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ───────────────────────── Project imports ─────────────────────────
from agent.graph import build_graph                          # deterministic graph
from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node

# ───────────────────────── Langfuse (strict, no try/except) ───────
# ⛔️ NOTE: In production, keep these in Secrets/ENV.
from langfuse.langchain import CallbackHandler  # will fail if not installed (good for debugging)

_missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"[Langfuse] Missing environment variables: {_missing}")
handler = CallbackHandler()  # if keys are invalid, errors will appear in logs

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

# Lazy graph singleton
_graph = None
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

# ───────────────────────── Callbacks ─────────────────────────
def do_identify(image):
    if image is None:
        return ("(Upload an image)", {}, [], None, {}, [], "test1")

    buf = io.BytesIO(); image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in = {"messages": [], "image_bytes": img_bytes}

    # Langfuse EXACT as requested (handler created above, fixed thread_id)
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    # first finalize AI message
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)

    if not latin:
        return (
            "Not identified",
            id_report,
            [{"role": "assistant", "content": "Identification failed. Please upload another image and click **Identify**."}],
            img_bytes,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identified: **{latin}**."
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    return latin, id_report, ui_msgs, img_bytes, extra_ctx, state_out.get("messages", []), "test1"

def redo_identify(last_image, prev_thread_id):
    if not last_image:
        return ("(Upload an image)", {}, [], None, {}, [], "test1")

    state_in = {"messages": [], "image_bytes": last_image}
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    if not latin:
        return (
            "Not identified",
            id_report,
            [{"role": "assistant", "content": "Identification failed. Try a different image."}],
            last_image,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identified: **{latin}**."
    return latin, id_report, [{"role": "assistant", "content": first_msg}], last_image, extra_ctx, state_out.get("messages", []), "test1"

def do_chat(user_msg, current_taxon, ui_messages, id_report, extra_ctx, lc_messages, thread_id):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages
    if not current_taxon or current_taxon.startswith("("):
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Please identify a species first."}
        ], "", lc_messages

    # Guard: if no OpenAI key → do not hang
    if not os.getenv("OPENAI_API_KEY"):
        msg = "Q&A disabled: missing `OPENAI_API_KEY` in the environment."
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
            {"role": "assistant", "content": f"Q&A error: {e}"}
        ], "", lc_hist

    last_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    answer = (last_ai.content if last_ai else "").strip() or "I couldn’t generate a Q&A response right now."
    new_ui = ui_messages + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", state_out.get("messages", [])

def reset_all():
    return "(Upload an image)", {}, [], None, {}, [], "test1"

# ───────────────────────── UI (Gradio) ─────────────────────────
theme = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")

CUSTOM_CSS = """
.app-header { display:flex; gap:.75rem; align-items:center; }
.app-badge { font-size:.75rem; padding:.2rem .5rem; border-radius:999px; background:rgba(99,102,241,.12); color:#6366f1; border:1px solid rgba(99,102,241,.2) }
.kpi-card { border:1px solid rgba(148,163,184,.25); border-radius:16px; padding:14px; background:linear-gradient(180deg, rgba(241,245,249,.75), rgba(255,255,255,.8)); }
.kpi-title { font-weight:600; font-size: .9rem; color:#334155; margin-bottom:6px }
.kpi-value { font-size:1.25rem; font-weight:700; color:#0f172a }
.footer { opacity:.8; font-size:.85rem; }
"""

with gr.Blocks(title="MonoAgent · Primate ID + Q&A", theme=theme, css=CUSTOM_CSS) as demo:
    # Header
    gr.Markdown(
        """
<div class="app-header">
  <div style="font-size:1.6rem">🧠🐒 <b>MonoAgent</b></div>
  <span class="app-badge">LangGraph</span>
  <span class="app-badge">Langfuse</span>
</div>
<p style="margin:.25rem 0 0 0; color:#475569">
Upload a primate photo, identify the species, and ask questions with context-aware Q&A.
</p>
<hr style="margin-top:.8rem"/>
        """,
        elem_id="header-md"
    )

    with gr.Row():
        # LEFT: Inputs & Actions
        with gr.Column(scale=5, min_width=320):
            image_in = gr.Image(label="Upload image", type="pil", height=320)
            with gr.Row():
                btn_identify = gr.Button("🔍 Identify", variant="primary", size="lg")
                btn_reidentify = gr.Button("↻ Re-identify", variant="secondary")
                btn_reset = gr.Button("🧹 Reset")

            img_preview = gr.Image(label="Last image", interactive=False, height=220)

            with gr.Accordion("Prediction details", open=False):
                id_report = gr.JSON(label="Model output (debug)")

        # RIGHT: Results & Chat
        with gr.Column(scale=7, min_width=420):
            with gr.Group(elem_classes="kpi-card"):
                current_taxon = gr.Label(value="(Upload an image)", label="Identified species")
                gr.Markdown(
                    """
<div style="font-size:.9rem; color:#475569">
Tip: good, well-lit frontal images (no heavy occlusions) yield better results.
</div>
                    """,
                    elem_id="tip-md"
                )

            chat = gr.Chatbot(
                label="Ask about the species",
                type="messages",
                height=420,
            )
            user_box = gr.Textbox(
                placeholder="Ask about this species (diet, range, behavior...)",
                label="Your question",
            )
            btn_ask = gr.Button("Send")

    # Hidden state (unchanged signatures)
    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_chat_msgs = gr.State([])
    st_lc = gr.State([])
    st_thread = gr.State("test1")

    # Wiring
    btn_identify.click(
        do_identify, [image_in],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread]
    ).then(lambda m: m, st_chat_msgs, chat
    ).then(lambda b: b, st_last_image, img_preview)

    btn_reidentify.click(
        redo_identify, [st_last_image, st_thread],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread]
    ).then(lambda m: m, st_chat_msgs, chat
    ).then(lambda b: b, st_last_image, img_preview)

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
    ).then(lambda m: m, st_chat_msgs, chat
    ).then(lambda b: b, st_last_image, img_preview)

    # Footer
    gr.Markdown(
        """
<hr/>
<div class="footer">
<strong>MonoAgent</strong> • Deterministic LangGraph pipeline with Langfuse tracing.<br/>
Keep credentials in environment variables. For production, configure secrets in your Space/runner.
</div>
        """
    )

if __name__ == "__main__":
    demo.launch()
