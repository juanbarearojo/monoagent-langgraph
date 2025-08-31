# app.py - Interfaz Mejorada
from typing import Any, Dict, List, Optional, Tuple
import io, os

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# ───────────────────────── Proyecto (ajusta si difiere) ─────────────────
from agent.graph import build_graph                          # grafo determinista
from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node

# ───────────────────────── Langfuse (estricto, sin try/except) ─────────
from langfuse.langchain import CallbackHandler

# valida presencia explícita
_missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"[Langfuse] Faltan variables de entorno: {_missing}")
handler = CallbackHandler()

# ───────────────────────── Helpers (sin cambios) ─────────────────────────
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

# ───────────────────────── Callbacks (sin cambios funcionales) ─────────────────────────
def do_identify(image):
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1", "", ""

    buf = io.BytesIO(); image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in = {"messages": [], "image_bytes": img_bytes}

    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)

    if not latin:
        return (
            "❌ No identificado",
            id_report,
            [{"role": "assistant", "content": "🔍 No se pudo identificar la especie. Intenta con otra imagen más clara."}],
            img_bytes,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
            "⚠️ Identificación fallida",
            "Prueba con una imagen de mejor calidad o con mejor iluminación."
        )

    # Extraer información adicional para mostrar
    confidence = id_report.get("entropy", 0)
    confidence_text = "Alta" if confidence < 0.5 else "Media" if confidence < 1.0 else "Baja"
    
    first_msg = finalize_ai.content if finalize_ai else f"✅ Identificado como **{latin}**"
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    
    status_title = f"🎯 Identificación exitosa"
    status_desc = f"Especie: {latin} | Confianza: {confidence_text}"
    
    return latin, id_report, ui_msgs, img_bytes, extra_ctx, state_out.get("messages", []), "test1", status_title, status_desc

def redo_identify(last_image, prev_thread_id):
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1", "", ""

    state_in = {"messages": [], "image_bytes": last_image}
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    latin, id_report, extra_ctx = normalize_id_output(state_out)
    finalize_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    
    if not latin:
        return (
            "❌ No identificado",
            id_report,
            [{"role": "assistant", "content": "🔍 No se pudo identificar la especie. Prueba con otra imagen."}],
            last_image,
            extra_ctx,
            state_out.get("messages", []),
            "test1",
            "⚠️ Identificación fallida",
            "Prueba con una imagen diferente."
        )

    confidence = id_report.get("entropy", 0)
    confidence_text = "Alta" if confidence < 0.5 else "Media" if confidence < 1.0 else "Baja"
    
    first_msg = finalize_ai.content if finalize_ai else f"✅ Re-identificado como **{latin}**"
    status_title = f"🔄 Re-identificación exitosa"
    status_desc = f"Especie: {latin} | Confianza: {confidence_text}"
    
    return latin, id_report, [{"role": "assistant", "content": first_msg}], last_image, extra_ctx, state_out.get("messages", []), "test1", status_title, status_desc

def do_chat(user_msg, current_taxon, ui_messages, id_report, extra_ctx, lc_messages, thread_id):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages
    if not current_taxon or current_taxon.startswith("(") or current_taxon.startswith("❌"):
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "⚠️ Primero necesitas identificar una especie para poder hacer preguntas sobre ella."}
        ], "", lc_messages

    if not os.getenv("OPENAI_API_KEY"):
        msg = "🔧 Sistema de preguntas deshabilitado: falta configuración de OpenAI."
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
            {"role": "assistant", "content": f"❌ Error procesando tu pregunta: {str(e)[:100]}..."}
        ], "", lc_hist

    last_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    answer = (last_ai.content if last_ai else "").strip() or "🤔 No pude generar una respuesta en este momento. Intenta reformular tu pregunta."
    new_ui = ui_messages + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", state_out.get("messages", [])

def reset_all():
    return "(Sube una imagen)", {}, [], None, {}, [], "test1", "🆕 Sesión reiniciada", "Sube una nueva imagen para comenzar"

# ───────────────────────── CSS Personalizado ─────────────────────────
custom_css = """
/* Tema principal */
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Contenedor principal */
.main-container {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin: 20px;
    padding: 30px;
}

/* Header */
.app-header {
    text-align: center;
    margin-bottom: 30px;
    padding: 20px;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 15px;
    color: white;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
}

.app-title {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
}

.app-subtitle {
    font-size: 1.1rem;
    margin: 10px 0 0 0;
    opacity: 0.9;
}

/* Botones principales */
.primary-btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
}

.secondary-btn {
    background: linear-gradient(135deg, #f093fb, #f5576c) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(245, 87, 108, 0.4) !important;
}

.secondary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(245, 87, 108, 0.6) !important;
}

/* Área de imagen */
.image-upload {
    border: 3px dashed #667eea !important;
    border-radius: 15px !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1)) !important;
    transition: all 0.3s ease !important;
}

.image-upload:hover {
    border-color: #764ba2 !important;
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
}

/* Panel de resultados */
.result-panel {
    background: linear-gradient(135deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
    border-radius: 15px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    padding: 20px;
    margin: 15px 0;
}

/* Chat */
.chat-container {
    background: white;
    border-radius: 15px;
    border: 1px solid rgba(102, 126, 234, 0.2);
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

/* Status cards */
.status-card {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    color: white;
    border-radius: 12px;
    padding: 15px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(79, 172, 254, 0.3);
    margin: 10px 0;
}

.status-card.success {
    background: linear-gradient(135deg, #43e97b, #38f9d7);
}

.status-card.warning {
    background: linear-gradient(135deg, #fa709a, #fee140);
}

.status-card.error {
    background: linear-gradient(135deg, #ff6b6b, #ee5a52);
}

/* Inputs */
.input-field {
    border-radius: 12px !important;
    border: 2px solid rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
    font-size: 1rem !important;
}

.input-field:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Animaciones */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.processing {
    animation: pulse 2s infinite;
}

/* Responsive */
@media (max-width: 768px) {
    .app-title {
        font-size: 2rem;
    }
    
    .main-container {
        margin: 10px;
        padding: 20px;
    }
}
"""

# ───────────────────────── UI Mejorada (Gradio) ─────────────────────────
with gr.Blocks(
    title="🔬 BioIdentify AI - Identificación Inteligente de Especies",
    css=custom_css,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    )
) as demo:
    
    # Header
    gr.HTML("""
        <div class="app-header">
            <h1 class="app-title">🔬 BioIdentify AI</h1>
            <p class="app-subtitle">Identificación inteligente de especies con IA avanzada</p>
        </div>
    """)
    
    with gr.Row():
        # Columna izquierda - Carga y identificación
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Carga tu imagen")
            image_in = gr.Image(
                label="Arrastra o selecciona una imagen",
                type="pil",
                elem_classes=["image-upload"],
                height=300
            )
            
            with gr.Row():
                btn_identify = gr.Button(
                    "🔍 Identificar Especie",
                    variant="primary",
                    elem_classes=["primary-btn"],
                    scale=2
                )
                btn_reidentify = gr.Button(
                    "🔄 Re-identificar",
                    variant="secondary", 
                    elem_classes=["secondary-btn"],
                    scale=1
                )
            
            btn_reset = gr.Button(
                "🧹 Nueva Sesión",
                variant="stop",
                elem_classes=["secondary-btn"]
            )
            
            # Status card
            status_display = gr.HTML(
                value="""
                <div class="status-card">
                    <strong>🆕 Sesión iniciada</strong><br>
                    <small>Sube una imagen para comenzar</small>
                </div>
                """,
                elem_classes=["status-card"]
            )
        
        # Columna derecha - Resultados
        with gr.Column(scale=1):
            gr.Markdown("### 🎯 Resultados de identificación")
            
            with gr.Group(elem_classes=["result-panel"]):
                current_taxon = gr.Label(
                    value="(Sube una imagen)",
                    label="🏷️ Especie identificada",
                    show_label=True
                )
                
                with gr.Accordion("📊 Detalles técnicos", open=False):
                    id_report = gr.JSON(
                        label="Información detallada",
                        show_label=False
                    )
    
    # Sección de chat
    gr.Markdown("### 💬 Pregunta sobre la especie identificada")
    
    with gr.Group(elem_classes=["chat-container"]):
        chat = gr.Chatbot(
            label="",
            type="messages",
            height=400,
            show_label=False,
            avatar_images=(
                "https://cdn-icons-png.flaticon.com/512/1077/1077114.png",  # Usuario
                "https://cdn-icons-png.flaticon.com/512/4712/4712027.png"   # Bot
            )
        )
        
        with gr.Row():
            user_box = gr.Textbox(
                placeholder="Escribe tu pregunta sobre la especie identificada...",
                label="",
                show_label=False,
                scale=4,
                elem_classes=["input-field"]
            )
            btn_ask = gr.Button(
                "➤ Enviar",
                variant="primary",
                scale=1,
                elem_classes=["primary-btn"]
            )

    # Estados (hidden)
    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_chat_msgs = gr.State([])
    st_lc = gr.State([])
    st_thread = gr.State("test1")
    st_status_title = gr.State("")
    st_status_desc = gr.State("")

    # Función para actualizar status
    def update_status_display(title, desc):
        if "exitosa" in title:
            card_class = "success"
        elif "fallida" in title or "Error" in title:
            card_class = "error"
        elif "reiniciada" in title:
            card_class = "warning"
        else:
            card_class = ""
            
        return f"""
        <div class="status-card {card_class}">
            <strong>{title}</strong><br>
            <small>{desc}</small>
        </div>
        """

    # Event handlers
    btn_identify.click(
        do_identify, [image_in],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_status_title, st_status_desc]
    ).then(
        lambda m: m, st_chat_msgs, chat
    ).then(
        update_status_display, [st_status_title, st_status_desc], status_display
    )

    btn_reidentify.click(
        redo_identify, [st_last_image, st_thread],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_status_title, st_status_desc]
    ).then(
        lambda m: m, st_chat_msgs, chat
    ).then(
        update_status_display, [st_status_title, st_status_desc], status_display
    )

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
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_status_title, st_status_desc]
    ).then(
        lambda m: m, st_chat_msgs, chat
    ).then(
        update_status_display, [st_status_title, st_status_desc], status_display
    )

    # Footer
    gr.HTML("""
        <div style="text-align: center; margin-top: 30px; padding: 20px; background: rgba(102, 126, 234, 0.1); border-radius: 15px;">
            <p style="margin: 0; color: #667eea; font-weight: 500;">
                🌿 Powered by Advanced AI • Identificación precisa de especies • 🔬 BioIdentify AI
            </p>
        </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )