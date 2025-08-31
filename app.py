# app.py
from typing import Any, Dict, List, Optional, Tuple
import io, os
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from agent.graph import build_graph                          # grafo determinista
from agent.nodes.qa_about_taxon import qa_about_taxon as qa_node
from langfuse.langchain import CallbackHandler  # si no está instalado, fallará aquí (bien para depurar)


# valida presencia explícita
_missing = [k for k in ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST") if not os.getenv(k)]
if _missing:
    raise RuntimeError(f"[Langfuse] Faltan variables de entorno: {_missing}")
handler = CallbackHandler()  # si las claves son inválidas, verás el error en logs

# ───────────────────────── Helpers ─────────────────────────
def get_langfuse_trace_url(handler: CallbackHandler) -> Optional[str]:
    """Extrae la URL de la traza de Langfuse del handler"""
    try:
        # Método directo si existe trace_id
        if hasattr(handler, 'trace_id') and handler.trace_id:
            langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
            return f"{langfuse_host}/trace/{handler.trace_id}"
        
        # Método alternativo: buscar en el objeto trace
        if hasattr(handler, 'trace') and handler.trace:
            trace_id = getattr(handler.trace, 'id', None)
            if trace_id:
                langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                return f"{langfuse_host}/trace/{trace_id}"
        
        # Otro método: revisar runs recientes
        if hasattr(handler, 'runs') and handler.runs:
            for run_id, run_data in handler.runs.items():
                if hasattr(run_data, 'trace_id'):
                    langfuse_host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
                    return f"{langfuse_host}/trace/{run_data.trace_id}"
        
    except Exception as e:
        print(f"⚠️ Error obteniendo URL Langfuse: {e}")
    
    return None

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

# 🔗 Lista global para guardar todos los enlaces de trazas
_trace_urls = []

def add_trace_url(url: str, operation: str = "identificación"):
    """Añade una nueva URL de traza a la lista global"""
    global _trace_urls
    timestamp = __import__('datetime').datetime.now().strftime("%H:%M:%S")
    _trace_urls.append({
        "timestamp": timestamp,
        "operation": operation,
        "url": url,
        "display": f"🕐 {timestamp} - {operation.title()}"
    })
    return _trace_urls

def get_all_traces():
    """Retorna todas las trazas guardadas"""
    return _trace_urls

def clear_all_traces():
    """Limpia todas las trazas guardadas"""
    global _trace_urls
    _trace_urls = []
    return []

# ───────────────────────── Callbacks ─────────────────────────
def do_identify(image):
    if image is None:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1", []

    buf = io.BytesIO(); image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    state_in = {"messages": [], "image_bytes": img_bytes}

    # Langfuse EXACTO como pediste (handler creado arriba, thread_id fijo)
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    # 🔗 OPCIÓN A: Mostrar en consola
    trace_url = get_langfuse_trace_url(handler)
    if trace_url:
        print(f"🔍 Traza Langfuse: {trace_url}")
        # 🔗 OPCIÓN C: Guardar para historial
        traces_list = add_trace_url(trace_url, "identificación")
    else:
        traces_list = get_all_traces()

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
            traces_list,
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**."
    ui_msgs = [{"role": "assistant", "content": first_msg}]
    return latin, id_report, ui_msgs, img_bytes, extra_ctx, state_out.get("messages", []), "test1", traces_list

def redo_identify(last_image, prev_thread_id):
    if not last_image:
        return "(Sube una imagen)", {}, [], None, {}, [], "test1", get_all_traces()

    state_in = {"messages": [], "image_bytes": last_image}
    state_out = get_graph().invoke(
        state_in,
        config={"callbacks": [handler], "configurable": {"thread_id": "test1"}},
    )

    # 🔗 OPCIÓN A: Mostrar en consola
    trace_url = get_langfuse_trace_url(handler)
    if trace_url:
        print(f"🔍 Traza Langfuse: {trace_url}")
        # 🔗 OPCIÓN C: Guardar para historial
        traces_list = add_trace_url(trace_url, "re-identificación")
    else:
        traces_list = get_all_traces()

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
            traces_list,
        )

    first_msg = finalize_ai.content if finalize_ai else f"Identificado: **{latin}**."
    return latin, id_report, [{"role": "assistant", "content": first_msg}], last_image, extra_ctx, state_out.get("messages", []), "test1", traces_list

def do_chat(user_msg, current_taxon, ui_messages, id_report, extra_ctx, lc_messages, thread_id, traces_list):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return ui_messages, "", lc_messages, traces_list
    if not current_taxon or current_taxon.startswith("("):
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Primero identifica una especie."}
        ], "", lc_messages, traces_list

    # guard si no hay clave de OpenAI → no colgamos
    if not os.getenv("OPENAI_API_KEY"):
        msg = "QA deshabilitado: falta `OPENAI_API_KEY` en el entorno."
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": msg}
        ], "", lc_messages, traces_list

    lc_hist = list(lc_messages) + [HumanMessage(content=user_msg)]
    state_in = {"messages": lc_hist, "current_taxon": current_taxon, "context_md": build_context_md(extra_ctx)}

    try:
        state_out = qa_node(state_in)
    except Exception as e:
        return ui_messages + [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"Error en QA: {e}"}
        ], "", lc_hist, traces_list

    last_ai = next((m for m in reversed(state_out.get("messages", [])) if isinstance(m, AIMessage)), None)
    answer = (last_ai.content if last_ai else "").strip() or "No he podido generar respuesta de QA ahora mismo."
    new_ui = ui_messages + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": answer},
    ]
    return new_ui, "", state_out.get("messages", []), traces_list

def reset_all():
    clear_all_traces()  # 🧹 Limpiar historial de trazas
    return "(Sube una imagen)", {}, [], None, {}, [], "test1", []

# ───────────────────────── CSS Personalizado ─────────────────────────
custom_css = """
/* Tema principal con gradientes y glassmorphism */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Header estilizado */
.main-header {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 20px !important;
    padding: 25px !important;
    margin-bottom: 25px !important;
    text-align: center !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
}

/* Cards con efecto glass */
.glass-card {
    background: rgba(255, 255, 255, 0.15) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
}

/* Botones modernos */
.modern-btn {
    background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 12px 24px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.modern-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.3) !important;
    background: linear-gradient(45deg, #FF5252, #26C6DA) !important;
}

/* Botón de reset especial */
.reset-btn {
    background: linear-gradient(45deg, #FF4081, #9C27B0) !important;
}

.reset-btn:hover {
    background: linear-gradient(45deg, #E91E63, #673AB7) !important;
}

/* Botón re-identificar */
.reidentify-btn {
    background: linear-gradient(45deg, #FFC107, #FF9800) !important;
}

.reidentify-btn:hover {
    background: linear-gradient(45deg, #FFB300, #F57C00) !important;
}

/* Input boxes mejorados */
.modern-input {
    background: rgba(255, 255, 255, 0.2) !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    color: white !important;
    backdrop-filter: blur(10px) !important;
    font-size: 16px !important;
}

.modern-input:focus {
    border-color: #4ECDC4 !important;
    box-shadow: 0 0 0 3px rgba(78, 205, 196, 0.3) !important;
    outline: none !important;
}

.modern-input::placeholder {
    color: rgba(255, 255, 255, 0.7) !important;
}

/* Chatbot mejorado */
.chatbot-container {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37) !important;
}

/* Mensajes del chat */
.message {
    background: rgba(255, 255, 255, 0.15) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    margin: 8px 0 !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Labels y texto */
.gradio-label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3) !important;
}

/* JSON viewer */
.json-viewer {
    background: rgba(0, 0, 0, 0.3) !important;
    border-radius: 15px !important;
    padding: 15px !important;
    font-family: 'Courier New', monospace !important;
    color: #00ff88 !important;
    backdrop-filter: blur(10px) !important;
}

/* Efectos de hover para cards */
.glass-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.5) !important;
    transition: all 0.3s ease !important;
}

/* Animaciones sutiles */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

.float-animation {
    animation: float 6s ease-in-out infinite !important;
}

/* Responsive design */
@media (max-width: 768px) {
    .glass-card {
        margin: 5px !important;
        padding: 15px !important;
    }
    
    .modern-btn {
        padding: 10px 20px !important;
        font-size: 12px !important;
    }
}
"""

# ───────────────────────── UI (Gradio) ─────────────────────────
with gr.Blocks(
    title="🔬 MonoAgent · Identificación + QA", 
    css=custom_css,
    theme=gr.themes.Glass()
) as demo:
    
    # Header principal
    gr.HTML("""
        <div class="main-header float-animation">
            <h1 style="color: white; font-size: 2.5em; margin: 0; text-shadow: 0 4px 8px rgba(0,0,0,0.3);">
                🔬 MonoAgent AI
            </h1>
            <p style="color: rgba(255,255,255,0.9); font-size: 1.2em; margin: 10px 0 0 0; font-weight: 300;">
                Identificación de Especies + Asistente Inteligente
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<div class="glass-card">')
            
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        📸 Subir Imagen
                    </h3>
                </div>
            """)
            
            image_in = gr.Image(
                label="", 
                type="pil",
                elem_classes=["modern-input"],
                height=300
            )
            
            with gr.Row():
                btn_identify = gr.Button(
                    "🔍 Identificar", 
                    elem_classes=["modern-btn"],
                    size="lg"
                )
                btn_reidentify = gr.Button(
                    "↻ Re-identificar", 
                    elem_classes=["modern-btn", "reidentify-btn"],
                    size="lg"
                )
            
            btn_reset = gr.Button(
                "🧹 Reiniciar Todo", 
                elem_classes=["modern-btn", "reset-btn"],
                size="lg",
                variant="secondary"
            )
            
            gr.HTML('</div>')
            
        with gr.Column(scale=1):
            gr.HTML('<div class="glass-card">')
            
            gr.HTML("""
                <div style="text-align: center; margin-bottom: 20px;">
                    <h3 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                        🧬 Resultados
                    </h3>
                </div>
            """)
            
            current_taxon = gr.Label(
                value="(Sube una imagen)", 
                label="🦋 Especie Identificada",
                elem_classes=["glass-card"]
            )
            
            id_report = gr.JSON(
                label="📊 Detalle de Predicción",
                elem_classes=["json-viewer"]
            )
            
            gr.HTML('</div>')

    # Sección de Chat
    gr.HTML("""
        <div class="glass-card" style="margin-top: 20px;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    💬 Pregunta sobre la Especie
                </h3>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 14px;">
                    Haz preguntas sobre la especie identificada
                </p>
            </div>
    """)
    
    chat = gr.Chatbot(
        label="", 
        type="messages", 
        height=420,
        elem_classes=["chatbot-container"],
        avatar_images=("🧑‍🔬", "🤖")
    )
    
    with gr.Row():
        user_box = gr.Textbox(
            placeholder="Escribe tu pregunta aquí... ¿Dónde vive esta especie?", 
            label="",
            elem_classes=["modern-input"],
            scale=4
        )
        btn_ask = gr.Button(
            "📤 Enviar", 
            elem_classes=["modern-btn"],
            scale=1,
            size="lg"
        )
    
                gr.HTML('</div>')

    # 🔗 Sección de Trazabilidad (OPCIÓN C)
    gr.HTML("""
        <div class="glass-card" style="margin-top: 20px;">
            <div style="text-align: center; margin-bottom: 20px;">
                <h3 style="color: white; margin: 0; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">
                    🔗 Trazabilidad Langfuse
                </h3>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; font-size: 14px;">
                    Historial de todas las ejecuciones del grafo
                </p>
            </div>
    """)
    
    traces_display = gr.HTML(
        value="<div class='trace-container'><p style='color: rgba(255,255,255,0.7); text-align: center; margin: 20px;'>No hay trazas aún. Ejecuta una identificación.</p></div>",
        label=""
    )
    
    def update_traces_display(traces_list):
        """Genera HTML para mostrar las trazas"""
        if not traces_list:
            return "<div class='trace-container'><p style='color: rgba(255,255,255,0.7); text-align: center; margin: 20px;'>No hay trazas aún. Ejecuta una identificación.</p></div>"
        
        html_content = "<div class='trace-container'>"
        for trace in reversed(traces_list[-10:]):  # Mostrar últimas 10 trazas
            html_content += f"""
                <div class='trace-item'>
                    <span style='color: #00ff88;'>{trace['display']}</span><br>
                    <a href='{trace['url']}' target='_blank' class='trace-link'>
                        🔗 {trace['url']}
                    </a>
                </div>
            """
        html_content += "</div>"
        return html_content
    
    gr.HTML('</div>')

    # Estados (actualizados)
    st_last_image = gr.State(None)
    st_extra_ctx = gr.State({})
    st_chat_msgs = gr.State([])
    st_lc = gr.State([])
    st_thread = gr.State("test1")
    st_traces = gr.State([])  # 🔗 Nuevo estado para trazas

    # Eventos (actualizados con trazas)
    btn_identify.click(
        do_identify, [image_in],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_traces]
    ).then(lambda m: m, st_chat_msgs, chat).then(
        update_traces_display, st_traces, traces_display
    )

    btn_reidentify.click(
        redo_identify, [st_last_image, st_thread],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_traces]
    ).then(lambda m: m, st_chat_msgs, chat).then(
        update_traces_display, st_traces, traces_display
    )

    btn_ask.click(
        do_chat, [user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread, st_traces],
        [st_chat_msgs, user_box, st_lc, st_traces]
    ).then(lambda m: m, st_chat_msgs, chat)

    user_box.submit(
        do_chat, [user_box, current_taxon, st_chat_msgs, id_report, st_extra_ctx, st_lc, st_thread, st_traces],
        [st_chat_msgs, user_box, st_lc, st_traces]
    ).then(lambda m: m, st_chat_msgs, chat)

    btn_reset.click(
        reset_all, [],
        [current_taxon, id_report, st_chat_msgs, st_last_image, st_extra_ctx, st_lc, st_thread, st_traces]
    ).then(lambda m: m, st_chat_msgs, chat).then(
        update_traces_display, st_traces, traces_display
    )

if __name__ == "__main__":
    demo.launch()