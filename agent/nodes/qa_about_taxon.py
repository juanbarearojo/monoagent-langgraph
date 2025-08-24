# agent/nodes/clarify.py
from __future__ import annotations
from typing import Any, Dict, List
from datetime import datetime
from langchain_core.messages import AIMessage

# Opcional: mapa de errores "técnicos" a textos amigables
_IMAGE_ERROR_MAP = {
    "too_large": "La imagen es demasiado grande. Prueba con un archivo < 5 MB.",
    "unsupported_format": "Formato no soportado. Acepto JPG o PNG.",
    "download_failed": "No pude descargar la imagen desde la URL proporcionada.",
    "empty_bytes": "No recibí datos de imagen válidos.",
    "corrupted": "La imagen parece estar corrupta o incompleta.",
}

def _collect_reasons(state: Dict[str, Any]) -> List[str]:
    tmp = state.get("_tmp", {}) or {}
    reasons: List[str] = []

    # 1) Validación de imagen
    image_ok = tmp.get("image_ok")
    image_error = tmp.get("image_error")
    if image_ok is False:
        if image_error and image_error in _IMAGE_ERROR_MAP:
            reasons.append(_IMAGE_ERROR_MAP[image_error])
        else:
            reasons.append("La imagen no es válida o no pude procesarla correctamente.")

    # 2) Inferencia local con confianza baja / entropía alta
    if tmp.get("low_confidence") or tmp.get("high_entropy"):
        reasons.append("La predicción local no fue lo suficientemente confiable (incertidumbre elevada).")

    # 3) Fallbacks / visión externa
    vision_status = tmp.get("vision_status")  # p.ej. "ok", "no_answer", "error"
    if vision_status in ("no_answer", "error"):
        reasons.append("El verificador externo de visión no pudo confirmar la especie.")

    # 4) Mapeo de etiqueta a nombre científico
    if tmp.get("need_fallback"):
        reasons.append("No pude mapear la etiqueta a un nombre científico de forma segura.")

    # 5) Seguridad: si no hay ninguna razón explícita, da una genérica
    if not reasons:
        reasons.append("No pude identificar la especie con suficiente calidad.")

    return reasons

def _make_tips(state: Dict[str, Any]) -> str:
    """Consejos para una mejor imagen."""
    taxon = state.get("current_taxon")
    base = [
        "Rostro visible y sin oclusiones.",
        "Buena iluminación natural (evita contraluces).",
        "Fondo relativamente simple y poco ruido visual.",
        "El sujeto debe ocupar una porción significativa del encuadre.",
        "Evita desenfoque: apoya el móvil o usa ambos codos para estabilizar.",
    ]
    if taxon:
        base.insert(0, f"Si puedes, intenta capturar rasgos distintivos de **{taxon}** (cara, pelaje, orejas).")
    return "\n".join(f"- {t}" for t in base)

def clarify_or_fail(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Genera un mensaje de aclaración/fallo con:
      - Motivos detectados
      - Siguiente acción solicitada (subir imagen válida)
      - Consejos para mejorar la captura
      - Requisitos de formato/tamaño
    Deja trazas en _tmp para la UI/analítica.
    """
    reasons = _collect_reasons(state)
    tips_md = _make_tips(state)
    taxon = state.get("current_taxon")
    user_binomial = state.get("_tmp", {}).get("user_binomial")

    # Encabezado contextual
    if taxon:
        hdr = f"He registrado el taxón **{taxon}**, pero necesito una imagen válida para confirmar."
    elif user_binomial:
        hdr = f"He detectado el nombre científico **{user_binomial}**, pero necesito una imagen para confirmar."
    else:
        hdr = "Para continuar necesito **una imagen válida del primate**."

    # Construcción del mensaje final en Markdown
    msg_lines = [
        hdr,
        "",
        "### ¿Qué ha pasado?",
        *(f"- {r}" for r in reasons),
        "",
        "### Qué necesito ahora",
        "- Sube una **imagen en JPG o PNG** (ideal < 5 MB).",
        "- Si usas URL, asegúrate de que es pública y directa al archivo.",
        "",
        "### Consejos para una mejor foto",
        tips_md,
    ]

    # Requisitos explícitos
    reqs = [
        "- **Formatos**: JPG, PNG",
        "- **Tamaño**: recomendable < 5 MB",
        "- **Resolución**: ≥ 512 px en el lado corto"
    ]
    msg_lines += ["", "### Requisitos mínimos", *reqs]

    text = "\n".join(msg_lines)

    # Merge del estado: no pisar, solo añadir/actualizar
    new_messages = list(state.get("messages", [])) + [AIMessage(content=text)]
    new_tmp = {
        **(state.get("_tmp", {}) or {}),
        "awaiting_image": True,
        "fail_reason": "; ".join(reasons),
        "last_clarify_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    return {
        **state,
        "messages": new_messages,
        "_tmp": new_tmp,
    }
