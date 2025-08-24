# agent/tools/gpt.py
from __future__ import annotations
from typing import Any, Dict, Optional
import base64
import os
import re

from openai import OpenAI

try:
    from agent.prompts import PROMPT_BINOMIAL
except Exception:
    PROMPT_BINOMIAL = 'Devuelve solo el nombre científico binomial en formato "Genus species".'

_BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+ [a-z]+$")

def _valid_binomial(s: str) -> bool:
    return bool(s and _BINOMIAL_RE.match(s.strip()))

def _bytes_to_data_url(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def ask_binomial(
    image_bytes: Optional[bytes] = None,
    url: Optional[str] = None,
    prompt: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Llama a OpenAI Chat Completions con entrada multimodal (texto + imagen).
    Devuelve: { 'latin_name': str, 'status': 'ok|invalid|empty|error', 'error'?: str }
    Requiere: OPENAI_API_KEY
    """
    if not image_bytes and not (url and str(url).strip()):
        return {"status": "empty", "latin_name": ""}

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "error", "latin_name": "", "error": "OPENAI_API_KEY missing"}

    client = OpenAI(api_key=api_key)

    # Prepara image_url (data URL si son bytes)
    image_url = url.strip() if (url and str(url).strip()) else _bytes_to_data_url(image_bytes)  # type: ignore[arg-type]
    prompt = prompt or PROMPT_BINOMIAL

    # Chat Completions: contenido multimodal estable
    # https://platform.openai.com/docs/guides/vision (estructura messages -> content list)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return {"status": "error", "latin_name": "", "error": str(e)}

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"status": "error", "latin_name": "", "error": f"no_content: {e}"}

    if not _valid_binomial(text):
        return {"status": "invalid", "latin_name": "", "error": f"model_output='{text}'"}

    return {"status": "ok", "latin_name": text}


# agent/tools/gpt.py (añade al final)

def ask_gpt_text(
    prompt: str,
    model: str = "gpt-4.1-mini",   # o gpt-4o-mini si prefieres
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> Dict[str, Any]:
    """
    Llama a OpenAI Chat Completions con solo texto.
    Devuelve: { 'answer': str, 'status': 'ok|error', 'error'?: str }
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return {"status": "error", "answer": "", "error": "OPENAI_API_KEY missing"}

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        return {"status": "error", "answer": "", "error": str(e)}

    try:
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return {"status": "error", "answer": "", "error": f"no_content: {e}"}

    return {"status": "ok", "answer": text}

