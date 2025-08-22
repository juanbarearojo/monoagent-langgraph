# tests/test_gpt_integration.py
import os
import re
import pytest
import requests
from agent.tools.gpt import ask_binomial

pytestmark = pytest.mark.integration

# Imagen estable (macaco japonés)
WIKI_IMG_MACAQUE = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Macaque_ds.jpg/500px-Macaque_ds.jpg"

# Jirafa (subespecie 'angolensis' en la página; el modelo debería devolver binomial de 2 palabras)
WIKI_IMG_GIRAFFE = "https://upload.wikimedia.org/wikipedia/commons/e/e0/Giraffa_camelopardalis_angolensis.jpg"

# Puedes añadir extras por entorno si quieres sin tocar el archivo
OPTIONAL_ANIMALS = []
if os.getenv("IMG_URL_EXTRA"):
    OPTIONAL_ANIMALS.append(("Extra", os.getenv("IMG_URL_EXTRA")))

# Lista final de pruebas por URL (incluye jirafa hardcodeada)
ANIMALS_BY_URL = [
    ("Japanese macaque", WIKI_IMG_MACAQUE),
    ("Giraffe (angolensis)", WIKI_IMG_GIRAFFE),
] + OPTIONAL_ANIMALS

_BINOMIAL_RE = re.compile(r"^[A-Z][a-z]+ [a-z]+$")

def _download(url: str) -> bytes:
    headers = {
        # Cumple la Wikimedia User-Agent Policy: nombre del cliente + contacto
        # Cambia el email/URL por el tuyo si quieres.
        "User-Agent": "MonoAgent-LangGraph/0.1 (+https://example.com/; contact: you@example.com)"
    }
    r = requests.get(url, timeout=20, headers=headers)
    r.raise_for_status()
    return r.content


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("label,url", ANIMALS_BY_URL)
def test_ask_binomial_with_url_real(label, url):
    out = ask_binomial(url=url, model="gpt-4.1-mini")

    # --- Salida para verificación manual ---
    print(f"\n[GPT][URL] Animal={label}  Status={out.get('status')}  Latin='{out.get('latin_name','')}'  URL={url}")

    # --- Asserts mínimos (verificación humana del nombre) ---
    assert out["status"] in {"ok", "invalid", "error"}
    if out["status"] == "ok":
        assert _BINOMIAL_RE.match(out["latin_name"]) or out["latin_name"].count(" ") == 1

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.parametrize("label,url", ANIMALS_BY_URL)
def test_ask_binomial_with_bytes_real(label, url):
    img = _download(url)
    out = ask_binomial(image_bytes=img, model="gpt-4.1-mini")

    # --- Salida para verificación manual ---
    print(f"\n[GPT][BYTES] Animal={label}  Status={out.get('status')}  Latin='{out.get('latin_name','')}'  Bytes={len(img)}")

    # --- Asserts mínimos ---
    assert out["status"] in {"ok", "invalid", "error"}
    if out["status"] == "ok":
        assert _BINOMIAL_RE.match(out["latin_name"]) or out["latin_name"].count(" ") == 1

if __name__ == "__main__":
    import pytest, sys
    # Ejecuta este propio archivo, mostrando los prints
    sys.exit(pytest.main(["-q", "-s", "-m", "integration", __file__]))