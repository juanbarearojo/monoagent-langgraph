# tests/test_ensure_image.py
import importlib

ensure_mod = importlib.import_module("agent.nodes.ensure_image")
ensure_image = ensure_mod.ensure_image

def test_returns_ok_when_image_bytes_present():
    state = {"image_bytes": b"\x89PNG\x00\x00"}
    out = ensure_image(state)  # parcial
    assert out["_tmp.image_ok"] is True
    assert "image_bytes" not in out  # no sobreescribe si ya hay bytes

def test_converts_pil_image_to_bytes():
    try:
        from PIL import Image
    except Exception:
        # Si no hay PIL instalado, la prueba no aplica
        import pytest
        pytest.skip("Pillow not installed")

    img = Image.new("RGB", (8, 8), color=(10, 20, 30))
    state = {"image": img}
    out = ensure_image(state)
    assert out["_tmp.image_ok"] is True
    assert isinstance(out.get("image_bytes"), (bytes, bytearray))
    assert len(out["image_bytes"]) > 0

def test_downloads_from_url_with_monkeypatch(monkeypatch):
    # Evitamos red real: simulamos descarga
    fake_bytes = b"\xff\xd8\xff\xdb"  # JPEG magic
    utils_mod = importlib.import_module("agent.nodes.ensure_image")

    def fake_download(url: str, timeout: float = 20.0) -> bytes:
        assert url == "https://example.com/test.jpg"
        return fake_bytes

    monkeypatch.setattr(utils_mod, "_download_to_bytes", fake_download)

    state = {"image_url": "https://example.com/test.jpg"}
    out = ensure_image(state)
    assert out["_tmp.image_ok"] is True
    assert out.get("image_bytes") == fake_bytes

def test_returns_false_when_no_image_anywhere():
    state = {}
    out = ensure_image(state)
    assert out["_tmp.image_ok"] is False

# --- Ejecutable como script ---
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main(["-q", "-s", __file__]))
