# tests/test_router_input.py
import importlib

router_mod = importlib.import_module("agent.nodes.router")
router_input = router_mod.router_input

def test_route_classify_when_image_bytes():
    state = {"image_bytes": b"\x00\x01"}
    out = router_input(state)
    assert out["_tmp.route"] == "classify"

def test_route_classify_when_image_url():
    state = {"image_url": "https://example.com/img.jpg"}
    out = router_input(state)
    assert out["_tmp.route"] == "classify"

def test_route_qa_when_taxon_no_image():
    state = {"current_taxon": "Macaca fuscata"}
    out = router_input(state)
    assert out["_tmp.route"] == "qa"

def test_route_ask_image_when_no_image_no_taxon():
    state = {}
    out = router_input(state)
    assert out["_tmp.route"] == "ask_image"

def test_force_route_overrides_everything_classify():
    state = {"force_route": "classify"}
    out = router_input(state)
    assert out["_tmp.route"] == "classify"

def test_force_route_overrides_even_if_image_present():
    state = {"force_route": "qa", "image_bytes": b"\xAA\xBB"}
    out = router_input(state)
    assert out["_tmp.route"] == "qa"

def test_force_route_ask_image():
    state = {"force_route": "ask_image", "current_taxon": "Macaca fuscata"}
    out = router_input(state)
    assert out["_tmp.route"] == "ask_image"

# --- Ejecutable directamente como script o módulo ---
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main(["-q", "-s", __file__]))
