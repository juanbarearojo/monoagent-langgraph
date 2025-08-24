# tests/test_map_scientific.py
import importlib

mod = importlib.import_module("agent.nodes.map_scientific")
map_to_scientific_name = mod.map_to_scientific_name

def test_exact_label_maps_to_binomial():
    state = {"_tmp.pred_label": "Japanese_macaque"}
    out = map_to_scientific_name(state)
    assert out["_tmp.latin_name"] == "Macaca fuscata"
    assert out["_tmp.latin_source"] == "map"

def test_normalized_label_with_spaces():
    state = {"_tmp.pred_label": "Common squirrel monkey"}  # con espacios
    out = map_to_scientific_name(state)
    assert out["_tmp.latin_name"] == "Saimiri sciureus"

def test_unknown_label_triggers_fallback():
    state = {"_tmp.pred_label": "Unknown_monkey"}
    out = map_to_scientific_name(state)
    assert out.get("_tmp.need_fallback") is True
    assert out.get("_tmp.latin_name") is None

# --- Ejecutable como script ---
if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main(["-q", "-s", __file__]))
