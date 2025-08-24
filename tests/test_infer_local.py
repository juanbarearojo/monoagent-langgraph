# tests/test_infer_local_real.py
import importlib
import sys

# Nodos reales
ensure_mod = importlib.import_module("agent.nodes.ensure_image")
infer_mod = importlib.import_module("agent.nodes.infer_local")
map_mod = importlib.import_module("agent.nodes.map_scientific")

ensure_image = ensure_mod.ensure_image
infer_local = infer_mod.infer_local
map_to_scientific_name = map_mod.map_to_scientific_name

# URL fija (thumbnail 250px, rápido; tu pipeline reescala a 224)
IMG_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Macaque_ds.jpg/250px-Macaque_ds.jpg"

def test_infer_local_end_to_end_real_hardcoded():
    """
    Flujo real: ensure_image -> infer_local -> map_to_scientific_name
    Requiere que agent/tools/vision.py tenga el modelo y labels.json en agent/model/.
    """
    # 1) Normalizar a bytes desde URL
    state = {"image_url": IMG_URL, "topk": 5}
    out_ensure = ensure_image(state)
    state.update(out_ensure)

    assert state.get("image_bytes"), "ensure_image no dejó image_bytes"
    assert out_ensure.get("_tmp.image_ok") is True, f"Descarga/normalización fallida: {out_ensure}"

    # 2) Inferencia local real (TorchScript)
    out_infer = infer_local(state)
    print("\n[INFER]", {k: (v if k != "_tmp.topk_list" else f"{len(v)} items") for k, v in out_infer.items()})

    assert "_tmp.error" not in out_infer, f"Error infer_local: {out_infer.get('_tmp.error')}"
    assert isinstance(out_infer.get("_tmp.topk_list"), list) and len(out_infer["_tmp.topk_list"]) >= 1
    assert 0.0 <= float(out_infer.get("_tmp.p1", 0.0)) <= 1.0
    assert 0.0 <= float(out_infer.get("_tmp.p2", 0.0)) <= 1.0

    # 3) Map a nombre científico si hay label top-1
    top1 = out_infer.get("_tmp.pred_label")
    if top1:
        mapped = map_to_scientific_name({"_tmp.pred_label": top1})
        print("[TOP1]", top1, "| [MAP]", mapped)
    else:
        print("[TOP1] sin label; posible fallback a gpt")

# --- Ejecutable como script/módulo ---
if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(["-q", "-s", __file__]))
