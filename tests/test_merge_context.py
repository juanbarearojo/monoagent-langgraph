# tests/test_merge_context.py
from agent.tools.wiki import fetch_fullpage
from agent.nodes.merge_context import merge_context

def test_merge_context_basic():
    state = {}
    # Wiki en inglés (firma fija sin lang)
    page = fetch_fullpage("Macaca fuscata")
    state["wiki"] = page
    # Simula un DDG sencillo (opcional)
    state["ddg"] = {
        "top_snippet": "The Japanese macaque is native to Japan and is known as the snow monkey.",
        "results": [
            {"title": "National Geographic - Snow Monkeys", "url": "https://www.nationalgeographic.com/animals/mammals/facts/japanese-macaque"},
            {"title": "IUCN Red List", "url": "https://www.iucnredlist.org/species/12550/17951081"},
        ],
    }
    state["context_max_chars"] = 6000

    out = merge_context(state)
    ctx = out.get("_tmp.context", "")
    srcs = out.get("_tmp.sources", [])

    print("\n--- Context preview ---\n", (ctx[:500] + "…") if len(ctx) > 500 else ctx)
    print("\nSources:", srcs[:5])

    assert isinstance(ctx, str) and len(ctx) > 300
    assert isinstance(srcs, list) and len(srcs) >= 1
    # Debe contener la URL de Wikipedia
    assert page.get("url") in srcs


# --- Permite ejecutar este archivo directamente con Python ---
if __name__ == "__main__":
    import pytest, sys
    # -q: salida concisa  |  -s: no capturar stdout (muestra prints)
    sys.exit(pytest.main(["-q", "-s", __file__]))
