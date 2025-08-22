# tests/test_node_ddg_search.py
import importlib

def test_retrieve_ddg_fills_tmp(monkeypatch):
    # Mock directo de la tool para control total
    fake_payload = {
        "status": "ok",
        "top_snippet": "From IUCN and ADW.",
        "results": [
            {"title": "Macaca fuscata – IUCN Red List", "url": "https://www.iucnredlist.org/", "snippet": "Endangered..."},
            {"title": "Macaca fuscata – ADW", "url": "https://animaldiversity.org/", "snippet": "Natural history..."},
        ],
    }

    ddg_tools = importlib.import_module("agent.tools.ddg")
    monkeypatch.setattr(ddg_tools, "search_summary", lambda latin: fake_payload)

    node = importlib.import_module("agent.nodes.ddg_search")
    # El nodo importa search_summary al cargar; reimportamos para refrescar el monkeypatch
    importlib.reload(node)

    state = {"_tmp.latin_name": "Macaca fuscata"}
    out = node.retrieve_ddg(state)

    assert out["_tmp.ddg"]["status"] == "ok"
    assert out["_tmp.ddg_status"] == "ok"
    assert out["_tmp.ddg_snippet"] == "From IUCN and ADW."
    assert len(out["_tmp.ddg_results"]) == 2

def test_retrieve_ddg_handles_missing_taxon():
    node = importlib.import_module("agent.nodes.ddg_search")
    state = {}
    out = node.retrieve_ddg(state)
    assert out["_tmp.ddg"]["status"] in {"empty", "error"}
    assert out["_tmp.ddg_results"] == []
    assert out["_tmp.ddg_snippet"] == ""

def test_merge_context_with_ddg(monkeypatch):
    # Preparamos DDG en state y verificamos que merge_context lo consuma
    merge = importlib.import_module("agent.nodes.merge_context")

    ddg_payload = {
        "status": "ok",
        "top_snippet": "From IUCN and ADW.",
        "results": [
            {"title": "Macaca fuscata – IUCN Red List", "url": "https://www.iucnredlist.org/", "snippet": "Endangered..."},
            {"title": "Macaca fuscata – ADW", "url": "https://animaldiversity.org/", "snippet": "Natural history..."},
        ],
    }

    state = {
        "current_taxon": "Macaca fuscata",
        "wiki": {
            "title": "Macaca fuscata",
            "url": "https://en.wikipedia.org/wiki/Japanese_macaque",
            "plain_text": "The Japanese macaque is a terrestrial Old World monkey species...",
            "infobox": {"Status": "LC", "Family": "Cercopithecidae"},
        },
        "_tmp.ddg": ddg_payload,
    }

    out = merge.merge_context(state)
    ctx = out.get("_tmp.context", "")
    assert "Macaca fuscata | https://en.wikipedia.org/wiki/Japanese_macaque" in ctx
    assert "<INFOBOX>" in ctx and "Family: Cercopithecidae" in ctx
    assert "<WIKIPEDIA>" in ctx and "Japanese macaque" in ctx
    assert "<WEB_SNIPPETS>" in ctx and "IUCN Red List" in ctx
    assert out["_tmp.context_status"] == "ok"
