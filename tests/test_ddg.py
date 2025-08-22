# tests/test_ddg.py
import os
import importlib
from typing import List, Dict

def test_search_summary_seed_without_net(monkeypatch):
    # Aseguramos que no se use red
    monkeypatch.setenv("DDG_ENABLE_NET", "0")
    ddg = importlib.import_module("agent.tools.ddg")

    out = ddg.search_summary("Macaca fuscata")
    assert out["status"] == "ok"
    assert isinstance(out["results"], list) and len(out["results"]) >= 1
    assert "IUCN" in out["results"][0]["title"]

def test_search_summary_empty_input(monkeypatch):
    monkeypatch.setenv("DDG_ENABLE_NET", "0")
    ddg = importlib.import_module("agent.tools.ddg")

    out = ddg.search_summary("")
    assert out["status"] == "empty"
    assert out["results"] == []

def test_search_summary_with_net_mock(monkeypatch):
    # Fuerza camino "net" y moquea _net_search
    monkeypatch.setenv("DDG_ENABLE_NET", "1")
    ddg = importlib.import_module("agent.tools.ddg")

    def fake_net(query: str, max_results: int = 5, timeout: float = 5.0) -> List[Dict[str, str]]:
        return [
            {"title": "Macaca fuscata - IUCN Red List", "url": "https://www.iucnredlist.org/species/xxxxx", "snippet": "Endangered..."},
            {"title": "Macaca fuscata - ADW", "url": "https://animaldiversity.org/accounts/Macaca_fuscata/", "snippet": "Natural history..."},
        ]

    monkeypatch.setattr(ddg, "_net_search", fake_net)
    out = ddg.search_summary("Macaca fuscata")
    assert out["status"] == "ok"
    assert len(out["results"]) == 2
    assert "IUCN" in out["top_snippet"] or "trusted" in out["top_snippet"]
