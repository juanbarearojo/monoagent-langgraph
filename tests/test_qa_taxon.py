# tests/test_qa_taxon_integration.py
import os
import pytest
from langchain_core.messages import HumanMessage
from agent.nodes.qa_about_taxon import qa_about_taxon


@pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY no configurada, se omite test de integración con GPT",
)
def test_qa_about_taxon_integration():
    state = {
        "current_taxon": "Alouatta palliata",
        "context_md": (
            "Alouatta palliata, también conocido como mono aullador de manto, "
            "es una especie de primate que habita en los bosques de Centroamérica."
        ),
        "messages": [HumanMessage(content="¿Dónde vive este primate?")],
    }

    out = qa_about_taxon(state)
    answer = out["messages"][-1].content

    print("\n--- Respuesta QA ---")
    print(answer)
    print("-------------------\n")

    assert isinstance(answer, str)
    assert len(answer.strip()) > 0
    assert "Alouatta" in answer or "vive" in answer


# --- Permite ejecutar este archivo directamente con Python ---
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-s", __file__]))
