# agent/graph.py
from __future__ import annotations
from langgraph.graph import StateGraph, START, END  # type: ignore
from agent.state import ChatVisionState  # type: ignore

# Nodos
from agent.nodes.router import router_input
from agent.nodes.ensure_image import ensure_image
from agent.nodes.infer_local import infer_local
from agent.nodes.gate_uncertainty import gate_uncertainty
from agent.nodes.map_scientific import map_to_scientific_name
from agent.nodes.ask_gpt41_vision import ask_gpt41_vision
from agent.nodes.wiki_fullpage import fetch_wikipedia_fullpage
from agent.nodes.ddg_search import retrieve_ddg
from agent.nodes.merge_context import merge_context
from agent.nodes.finalize import finalize_answer
from agent.nodes.qa_about_taxon import qa_about_taxon
from agent.nodes.prompt_for_image import prompt_for_image
from agent.nodes.clarify import clarify_or_fail
from agent.nodes.capture_user_taxon import capture_user_taxon

def build_graph():
    g = StateGraph(ChatVisionState)

    # --- registrar nodos ---
    g.add_node("router_input", router_input)
    g.add_node("ensure_image", ensure_image)
    g.add_node("infer_local", infer_local)
    g.add_node("gate_uncertainty", gate_uncertainty)
    g.add_node("map_to_scientific_name", map_to_scientific_name)
    g.add_node("ask_gpt41_vision", ask_gpt41_vision)
    g.add_node("fetch_wikipedia_fullpage", fetch_wikipedia_fullpage)
    g.add_node("retrieve_ddg", retrieve_ddg)
    g.add_node("merge_context", merge_context)
    g.add_node("finalize_answer", finalize_answer)
    g.add_node("qa_about_taxon", qa_about_taxon)
    g.add_node("prompt_for_image", prompt_for_image)
    g.add_node("clarify_or_fail", clarify_or_fail)
    g.add_node("capture_user_taxon", capture_user_taxon)

    # --- entrada ---
    g.add_edge(START, "router_input")

    # --- router → classify / ask_image ---
    def route_selector(state: ChatVisionState):
        # ¡ahora _tmp sí persiste!
        r = state.get("_tmp", {}).get("route")
        return "ensure_image" if r == "classify" else "ask_image"

    g.add_conditional_edges(
        "router_input",
        route_selector,
        {
            "ensure_image": "ensure_image",
            "ask_image": "capture_user_taxon",
        },
    )

    # tras capturar, pedimos imagen
    g.add_edge("capture_user_taxon", "prompt_for_image")

    # ensure_image -> infer o clarificar
    def img_ok(state: ChatVisionState):
        return "infer_local" if state.get("_tmp", {}).get("image_ok") else "clarify_or_fail"

    g.add_conditional_edges(
        "ensure_image",
        img_ok,
        {
            "infer_local": "infer_local",
            "clarify_or_fail": "clarify_or_fail",
        },
    )

    # infer -> gate
    g.add_edge("infer_local", "gate_uncertainty")

    # gate -> accept o review
    def gate_sel(state: ChatVisionState):
        return "map_to_scientific_name" if state.get("_tmp", {}).get("gate") == "ACCEPT" else "ask_gpt41_vision"

    g.add_conditional_edges(
        "gate_uncertainty",
        gate_sel,
        {
            "map_to_scientific_name": "map_to_scientific_name",
            "ask_gpt41_vision": "ask_gpt41_vision",
        },
    )

    # map_to_scientific_name -> fallback o fan-out
    def map_next(state: ChatVisionState):
        return "ask_gpt41_vision" if state.get("_tmp", {}).get("need_fallback") else "fanout"

    g.add_conditional_edges(
        "map_to_scientific_name",
        map_next,
        {
            "ask_gpt41_vision": "ask_gpt41_vision",
            "fanout": "fetch_wikipedia_fullpage",
        },
    )
    g.add_edge("map_to_scientific_name", "retrieve_ddg")

    # ask_gpt41_vision -> clarificar o fan-out
    def vision_next(state: ChatVisionState):
        return "fanout" if state.get("_tmp", {}).get("vision_status") == "ok" else "clarify_or_fail"

    g.add_conditional_edges(
        "ask_gpt41_vision",
        vision_next,
        {
            "fanout": "fetch_wikipedia_fullpage",
            "clarify_or_fail": "clarify_or_fail",
        },
    )
    g.add_edge("ask_gpt41_vision", "retrieve_ddg")

    # ambas ramas -> merge
    g.add_edge("fetch_wikipedia_fullpage", "merge_context")
    g.add_edge("retrieve_ddg", "merge_context")

    # merge -> finalize
    g.add_edge("merge_context", "finalize_answer")

    # terminales
    g.add_edge("prompt_for_image", END)
    g.add_edge("clarify_or_fail", END)
    g.add_edge("finalize_answer", END)
    g.add_edge("qa_about_taxon", END)

    return g.compile()
