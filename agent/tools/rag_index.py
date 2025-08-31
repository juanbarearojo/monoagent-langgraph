# agent/tools/rag_index.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional, Sequence, Tuple
import json, os
import numpy as np
import faiss

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------------- Paths / discovery ----------------
def _guess_idx_dir() -> Path:
    """
    Descubre indices/global con orden de prioridad:
      1) RAG_INDEX_DIR (env)
      2) <repo root>/indices/global      (# típico: monoagent-langgraph/indices/global)
      3) CWD/indices/global
      4) agent/indices/global            (legacy)
    """
    # 1) env var
    env = os.getenv("RAG_INDEX_DIR")
    if env:
        p = Path(env)
        if (p / "meta.jsonl").exists() and (p / "dense.faiss").exists():
            return p

    here = Path(__file__).resolve()
    candidates: list[Path] = []

    # 2) repo root ≈ subir 2 o 3 niveles según layout
    # .../agent/tools/rag_index.py -> parents[2] ~= repo root
    if len(here.parents) >= 3:
        candidates.append(here.parents[2] / "indices" / "global")
    if len(here.parents) >= 4:
        candidates.append(here.parents[3] / "indices" / "global")

    # 3) CWD
    candidates.append(Path.cwd() / "indices" / "global")

    # 4) legacy: agent/indices/global
    if len(here.parents) >= 2:
        candidates.append(here.parents[1] / "indices" / "global")

    for c in candidates:
        if (c / "meta.jsonl").exists() and (c / "dense.faiss").exists():
            return c

    # último recurso: el más probable (repo root /indices/global)
    fallback = (here.parents[2] / "indices" / "global") if len(here.parents) >= 3 else Path.cwd() / "indices" / "global"
    return fallback

ROOT = Path(__file__).resolve().parents[1]  # agent/
IDX_DIR = _guess_idx_dir()
META_PATH = IDX_DIR / "meta.jsonl"
FAISS_PATH = IDX_DIR / "dense.faiss"

_META: Optional[List[Dict[str, Any]]] = None
_INDEX: Optional[faiss.Index] = None
_EMB_MODEL: Optional[Any] = None
_EMB_NAME = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Secciones / priorización -------------------------------------------------
SECTION_ORDER: Sequence[str] = ["Identification","Habitat","Diet","Conservation","Behavior","Body"]
SEC_WEIGHT: Dict[str, float] = {s: 1.0 - i*0.1 for i, s in enumerate(SECTION_ORDER)}

# --- Carga perezosa -----------------------------------------------------------
def _load_meta() -> List[Dict[str, Any]]:
    global _META
    if _META is None:
        if not META_PATH.exists():
            raise FileNotFoundError(
                f"meta.jsonl no encontrado: {META_PATH}\n"
                f"Sugerencias:\n"
                f"  - Exporta RAG_INDEX_DIR=RUTA\\a\\indices\\global\n"
                f"  - Verifica que ejecutaste build_index.py y existen dense.faiss/meta.jsonl.\n"
            )
        _META = [json.loads(line) for line in META_PATH.read_text(encoding="utf-8").splitlines()]
    return _META

def _load_index() -> faiss.Index:
    global _INDEX
    if _INDEX is None:
        if not FAISS_PATH.exists():
            raise FileNotFoundError(
                f"dense.faiss no encontrado: {FAISS_PATH}\n"
                f"Sugerencias:\n"
                f"  - Exporta RAG_INDEX_DIR=RUTA\\a\\indices\\global\n"
                f"  - Verifica que ejecutaste build_index.py.\n"
            )
        _INDEX = faiss.read_index(str(FAISS_PATH))
    return _INDEX

def _load_model() -> Any:
    global _EMB_MODEL
    if _EMB_MODEL is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers no está instalado.")
        _EMB_MODEL = SentenceTransformer(_EMB_NAME)
    return _EMB_MODEL

# --- Utilidades ---------------------------------------------------------------
def _encode(texts: Sequence[str]) -> np.ndarray:
    model = _load_model()
    X = model.encode(list(texts), normalize_embeddings=True)
    return np.asarray(X, dtype="float32")

def _sim_from_l2sq(d: float) -> float:
    # Con embeddings unitarios: L2^2 = 2(1 - cos)
    s = 1.0 - float(d)/2.0
    return max(0.0, min(1.0, s))

# --- Búsqueda semántica -------------------------------------------------------
def search(
    query: str,
    k: int = 8,
    filter_latin: Optional[str] = None,
    filter_species_id: Optional[str] = None,
    filter_sections: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    meta = _load_meta()
    index = _load_index()

    qv = _encode([query])
    K = max(k*4, 20)
    D, I = index.search(qv, K)
    cand: List[Dict[str, Any]] = []

    for d, idx in zip(D[0], I[0]):
        if int(idx) < 0:
            continue
        m = meta[idx]
        if filter_latin and (m.get("latin_name","").lower() != filter_latin.lower()):
            continue
        if filter_species_id and (m.get("species_id") != filter_species_id):
            continue
        if filter_sections and (m.get("section") not in set(filter_sections)):
            continue

        sec = m.get("section", "Body")
        w = SEC_WEIGHT.get(sec, 0.7)
        sim = _sim_from_l2sq(float(d))
        score = sim * w

        out = dict(m)
        out.update({"dist": float(d), "score": float(score)})
        cand.append(out)

    cand.sort(key=lambda x: x["score"], reverse=True)
    return cand[:k]

# --- Retrieval por especie (queryless) ---------------------------------------
def retrieve_species(latin_name: str, top_k_sections: int = 3) -> List[Dict[str, Any]]:
    meta = _load_meta()
    ln = (latin_name or "").strip().lower()
    cand = [m for m in meta if (m.get("latin_name","").lower() == ln)]
    if not cand:
        return []

    chosen: List[Dict[str, Any]] = []
    used_secs = set()

    for sec in SECTION_ORDER:
        if len(chosen) >= top_k_sections: break
        for m in cand:
            if m.get("section") == sec and sec not in used_secs:
                chosen.append(m); used_secs.add(sec); break

    if len(chosen) < top_k_sections:
        for m in cand:
            if len(chosen) >= top_k_sections: break
            if m not in chosen:
                chosen.append(m)

    return chosen[:top_k_sections]

# --- Calentamiento ------------------------------------------------------------
def warmup(load_model: bool = False) -> Tuple[int, int, str]:
    meta = _load_meta()
    index = _load_index()
    if load_model:
        _load_model()
    dim = int(getattr(index, "d", 0))
    return len(meta), dim, str(IDX_DIR)
