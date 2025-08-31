# build_index.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List, Tuple
import os, json, re, hashlib, time, argparse, logging
from datetime import datetime

# ---- Progreso / Logging ----
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False

# ---- Embeddings / Indexing ----
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

# ---- OpenAI Structured Outputs ----
from pydantic import BaseModel, Field
from openai import OpenAI
OPENAI_ENABLED = bool(os.getenv("OPENAI_API_KEY"))
client = OpenAI() if OPENAI_ENABLED else None

# ------------------- Paths -------------------
ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "corpus"
IDX_DIR = ROOT / "indices" / "global"
IDX_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Secciones -------------------
SECTIONS = ["Identification","Habitat","Diet","Conservation","Behavior","Body"]
SEC_PATTERNS = {
    "Identification": r"(Identification|Morphology|Diagnostic)",
    "Habitat":        r"(Habitat|Range|Distribution)",
    "Diet":           r"(Diet|Feeding|Food|Exudativore|Gummivore)",
    "Conservation":   r"(Conservation|Threats|IUCN|Status)",
    "Behavior":       r"(Behavior|Behaviour|Social)",
}

# ------------------- Logging helpers -------------------
log = logging.getLogger("build_index")

def setup_logging(verbose: bool):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

def banner(msg: str):
    print("\n" + "="*80)
    print(msg)
    print("="*80)

def stamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

# ------------------- Utils texto -------------------
def snake_to_binomial(s: str) -> str:
    g,e = s.split("_", 1)
    return g.capitalize() + " " + e

def infer_latin_from_filename(name: str) -> tuple[Optional[str], Optional[str]]:
    m = re.match(r"([a-z]+_[a-z]+)", name)
    if not m: return None, None
    sid = m.group(1)
    return snake_to_binomial(sid), sid

def infer_year(name: str, text_head: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", name)
    if m: return int(m.group(0))
    m = re.search(r"(19|20)\d{2}", text_head[:1500])
    return int(m.group(0)) if m else None

def extract_text(pdf_path: Path) -> str:
    """
    Intenta extraer con pypdf; si falla o devuelve poco texto,
    usa PyMuPDF (pymupdf) como fallback. Si tampoco, devuelve "".
    """
    # 1) pypdf primero
    try:
        reader = PdfReader(str(pdf_path))
        txt = "\n\n".join(p.extract_text() or "" for p in reader.pages)
        if txt and len(txt.strip()) > 50:
            return txt
    except Exception as e:
        log.warning(f"pypdf fallo en {pdf_path.name}: {e}")

    # 2) PyMuPDF fallback
    try:
        import fitz  # pymupdf
        with fitz.open(str(pdf_path)) as doc:
            blocks = []
            for page in doc:
                blocks.append(page.get_text("text"))
        txt2 = "\n\n".join(blocks)
        if txt2 and len(txt2.strip()) > 0:
            log.info(f"PyMuPDF usado para {pdf_path.name}")
            return txt2
    except Exception as e:
        log.warning(f"PyMuPDF fallo en {pdf_path.name}: {e}")

    # 3) Nada…
    log.error(f"No se pudo extraer texto de {pdf_path.name} (¿escaneado sin OCR?)")
    return ""

def split_chunks(txt: str, max_chars=1800, overlap=300) -> List[str]:
    # ~450 tokens ≈ 1800 chars
    txt = re.sub(r"(\w)-\n(\w)", r"\1\2", txt)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    blocks, out = txt.split("\n\n"), []
    buf = ""
    for b in blocks:
        b = b.strip()
        if not b: continue
        if len(buf) + len(b) + 2 <= max_chars:
            buf = (buf + "\n\n" + b).strip()
        else:
            if buf: out.append(buf)
            while len(b) > max_chars:
                out.append(b[:max_chars])
                b = b[max_chars-overlap:]
            buf = b
    if buf: out.append(buf)
    return out

def guess_section_head(text: str) -> str:
    head = text[:400].replace("\n"," ")
    for sec, pat in SEC_PATTERNS.items():
        if re.search(pat, head, re.I): return sec
    return "Body"

# ------------------- Cache simple -------------------
CACHE_PATH = IDX_DIR / "openai_cache.jsonl"
_cache_map: Dict[str, Dict[str, Any]] = {}

def _load_cache():
    if CACHE_PATH.exists():
        for line in CACHE_PATH.read_text(encoding="utf-8").splitlines():
            try:
                obj = json.loads(line)
                _cache_map[obj["key"]] = obj
            except Exception:
                pass

def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    return _cache_map.get(key)

def _cache_put(key: str, value: Dict[str, Any]):
    rec = {"key": key, "value": value}
    _cache_map[key] = rec
    with open(CACHE_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def _hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:32]

# ------------------- OpenAI Structured Outputs -------------------
class DocMeta(BaseModel):
    latin_name: Optional[str] = Field(None, description="Binomial, e.g., 'Cebuella pygmaea'")
    species_id: Optional[str] = Field(None, description="snake_case: 'cebuella_pygmaea'")
    year: Optional[int]       = Field(None, description="Publication year if explicit")
    title: Optional[str]      = Field(None, description="Document title if detected")
    source_confidence: Literal["low","medium","high"] = "medium"

class ChunkMeta(BaseModel):
    section: Literal[tuple(SECTIONS)] = "Body"
    rationale: Optional[str] = None

DOC_SYS = (
    "You are a taxonomic metadata extractor for primate PDFs."
    " Return ONLY the fields requested by the JSON schema."
)
CHUNK_SYS = (
    "You classify a text chunk from a species monograph into a section label."
)

def _openai_parse_with_retries(model: str, system: str, user: str, schema_model, retries=3, backoff=2.0):
    """
    Usa la API beta de Chat Completions con Structured Outputs (Pydantic).
    """
    if not OPENAI_ENABLED:
        raise RuntimeError("OPENAI_API_KEY not set")
    last_err = None
    for i in range(retries):
        try:
            resp = client.beta.chat.completions.parse(
                model=model,  # p.ej., "gpt-4o-mini"
                messages=[
                    {"role":"system","content": system},
                    {"role":"user","content": user}
                ],
                response_format=schema_model
            )
            return resp.choices[0].message.parsed
        except Exception as e:
            last_err = e
            log.warning(f"[OpenAI] intento {i+1}/{retries} fallido: {e}")
            if i < retries-1:
                time.sleep(backoff*(i+1))
            else:
                raise last_err

def extract_doc_meta_openai(text_head: str, fallback_name: str|None=None) -> Dict[str, Any]:
    key = "docmeta:" + _hash((fallback_name or "") + text_head[:4000])
    hit = _cache_get(key)
    if hit:
        log.info("Cache hit docmeta")
        return hit["value"]
    user = (
        "From the following primate PDF header/body excerpt, infer fields if present. "
        "Prefer explicit mentions; avoid guessing.\n\n"
        f"FILENAME: {fallback_name or 'unknown'}\n\n"
        f"TEXT HEAD:\n{text_head[:4000]}"
    )
    out: DocMeta = _openai_parse_with_retries("gpt-4o-mini", DOC_SYS, user, DocMeta)
    sid = out.species_id
    if not sid and out.latin_name and " " in out.latin_name:
        g,e = out.latin_name.split(" ",1)
        sid = f"{g.lower()}_{e.lower()}"
    val = {
        "latin_name": out.latin_name,
        "species_id": sid,
        "year": out.year,
        "title": out.title,
        "source_confidence": out.source_confidence,
    }
    _cache_put(key, val)
    return val

def classify_chunk_section_openai(text: str) -> Dict[str, Any]:
    key = "chunksec:" + _hash(text[:1800])
    hit = _cache_get(key)
    if hit:
        log.info("Cache hit chunksec")
        return hit["value"]
    user = (
        "Label this chunk with one of the allowed section names. "
        f"Allowed sections: {', '.join(SECTIONS)}.\n\nCHUNK:\n{text[:1800]}"
    )
    out: ChunkMeta = _openai_parse_with_retries("gpt-4o-mini", CHUNK_SYS, user, ChunkMeta)
    val = {"section": out.section, "rationale": out.rationale}
    _cache_put(key, val)
    return val

# ------------------- Fallbacks locales -------------------
def fallback_doc_meta(head: str, pdf_name: str) -> Tuple[str, str, Optional[int]]:
    latin, species_id = infer_latin_from_filename(pdf_name)
    if not latin:
        m = re.search(r"\b([A-Z][a-z]+ [a-z]+)\b", head)
        if m:
            latin = m.group(1)
            species_id = latin.lower().replace(" ", "_")
        else:
            latin, species_id = "Unknown", "unknown"
    year = infer_year(pdf_name, head)
    return latin, species_id, year

def fallback_chunk_section(ch: str) -> str:
    return guess_section_head(ch)

# ------------------- Index builder -------------------
def build_index(
    use_openai_docmeta: bool = True,
    use_openai_section: bool = True,
    max_pdfs: Optional[int] = None,
    chunk_chars: int = 1800,
    chunk_overlap: int = 300,
    hnsw_m: int = 32,
    hnsw_efc: int = 200,
    ef_search: int = 64,
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    heartbeat_every: int = 250,
    use_tqdm: bool = True,
):
    start_t = time.perf_counter()
    banner("🧭 INICIO: Construcción de índice FAISS (HNSW) con metadatos OpenAI")
    print(f"[{stamp()}] Configuración:")
    print(f"  - OPENAI_ENABLED: {OPENAI_ENABLED}")
    print(f"  - embedding_model: {embedding_model}")
    print(f"  - chunk_chars/overlap: {chunk_chars}/{chunk_overlap}")
    print(f"  - HNSW M/efConstruction/efSearch: {hnsw_m}/{hnsw_efc}/{ef_search}")
    print(f"  - Corpus dir: {PDF_DIR}")
    print(f"  - Salida: {IDX_DIR}\n")

    if use_tqdm and not HAS_TQDM:
        print("⚠️  tqdm no disponible, continuo sin barras. `pip install tqdm` para activarlas.")
        use_tqdm = False

    _load_cache()

    banner("🔎 Cargando modelo de embeddings")
    model = SentenceTransformer(embedding_model)

    vecs: List[np.ndarray] = []
    meta: List[Dict[str, Any]] = []

    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    total_pdfs = len(pdfs)
    if max_pdfs is not None:
        pdfs = pdfs[:max_pdfs]
    print(f"[{stamp()}] PDFs a procesar: {len(pdfs)} / {total_pdfs}")

    banner("📄 Procesando PDFs")
    pdf_iter = tqdm(pdfs, desc="PDFs", unit="pdf") if use_tqdm else pdfs
    processed_chunks = 0

    for pdf in pdf_iter:
        try:
            log.info(f"Extrayendo texto: {pdf.name}")
            raw = extract_text(pdf)
            head = raw[:4000]
            if not raw.strip():
                log.warning(f"PDF vacío (¿texto escaneado sin OCR?): {pdf.name}")
                continue

            # ---- Doc metadata (OpenAI + fallback) ----
            print(f"[{stamp()}] 🏷️ Metadatos doc → {pdf.name}")
            if use_openai_docmeta and OPENAI_ENABLED:
                try:
                    docm = extract_doc_meta_openai(head, fallback_name=pdf.name)
                    latin = docm.get("latin_name") or "Unknown"
                    species_id = docm.get("species_id") or "unknown"
                    year = docm.get("year")
                except Exception as e:
                    log.warning(f"OpenAI docmeta fallo ({pdf.name}): {e}")
                    latin, species_id, year = fallback_doc_meta(head, pdf.name)
            else:
                latin, species_id, year = fallback_doc_meta(head, pdf.name)

            # ---- Split chunks ----
            chunks = split_chunks(raw, max_chars=chunk_chars, overlap=chunk_overlap)
            print(f"[{stamp()}] 🧩 Chunks: {len(chunks)}")

            chunk_iter = tqdm(chunks, desc=f"Chunks:{pdf.name}", unit="ch", leave=False) if use_tqdm else chunks
            for idx, ch in enumerate(chunk_iter):
                # Section label (OpenAI + fallback)
                if use_openai_section and OPENAI_ENABLED:
                    try:
                        sec_info = classify_chunk_section_openai(ch)
                        sec = sec_info.get("section", "Body")
                        rationale = sec_info.get("rationale")
                    except Exception as e:
                        log.warning(f"OpenAI chunksec fallo ({pdf.name} #{idx}): {e}")
                        sec = fallback_chunk_section(ch)
                        rationale = None
                else:
                    sec = fallback_chunk_section(ch)
                    rationale = None

                uid = f"{species_id}|{pdf.name}|{idx:05d}"
                meta.append({
                    "id": uid,
                    "latin_name": latin,
                    "species_id": species_id,
                    "file": pdf.name,
                    "year": year,
                    "section": sec,
                    "section_rationale": rationale,
                    "text": ch
                })
                vecs.append(model.encode(ch, normalize_embeddings=True))
                processed_chunks += 1

                # Heartbeat visible
                if heartbeat_every and (processed_chunks % heartbeat_every == 0):
                    print(f"[{stamp()}] ⏱️ Heartbeat: {processed_chunks} chunks procesados...")

        except Exception as e:
            log.error(f"Error procesando {pdf.name}: {e}")

    if not vecs:
        print("No chunks found — nada que indexar.")
        return

    banner("🏗️ Construyendo índice FAISS (HNSW)")
    X = np.vstack(vecs).astype("float32")
    dim = X.shape[1]
    print(f"[{stamp()}] Matriz embeddings: {X.shape} (dim={dim})")
    index = faiss.IndexHNSWFlat(dim, hnsw_m)
    index.hnsw.efConstruction = hnsw_efc
    try:
        hnsw = faiss.downcast_IndexHNSW(index)
        hnsw.efSearch = ef_search
    except Exception:
        pass

    print(f"[{stamp()}] Añadiendo vectores al índice…")
    index.add(X)

    banner("💾 Persistiendo artefactos")
    faiss_path = IDX_DIR / "dense.faiss"
    meta_path  = IDX_DIR / "meta.jsonl"
    mani_path  = IDX_DIR / "manifest.json"

    faiss.write_index(index, str(faiss_path))
    with open(meta_path, "w", encoding="utf-8") as f:
        for m in meta: f.write(json.dumps(m, ensure_ascii=False) + "\n")
    manifest = {
        "embedding_model": embedding_model,
        "dimension": int(dim),
        "built_at": datetime.utcnow().isoformat()+"Z",
        "chunks": int(X.shape[0]),
        "corpus_dir": "data/corpus",
        "openai_docmeta": bool(use_openai_docmeta and OPENAI_ENABLED),
        "openai_section": bool(use_openai_section and OPENAI_ENABLED),
        "hnsw_m": hnsw_m,
        "hnsw_efConstruction": hnsw_efc,
        "efSearch_default": ef_search,
        "tqdm": use_tqdm,
        "heartbeat_every": heartbeat_every,
    }
    mani_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_t
    banner("✅ FIN")
    print(f"[{stamp()}] Índice: {faiss_path}")
    print(f"[{stamp()}] Meta:   {meta_path}")
    print(f"[{stamp()}] Manifiesto: {mani_path}")
    print(f"[{stamp()}] PDFs procesados: {len(pdfs)} | chunks: {X.shape[0]} | dim: {dim}")
    print(f"[{stamp()}] Tiempo total: {elapsed:.1f}s\n")

# ------------------- CLI -------------------
def parse_args():
    ap = argparse.ArgumentParser(description="Build FAISS index with OpenAI-assisted metadata (con progreso)")
    ap.add_argument("--no-openai-docmeta", action="store_true", help="No usar OpenAI para metadatos de documento")
    ap.add_argument("--no-openai-section", action="store_true", help="No usar OpenAI para sección de chunks")
    ap.add_argument("--max-pdfs", type=int, default=None)
    ap.add_argument("--chunk-chars", type=int, default=1800)
    ap.add_argument("--chunk-overlap", type=int, default=300)
    ap.add_argument("--hnsw-m", type=int, default=32)
    ap.add_argument("--hnsw-efc", type=int, default=200)
    ap.add_argument("--ef-search", type=int, default=64)
    ap.add_argument("--embedding-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--heartbeat-every", type=int, default=250, help="Imprime heartbeat cada N chunks (0 para desactivar)")
    ap.add_argument("--no-tqdm", action="store_true", help="Desactiva barras de progreso tqdm")
    ap.add_argument("--verbose", action="store_true", help="Logging INFO detallado")
    return ap.parse_args()

def main():
    args = parse_args()
    setup_logging(verbose=args.verbose)
    build_index(
        use_openai_docmeta = not args.no_openai_docmeta,
        use_openai_section = not args.no_openai_section,
        max_pdfs = args.max_pdfs,
        chunk_chars = args.chunk_chars,
        chunk_overlap = args.chunk_overlap,
        hnsw_m = args.hnsw_m,
        hnsw_efc = args.hnsw_efc,
        ef_search = args.ef_search,
        embedding_model = args.embedding_model,
        heartbeat_every = args.heartbeat_every,
        use_tqdm = not args.no_tqdm,
    )

if __name__ == "__main__":
    main()
