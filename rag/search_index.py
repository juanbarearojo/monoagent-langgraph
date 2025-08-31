# rag/search_index.py
from pathlib import Path
import json, faiss, numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
IDX_DIR = ROOT / "indices" / "global"

# 1) Cargar FAISS + metadatos
index = faiss.read_index(str(IDX_DIR / "dense.faiss"))
meta = [json.loads(line) for line in (IDX_DIR / "meta.jsonl").read_text(encoding="utf-8").splitlines()]

# 2) Embedding de consulta (¡mismo modelo y normalización!)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
def query(text, k=5):
    q = model.encode([text], normalize_embeddings=True).astype("float32")
    D, I = index.search(q, k)
    out = []
    for rank, (dist, idx) in enumerate(zip(D[0], I[0]), 1):
        m = meta[idx]
        out.append({
            "rank": rank,
            "dist": float(dist),
            "latin_name": m["latin_name"],
            "section": m["section"],
            "file": m["file"],
            "id": m["id"],
            "text_preview": m["text"][:300].replace("\n"," ") + " …"
        })
    return out

# 3) Probar
for r in query("diet and habitat of Cebuella pygmaea", k=5):
    print(f"{r['rank']}. {r['latin_name']} | {r['section']} | {r['file']} | dist={r['dist']:.4f}")
    print(r["text_preview"], "\n")
