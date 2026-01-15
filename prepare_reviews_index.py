from datasets import load_dataset
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm.auto import tqdm
from multiprocessing import cpu_count
import subprocess, os, math

ds = load_dataset("guyhadad01/Hotels_reviews")["train"].to_json("HotelRec.jsonl")
JSONL_PATH   = "HotelRec.jsonl"
# URI = "./reviews_hotels.db"     # <— local file => Milvus Lite (no server)
COLLECTION = "reviews_server_index"

DIM          = 384                        # all-MiniLM-L6-v2
BATCH_ROWS   = 8192                       # tune for your RAM/VRAM
EMB_BATCH    = 2048                       
USE_CUDA     = True
PRECOUNT     = False                     
INDEX_PARAMS = {
    "metric_type": "IP",
    "index_type": "HNSW",
    "params": {
        "M": 16,               
        "efConstruction": 200
    },
}
SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 128}}

try:
    import orjson as _json
    loads = _json.loads
except Exception:
    import json as _json
    loads = _json.loads

def fast_count_lines(path: str) -> int:
    try:
        out = subprocess.check_output(["wc", "-l", path], text=True)
        return int(out.strip().split()[0])
    except Exception:
        pass
    # Fallback stream count
    cnt = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            cnt += chunk.count(b"\n")
    return cnt

def iter_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = loads(line)
            except Exception:
                continue
            title = (rec.get("title") or "").strip()
            date  = (rec.get("date") or "").strip()
            text  = (rec.get("text") or "").strip()
            if not (title or date or text):
                continue
            body = f"Title: {title}\nDate: {date}\n\n{text}".strip()
            meta = {
                "Rating": str(rec.get("rating", "")),
                "Name":   rec.get("Name", ""),
                "City":   rec.get("City", ""),
                "County": rec.get("County", ""),
            }
            yield body, meta

# ---------- Embeddings ----------
emb = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if USE_CUDA else "cpu"},
    encode_kwargs={
        "normalize_embeddings": True,     # makes IP ~= cosine
        "batch_size": EMB_BATCH,
    },
)

INDEX_PARAMS = {"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 4096}}
SEARCH_PARAMS = {"metric_type": "IP", "params": {"ef": 128}}


vector_store = Milvus(
    embedding_function=emb,  
    connection_args={
        "host": "localhost",   # or the Milvus container host
        "port": "19530",
    },
    collection_name=COLLECTION,
    index_params=INDEX_PARAMS, 
    search_params= SEARCH_PARAMS  , 
    drop_old=True,
    auto_id=True,                   # <-- we’ll supply our own IDs
)

def batched(iterable, n):
    buf = []
    for x in iterable:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf

# ---------- Ingest ----------
total = 21112546
pbar = tqdm(total=total, desc="→ Ingesting into Milvus", unit="docs", dynamic_ncols=True, smoothing=0.1)

added = 0
for rows in batched(iter_jsonl(JSONL_PATH), BATCH_ROWS):
    texts = [r[0] for r in rows]
    metas = [r[1] for r in rows]

    # Pre-embed
    vecs = emb.embed_documents(texts)  # list of lists (dim=384)

    # Push vectors directly
    vector_store.add_embeddings(texts, vecs, metadatas=metas)

    pbar.update(len(rows))
    added += len(rows)

pbar.close()

# Build index once
# vector_store.create_index()

print(f"Indexed {added} docs into Milvus collection '{COLLECTION}'")



