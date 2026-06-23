import pickle #to load bm25 model
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

###
EMBED_MODEL ="all-MiniLM-L6-v2"
CHROMA_DIR ="./chroma_db"
BM25_FILE ="./bm25_index.pkl"
TOP_K =10


# Load REsorces

print("Loading Embedding Model...")
_embed_model = SentenceTransformer(EMBED_MODEL)
print("Embedding Model Loaded.")

_chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
_chroma_collection = _chroma_client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"},
)

# Tokenize text into words
def _tokenise(text: str) -> list[str]:
    text = re.sub(r"[^\w\s\-/]", " ", text.lower())
    return [t for t in text.split() if len(t) >= 2]

# Load the BM25 index from disk
def _load_bm25():
    if not Path(BM25_FILE).exists():
        raise FileNotFoundError(f"BM25 index file not found: {BM25_FILE}")
    with open(BM25_FILE, "rb") as f:
        saved = pickle.load(f)
    return saved["bm25"], saved["chunks"]


# Merge two rankings into a single ranking
def _rrf(vec_ids: list,bm25_ids: list, k: int = 60) -> list:
    scores ={}
    for rank, id_ in enumerate(vec_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (k +rank + 1)
    for rank, id_ in enumerate(bm25_ids):
        scores[id_] = scores.get(id_, 0) + 1 / (k +rank + 1)
    return sorted(scores, key=scores.get, reverse=True) 

def retrieve(query: str, domain: str, top_k: int = TOP_K) -> list[dict]:
    #Embade Query
    query_vec = _embed_model.encode(
        query, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    # Vector Search
    n = min(top_k * 2, _chroma_collection.count())
    if n == 0:
        print("  No documents ingested yet. Run ingest.py first.")
        return []

    vec_results = _chroma_collection.query(
        query_embeddings=[query_vec],
        n_results=n,
        where={"domain": {"$eq": domain}},
        include=["documents", "metadatas", "distances"],
    )

    vec_ids    = vec_results["ids"][0]
    vec_lookup = {}     # id → chunk dict
    for i, id_ in enumerate(vec_ids):
        meta = vec_results["metadatas"][0][i]
        vec_lookup[id_] = {
            "text":    vec_results["documents"][0][i],
            "domain":  meta.get("domain", domain),
            "section": meta.get("section", ""),
            "source":  meta.get("source", ""),
            "page":    int(meta.get("page", 0)),
            "score":   round(1 - vec_results["distances"][0][i], 4),
        }

    # BM25 Search
    bm25_model, bm25_chunks = _load_bm25()
    bm25_ids   = []
    bm25_lookup = {}

    if bm25_model:
        query_tokens = _tokenise(query)
        scores       = bm25_model.get_scores(query_tokens)

        # Get top results from same domain only
        domain_indices = [
            i for i, c in enumerate(bm25_chunks) if c["domain"] == domain
        ]
        ranked = sorted(domain_indices, key=lambda i: scores[i], reverse=True)

        for idx in ranked[: top_k * 2]:
            chunk  = bm25_chunks[idx]
            stem   = Path(chunk["source"]).stem.replace(" ", "_")
            # Reconstruct the same ID format used in ingest.py
            cid    = f"{stem}_chunk_{idx}"
            bm25_ids.append(cid)
            bm25_lookup[cid] = {**chunk, "score": float(scores[idx])}

    # RRF merge 
    merged_ids = _rrf(vec_ids, bm25_ids)

    # Build final result list 
    results = []
    for id_ in merged_ids[:top_k]:
        if id_ in vec_lookup:
            results.append(vec_lookup[id_])
        elif id_ in bm25_lookup:
            results.append(bm25_lookup[id_])

    return results

# Test code
if __name__ == "__main__":
    import sys
    query  = sys.argv[1] if len(sys.argv) > 1 else "What medications were prescribed?"
    domain = sys.argv[2] if len(sys.argv) > 2 else "medical"

    print(f"\nQuery : {query}")
    print(f"Domain: {domain}\n")

    chunks = retrieve(query, domain=domain, top_k=5)
    for i, c in enumerate(chunks, 1):
        print(f"[{i}] Section: {c['section']} | Page: {c['page']} | Score: {c['score']}")
        print(f"    {c['text'][:120]}...")
        print()


