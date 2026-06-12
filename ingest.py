import re       # reguler expressions
import sys
import pickle                                  
from pathlib import Path

import fitz                                      # pip install pymupdf
import spacy                                     # pip install spacy
import chromadb                                  # pip install chromadb
from rank_bm25 import BM25Okapi                  # pip install rank-bm25
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# Settings 
EMBED_MODEL   = "all-MiniLM-L6-v2"   
CHROMA_DIR    = "./chroma_db"          # loccation for chromaDB
BM25_FILE     = "./bm25_index.pkl"     # where BM25 index is saved
CHUNK_MAX     = 1000                   # max characters per chunk
SENTENCES     = 4 

MEDICAL_SECTIONS = [
    "chief complaint", "history of present illness", "past medical history",
    "medications", "allergies", "physical examination", "assessment",
    "plan", "diagnosis", "lab results", "discharge summary", "treatment",
    "review of systems", "family history", "social history", "vital signs",
]

LEGAL_SECTIONS = [
    "definitions", "recitals", "obligations", "payment terms",
    "confidentiality", "termination", "limitation of liability",
    "indemnification", "governing law", "dispute resolution",
    "representations and warranties", "intellectual property",
    "force majeure", "entire agreement", "amendments", "notices",
]

# 1. Parse PDF

def parse_pdf(path: str) -> str:
    doc = fitz.open(path)
    pages =[]

    for page in doc:
        text = page.get_text("text")
        if not text.strip():
            continue
    
        # Fix hyphenated line breaks: "diabe-\ntes" → "diabetes"
        text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)
        # Remove standalone page numbers
        text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)
        # Remove "Page X of Y" footers
        text = re.sub(r'(?i)page\s+\d+\s+of\s+\d+', '', text)
        # Collapse multiple spaces
        text = re.sub(r'[ \t]+', ' ', text)
        # Collapse 3+ blank lines → 2
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Fix abbreviations that confuse sentence splitter
        for abbr in ['Dr.', 'Mr.', 'Mrs.', 'vs.', 'etc.']:
            text = text.replace(abbr, abbr.replace('.', ''))

        pages.append(text.strip())


    doc.close()
    return "\n\n".join(pages)


# 2. Detect Domain 
def detect_domain(text: str) -> str:
    t = text.lower()
    med   = sum(1 for w in ["patient","diagnosis","physician","dosage","mg","prescription"] if w in t)
    legal = sum(1 for w in ["whereas","hereinafter","party","clause","agreement","liability"] if w in t)
    return "legal" if legal > med else "medical"
    
# 3. Chunking
def chunk_text(text: str, source: str, domain: str) -> list[dict]:

    # ragex for section headers 
    all_sections = MEDICAL_SECTIONS + LEGAL_SECTIONS
    pattern = re.compile(
        r'(?im)^(' + '|'.join(re.escape(s) for s in all_sections) + r')[\s:]*$'
    )

    matches = list(pattern.finditer(text))
    chunks  = []

    if not matches:
        # No section headers found → sentence chunk the whole doc
        return _sentence_chunks(text, section="general", source=source, domain=domain)

    for i, match in enumerate(matches):
        section  = match.group(0).strip().lower()
        start    = match.end()
        end      = matches[i+1].start() if i+1 < len(matches) else len(text)
        content  = text[start:end].strip()

        if not content:
            continue

        if len(content) <= CHUNK_MAX:
            chunks.append({
                "text":    f"{section}\n{content}",
                "domain":  domain,
                "section": section,
                "source":  source,
                "page":    0,       
            })
        else:
            # Section too long → break by sentences
            chunks.extend(
                _sentence_chunks(content, section=section, source=source, domain=domain)
            )

    return chunks

# Split text into groups of SENTENCES sentences with 1-sentence overlap.
def _sentence_chunks(text: str, section: str, source: str, domain: str) -> list[dict]:
    try:
        nlp  = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("Run: python -m spacy download en_core_web_sm")

    sents = [s.text.strip() for s in nlp(text).sents if s.text.strip()]
    chunks = []
    i=0

    while i < len(sents):
        group = sents[i : i + SENTENCES]
        chunks.append({
            "text":    " ".join(group),
            "domain":  domain,
            "section": section,
            "source":  source,
            "page":    0,
        })
        i += SENTENCES - 1     # 1-sentence overlap

    return chunks


# match chunk text back to a page number      
def _attach_pages(chunks: list[dict], full_text_by_page: list[tuple]):
    for chunk in chunks:
        snippet = chunk["text"][:60]
        for page_num, page_text in full_text_by_page:
            if snippet in page_text:
                chunk["page"] = page_num
                break

# 4.Embedding  : (Embed all chunk text and Returns list of float vectors)
def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    print(f"  Embedding {len(chunks)} chunks with {EMBED_MODEL}...")
    model   = SentenceTransformer(EMBED_MODEL)
    texts   = [c["text"] for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    return vectors.tolist()


#5.  Store
def store(chunks: list[dict], embeddings: list[list[float]], source_file: str):

    # Store in ChromaDB 
    client  = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name="rag_docs",
        metadata={"hnsw:space": "cosine"},
    )

    stem = Path(source_file).stem.replace(" ", "_")
    ids, docs, metas = [], [], []

    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        ids.append(f"{stem}_chunk_{i}")
        docs.append(chunk["text"])
        metas.append({
            "domain":  chunk["domain"],
            "section": chunk["section"],
            "source":  chunk["source"],
            "page":    chunk["page"],
        })

    collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
    print(f"  Stored {len(ids)} chunks in ChromaDB  (total: {collection.count()})")

    #  BM25 index (for keyword search)
    # Load existing index if present, then append new chunks
    existing_chunks, existing_corpus = [], []
    if Path(BM25_FILE).exists():
        with open(BM25_FILE, "rb") as f:
            saved = pickle.load(f)
            existing_chunks = saved["chunks"]
            existing_corpus = saved["corpus"]

    def _tokenise(text):
        text = re.sub(r"[^\w\s\-/]", " ", text.lower())
        return [t for t in text.split() if len(t) >= 2]

    new_corpus  = [_tokenise(c["text"]) for c in chunks]
    all_chunks  = existing_chunks + chunks
    all_corpus  = existing_corpus + new_corpus
    bm25_model  = BM25Okapi(all_corpus)

    with open(BM25_FILE, "wb") as f:
        pickle.dump({"bm25": bm25_model, "chunks": all_chunks, "corpus": all_corpus}, f)
    print(f"  BM25 index saved  ({len(all_chunks)} total chunks)")


# Main

def ingest(pdf_path :str, domain: str ="auto"):
    path=Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    
    print("\n--- Ingesting Document ---")
    print(f"Ingesting: {pdf_path}")
    print("---------------------------")

    # Parsing
    # keep page text in memory to match chunks back to page numbers later
    doc = fitz.open(str(path))
    page_raw =[(i+1, page.get_text("text")) for i, page in enumerate(doc)]
    doc.close()

    full_text =parse_pdf(str(path))

    # Domain Detection
    if domain == "auto":
        domain=detect_domain(full_text)
    print(f"Domain: {domain}")

    # Chunking
    chunks = chunk_text(full_text, source =path.name, domain=domain)
    _attach_pages(chunks, page_raw)
    sections= set(c["section"] for c in chunks)
    print(f" Chunks : {len(chunks)} | Sections : {sections}")

    # Embedding
    embeddings = embed_chunks(chunks)
    # Store
    store(chunks, embeddings, source_file=path.name)

    print("Ingestion complete.\n")
    return len(chunks)

if __name__ == "__main__":
    if len(sys.argv) <2 :
        print("Usage: python ingest.py <path_to_pdf> [--domain <domain_name>]")
        print("Example: python ingest.py medical_report.pdf --domain medical")
        sys.exit(1)
    
    pdf = sys.argv[1]
    dom = "auto"
    if "--domain" in sys.argv:
        idx = sys.argv.index("--domain")
        dom = sys.argv[idx+1] if idx+1 < len(sys.argv) else "auto"

    ingest(pdf, domain=dom)


        