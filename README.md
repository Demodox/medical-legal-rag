# 📄 Medical & Legal Document QA System

A conversational AI system that answers questions from medical and legal PDF documents. Upload a PDF, ask questions one by one, and get cited answers with page references.



## What it does

- Upload any medical or legal PDF
- Ask questions in natural language
- Get answers with exact page and section citations
- Ask follow-up questions — the system remembers the conversation
- Automatically detects if the document is medical or legal

---

## How it works

```
PDF → Parse → Chunk → Embed → Store
                                 ↓
Question → Search (Vector + BM25) → Rerank → LLM → Cited Answer
```

1. **Ingestion** — PDF is parsed, split into chunks, and stored in ChromaDB + BM25 index
2. **Retrieval** — Question is searched using both vector similarity and keyword matching
3. **Reranking** — Cross-encoder picks the best 4 chunks from the top 10
4. **Answer** — LLM reads those 4 chunks and answers with citations
5. **Hallucination check** — Flags if the answer is not grounded in the document

---

## Tech Stack

- **Embeddings** — sentence-transformers (all-MiniLM-L6-v2)
- **Vector store** — ChromaDB
- **Keyword search** — BM25 (rank-bm25)
- **Reranker** — CrossEncoder (ms-marco-MiniLM-L-6-v2)
- **LLM** — Gemini 2.0 Flash (primary) + Groq Llama 3.3 70B (fallback)
- **UI** — Streamlit
- **PDF parsing** — PyMuPDF

---

## Project Files

```
rag-doc-qa/
├── ingest.py       # Parse PDF, create chunks, build search indexes
├── retriever.py    # Hybrid BM25 + vector search with RRF fusion
├── qa.py           # Rerank, call LLM, check hallucination
├── app.py          # Streamlit UI
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**2. Add API keys**
```bash
cp .env.example .env
```
Edit `.env`:
```
GOOGLE_API_KEY=your_key_from_aistudio.google.com
GROQ_API_KEY=your_key_from_console.groq.com
```
Both are free — no credit card needed.

**3. Run**
```bash
streamlit run app.py
```

