import os
import re
import tempfile
import streamlit as st


# Page config 
st.set_page_config(
    page_title = "RAG Medical & Legal Document QA",
    page_icon  = "📄",
    layout     = "wide",
)

# CSS
st.markdown("""
<style>
    .main-header { font-size:2rem; font-weight:700; margin-bottom:0.2rem; }
    .sub-header  { font-size:0.95rem; color:#888; margin-bottom:1rem; }
    .tech-badge  {
        display:inline-block; background:#312e81; color:#a5b4fc;
        padding:2px 10px; border-radius:20px; font-size:0.75rem; margin:2px;
    }
    .conf-high { color:#22c55e; font-weight:500; }
    .conf-med  { color:#f59e0b; font-weight:500; }
    .conf-low  { color:#ef4444; font-weight:500; }
    .history-note {
        font-size:0.8rem; color:#6366f1;
        padding: 4px 10px; background:#1e1e2e;
        border-radius:8px; display:inline-block; margin-bottom:8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📄 Medical & Legal Document QA</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Ask follow-up questions naturally — the system remembers your conversation</div>', unsafe_allow_html=True)
st.divider()


# Session state 
if "ready"          not in st.session_state: st.session_state.ready          = False
if "collection"     not in st.session_state: st.session_state.collection     = None
if "bm25_model"     not in st.session_state: st.session_state.bm25_model     = None
if "bm25_chunks"    not in st.session_state: st.session_state.bm25_chunks    = []
if "bm25_corpus"    not in st.session_state: st.session_state.bm25_corpus    = []
if "ingested_files" not in st.session_state: st.session_state.ingested_files = []
if "messages"       not in st.session_state: st.session_state.messages       = []


# Helper: get in-memory ChromaDB collection
def get_collection():
    return st.session_state.collection



#  INGESTION — runs ONCE per PDF upload
def run_ingestion(uploaded_file, domain_choice):
    import fitz, chromadb
    from rank_bm25 import BM25Okapi
    from ingest import parse_pdf, detect_domain, chunk_text, _attach_pages, embed_chunks

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        # Parse
        doc       = fitz.open(tmp_path)
        pages_raw = [(i+1, p.get_text("text")) for i, p in enumerate(doc)]
        doc.close()
        full_text = parse_pdf(tmp_path)
        st.write(f"  ✓ Parsed {len(pages_raw)} pages")

        # Detect domain
        domain = domain_choice if domain_choice != "auto" else detect_domain(full_text)
        st.session_state.last_ingested_domain = domain
        st.write(f"  ✓ Domain: **{domain}**")

        # Chunk
        chunks = chunk_text(full_text, source=uploaded_file.name, domain=domain)
        _attach_pages(chunks, pages_raw)
        sections = set(c["section"] for c in chunks)
        st.write(f"  ✓ {len(chunks)} chunks — sections: {sections}")

        # Embed
        st.write("Embedding ...")
        embeddings = embed_chunks(chunks)
        st.write(f"  ✓ Embedded {len(chunks)} chunks")

        # Store in in-memory ChromaDB

        if "chroma_client" not in st.session_state:
            st.session_state.chroma_client = chromadb.EphemeralClient()

        chroma_client = st.session_state.chroma_client

        # delete old collection and create fresh 
        try:
            chroma_client.delete_collection("rag_docs")
        except:
            pass   

        collection = chroma_client.create_collection(
            "rag_docs",
            metadata={"hnsw:space": "cosine"},
        )

        stem  = uploaded_file.name.replace(".pdf","").replace(" ","_")
        collection.upsert(
            ids        = [f"{stem}_chunk_{i}" for i in range(len(chunks))],
            embeddings = embeddings,
            documents  = [c["text"] for c in chunks],
            metadatas  = [{"domain":c["domain"],"section":c["section"],
                           "source":c["source"],"page":c["page"]} for c in chunks],
        )
        st.session_state.collection = collection
        st.write(f"  ✓ Stored {collection.count()} chunks in ChromaDB (in memory)")

        # Build BM25
        def tokenise(text):
            text = re.sub(r"[^\w\s\-/]", " ", text.lower())
            return [w for w in text.split() if len(w) >= 2]

        new_corpus = [tokenise(c["text"]) for c in chunks]
        all_chunks = st.session_state.bm25_chunks + chunks
        all_corpus = st.session_state.bm25_corpus + new_corpus
        st.session_state.bm25_model  = BM25Okapi(all_corpus)
        st.session_state.bm25_chunks = all_chunks
        st.session_state.bm25_corpus = all_corpus
        st.write(f"  ✓ BM25 index built")
        
        st.write(f"  BM25 model: {st.session_state.bm25_model is not None}")
        st.write(f"  BM25 chunks: {len(st.session_state.bm25_chunks)}")
        st.write(f"  BM25 corpus: {len(st.session_state.bm25_corpus)}")

        st.session_state.ready = True
        st.session_state.ingested_files.append(uploaded_file.name)
        return len(chunks), domain

    finally:
        os.unlink(tmp_path)


# Query function — called for each user question

def run_query(question, query_domain, history):
    
    import retriever as ret
    from qa import answer as get_answer

    # Safety check — make sure collection exists 
    if st.session_state.collection is None:
        return {
            "answer":         "Document not loaded. Please re-upload your PDF.",
            "sources":        [],
            "confidence":     0.0,
            "low_confidence": True,
            "warning":        "Collection lost from memory — please re-upload.",
        }

    # Safety check — make sure BM25 exists 
    if st.session_state.bm25_model is None:
        return {
            "answer":         "Search index not ready. Please re-upload your PDF.",
            "sources":        [],
            "confidence":     0.0,
            "low_confidence": True,
            "warning":        "BM25 index lost from memory — please re-upload.",
        }

    # Point retriever at our in-memory stores
    ret._chroma_collection = st.session_state.collection

    def load_bm25_from_session():
        return st.session_state.bm25_model, st.session_state.bm25_chunks
    ret._load_bm25 = load_bm25_from_session


    result = get_answer(
        question = question,
        domain   = query_domain,
        history  = history,       #  conversation memory
    )
    return result


#  SIDEBAR

with st.sidebar:

    st.header("📁 Upload Document")

    num_questions = len([m for m in st.session_state.messages if m["role"] == "user"])
    if st.session_state.ready:
        st.success(f"✅ Ready — {num_questions} question(s) asked")
        st.caption("Follow-up questions are supported naturally.")

    st.info(
        "**How to use:**\n"
        "1. Upload a PDF\n"
        "2. Select Document Type\n"
        "3. Click 'Process Document' *(once)*\n"
        "4. Ask a question\n"
        "5. Ask follow-up questions freely\n\n"
    )

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    domain_choice = st.selectbox(
        "Document type",
        options=["auto", "medical", "legal"],
    )

    if uploaded_file:
        if uploaded_file.name in st.session_state.ingested_files:
            st.warning(f"'{uploaded_file.name}' already loaded.")
        else:
            if st.button(" Process Document", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    try:
                        n, dom = run_ingestion(uploaded_file, domain_choice)
                        st.success(f"✅ Done! {n} chunks ready.")
                        st.balloons()
                    except Exception as e:
                        import traceback
                        st.error(f"❌ {e}")
                        st.code(traceback.format_exc())

    if st.session_state.ingested_files:
        st.divider()
        st.subheader("📂 Loaded Documents")
        for f in st.session_state.ingested_files:
            st.markdown(f"• {f}")

    st.divider()
    st.subheader("⚙️ Settings")

    query_domain = st.selectbox(
        "Answer domain",
        ["auto", "medical", "legal"],
        key="query_domain",
    )

    # Show memory depth setting
    memory_turns = st.slider(
        "Conversation memory (turns)",
        min_value = 1,
        max_value = 6,
        value     = 3,
        help      = "How many previous Q+A pairs the LLM sees. More = better context, but slower.",
        key       = "memory_turns",
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset all", use_container_width=True):
            for key in ["ready", "collection", "chroma_client", "bm25_model",
                "bm25_chunks", "bm25_corpus", "ingested_files", "messages"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.divider()
    st.markdown("""
**About**

Conversational RAG system with:
- 🧠 Multi-turn conversation memory
- 🔍 Hybrid BM25 + Vector retrieval
- 🎯 Cross-encoder reranking
- 🛡️ Hallucination detection
- 🤖 Gemini + Groq LLM
    """)



#  MAIN CHAT AREA
# Show all previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":

            turn = msg.get("turn", "")
            if turn:
                st.markdown(
                    f'<span class="history-note">🔗 Turn {turn} — '
                    f'LLM saw {msg.get("history_used", 0)} previous message(s)</span>',
                    unsafe_allow_html=True,
                )

            if msg.get("sources"):
                with st.expander("📚 Source Chunks", expanded=False):
                    for i, s in enumerate(msg["sources"], 1):
                        st.markdown(
                            f"**[{i}] {s['source']}** — "
                            f"Page {s['page']}, *{s['section']}*"
                        )
                        preview = s["text"][:300] + ("..." if len(s["text"]) > 300 else "")
                        st.caption(preview)
                        if i < len(msg["sources"]): st.divider()

            if msg.get("warning"):
                st.warning(msg["warning"])

            conf  = msg.get("confidence", 1.0)
            cls   = "conf-high" if conf >= 0.65 else "conf-med" if conf >= 0.45 else "conf-low"
            label = "High" if conf >= 0.65 else "Medium" if conf >= 0.45 else "Low"
            st.markdown(
                f'<span class="{cls}">● Confidence: {conf:.0%} ({label})</span>',
                unsafe_allow_html=True,
            )


# Chat input 
if not st.session_state.ready:
    st.info("👆 Upload and process a PDF document first.")
    st.stop()

# Show welcome message 
num_q = len([m for m in st.session_state.messages if m["role"] == "user"])
if num_q == 0:
    st.markdown("""
           Welcome! You can now ask a question about the uploaded document.
    """)
elif num_q == 1:
    st.caption("You can now ask a follow-up question — the system remembers your previous question and answer.")

# Handle new question
if question := st.chat_input("Ask a question or a follow-up..."):

    # Show the user's message immediately
    st.chat_message("user").markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # Build history = all messages BEFORE this new question

    history_to_pass = st.session_state.messages[:-1]   # exclude the just-added user message

    # Count how many messages the LLM will actually see (for display)
    memory_turns   = st.session_state.get("memory_turns", 3)
    history_used   = min(len(history_to_pass), memory_turns * 2)
    current_turn   = num_q + 1   # turn number for display

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = run_query(
                    question     = question,
                    query_domain = st.session_state.query_domain,
                    history      = history_to_pass,   #  full conversation so far
                )
            except Exception as e:
                import traceback
                result = {
                    "answer":         f"Error: {e}",
                    "sources":        [],
                    "confidence":     0.0,
                    "low_confidence": True,
                    "warning":        traceback.format_exc(),
                }

        # Show answer
        st.markdown(result["answer"])

        # Show memory indicator (only for questions after the first)
        if current_turn > 1:
            st.markdown(
                f'<span class="history-note">🔗 Turn {current_turn} — '
                f'LLM saw {history_used} previous message(s)</span>',
                unsafe_allow_html=True,
            )

        # Show sources
        if result["sources"]:
            with st.expander("📚 Source Chunks", expanded=False):
                for i, s in enumerate(result["sources"], 1):
                    st.markdown(
                        f"**[{i}] {s['source']}** — "
                        f"Page {s['page']}, *{s['section']}*"
                    )
                    preview = s["text"][:300] + ("..." if len(s["text"]) > 300 else "")
                    st.caption(preview)
                    if i < len(result["sources"]): st.divider()

        # Warning
        if result.get("warning"):
            st.warning(result["warning"])

        # Confidence
        conf  = result.get("confidence", 1.0)
        cls   = "conf-high" if conf >= 0.65 else "conf-med" if conf >= 0.45 else "conf-low"
        label = "High" if conf >= 0.65 else "Medium" if conf >= 0.45 else "Low"
        st.markdown(
            f'<span class="{cls}">● Confidence: {conf:.0%} ({label})</span>',
            unsafe_allow_html=True,
        )

    # Save full answer to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "warning": result.get("warning"),
        "turn": current_turn,
        "history_used": history_used,
    })
