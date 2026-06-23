import os
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
from retriever import retrieve

load_dotenv() 

# Settings

EMBED_MODEL        = "all-MiniLM-L6-v2"
RERANK_MODEL       = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_N_AFTER_RERANK = 4

MAX_HISTORY_TURNS = 3

HALLUC_THRESHOLD = {
    "medical": 0.60,
    "legal":   0.50,
}

# Load models
_embed_model   = SentenceTransformer(EMBED_MODEL)
_cross_encoder = CrossEncoder(RERANK_MODEL)



# Prompt templates 
MEDICAL_PROMPT = """\
You are a medical document assistant having a conversation with a user.
Answer ONLY using the document context below.
Never infer or guess dosages or diagnoses.
If the answer is not in the context, say: "This information is not in the provided documents."
Always cite your source as [Page X, Section: Y].
Use the conversation history to understand follow-up questions.
{history}
DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {question}

ANSWER:"""


LEGAL_PROMPT = """\
You are a legal document assistant having a conversation with a user.
Answer ONLY using the document context below.
Preserve exact legal language — do not paraphrase clauses.
If the answer is not in the context, say: "This clause is not in the provided documents."
Always cite your source as [Page X, Clause: Y].
Use the conversation history to understand follow-up questions.
{history}
DOCUMENT CONTEXT:
{context}

CURRENT QUESTION: {question}

ANSWER:"""

# Format chat history into a readable string
def _format_history(history, max_turns=MAX_HISTORY_TURNS):
    if not history:
        return ""

    # question + answer = 2 messages per turn
    max_items = max_turns * 2

    # If history is very long, we slice from the end
    recent_history = history[-max_items:]

    # Build the formatted string
    lines = ["CONVERSATION SO FAR:"]

    for msg in recent_history:
        if msg["role"] == "user":
            lines.append(f"User asked: {msg['content']}")
        elif msg["role"] == "assistant":
            lines.append(f"You answered: {msg['content']}")

    # Add a blank line after history to separate it from the context
    lines.append("")

    return "\n".join(lines) + "\n"

# Build query using history
def _build_search_query(question, history):

    if not history:
        return question

    # Find the last assistant message (most recent answer)
    last_answer = ""
    for msg in reversed(history):
        if msg["role"] == "assistant":
            last_answer = msg["content"]
            break

    if not last_answer:
        return question

    # Combine: current question + first 150 chars of last answer
    expanded_query = question + " " + last_answer[:150]
    return expanded_query

#  STEP 1 — Rerank chunks
def rerank(question, chunks, top_n=TOP_N_AFTER_RERANK):

    if not chunks:
        return []

    # Score each (question, chunk) pair together
    pairs  = [(question, chunk["text"]) for chunk in chunks]
    scores = _cross_encoder.predict(pairs)

    # Sort by score, highest first
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

    return [chunk for score, chunk in ranked[:top_n]]

#  STEP 2 — Build prompt with history + context
def _build_prompt(question, chunks, domain, history):

    # Format the source chunks as numbered excerpts
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        label = (
            f"[Excerpt {i} | "
            f"File: {chunk['source']} | "
            f"Page: {chunk['page']} | "
            f"Section: {chunk['section']}]"
        )
        context_parts.append(label + "\n" + chunk["text"])

    context = "\n\n---\n\n".join(context_parts)

    # Format the conversation history
    history_text = _format_history(history)

    # Choose prompt template based on domain
    template = MEDICAL_PROMPT if domain == "medical" else LEGAL_PROMPT

    return template.format(
        history  = history_text,
        context  = context,
        question = question.strip(),
    )


#  STEP 3 — Call LLM (Gemini primary, Groq fallback)
def _call_llm(prompt):

    google_key = os.getenv("GOOGLE_API_KEY", "")
    groq_key   = os.getenv("GROQ_API_KEY",   "")

    # Try Gemini first
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(
                model = "gemini-2.0-flash",
                google_api_key = google_key,
                temperature = 0.0,
            )
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"  Gemini failed: {e} — trying Groq...")

    # Fallback to Groq
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                model = "llama-3.3-70b-versatile",
                api_key = groq_key,
                temperature = 0.0,
            )
            return llm.invoke(prompt).content.strip()
        except Exception as e:
            print(f"  Groq also failed: {e}")

    raise RuntimeError(
        "No LLM available. Add GOOGLE_API_KEY or GROQ_API_KEY to your .env file."
    )

#  STEP 4 — Hallucination check
def _check_hallucination(answer, chunks, domain):

    if not answer.strip() or not chunks:
        return {"score": 0.0, "low_confidence": True, "warning": "Empty answer."}

    threshold = HALLUC_THRESHOLD.get(domain, 0.60)

    # for answers that are very short, we can be more lenient with the threshold
    if len(answer.split()) < 30:
        threshold = min(threshold, 0.35)

    answer_vec = _embed_model.encode(answer, normalize_embeddings=True)
    chunk_vecs = _embed_model.encode(
        [c["text"] for c in chunks], normalize_embeddings=True
    )

    # Cosine similarity = dot product for normalised vectors
    sims = [float(np.dot(answer_vec, cv)) for cv in chunk_vecs]
    best_score = max(sims)
    low_conf = best_score < threshold

    return {
        "score":          round(best_score, 3),
        "low_confidence": low_conf,
        "warning": (
            f"Low confidence ({best_score:.2f} < {threshold:.2f}). "
            "Answer may not be fully grounded in the documents."
        ) if low_conf else None,
    }

#  MAIN FUNCTION — answer with conversation memory
def answer(question, domain="auto", history=None):

    # Default to empty history if not provided
    if history is None:
        history = []

    # Auto-domain Detection
    if domain == "auto":
        q = question.lower()

        legal_words= ["clause","contract","nda","agreement","party",
                  "arbitration","liquidated","termination","warranty",
                  "governing","jurisdiction","indemnif","confidential",
                  "corporation","llc","incorporated","delaware","liability"]
        
        medical_words = ["patient","dosage","medication","diagnosis",
                    "treatment","symptoms","prescription","mg","doctor",
                    "surgery","hospital","injury","clinical"]
        
        leg_score = sum(1 for w in legal_words   if w in q)
        med_score = sum(1 for w in medical_words if w in q)
        
        if leg_score > med_score:
            domain = "legal"
        elif med_score > leg_score:
            domain = "medical"
        else:
            # couldn't decide based on keywords — check what domain was ingested from session_state
            import streamlit as st
            ingested = getattr(st.session_state, "last_ingested_domain", None)
            domain = ingested if ingested else "medical"


    print(f"\nQuestion : {question}")
    print(f"Domain   : {domain}")
    print(f"History  : {len(history)} messages")

    # Step 1: Build expanded search query 

    search_query = _build_search_query(question, history)
    if search_query != question:
        print(f"  Expanded query: {search_query[:80]}...")

    # Step 2: Hybrid retrieval 
    print("  [1/4] Retrieving candidates...")
    candidates = retrieve(search_query, domain=domain, top_k=10)

    if not candidates:
        return {
            "answer":"No relevant documents found. Please ingest a document first.",
            "sources": [],
            "confidence": 0.0,
            "low_confidence": True,
            "warning": "No documents ingested.",
        }

    # Step 3: Rerank 
    print(f"  [2/4] Reranking {len(candidates)} → best {TOP_N_AFTER_RERANK}...")
    top_chunks = rerank(question, candidates, top_n=TOP_N_AFTER_RERANK)

    # Step 4: Build prompt with history + call LLM 
    print("  [3/4] Generating answer with conversation context...")
    prompt = _build_prompt(question, top_chunks, domain, history)
    answer_text = _call_llm(prompt)

    print(f"  Answer: {answer_text[:100]}...")

    # Step 5: Hallucination check 
    print("  [4/4] Checking answer quality...")
    confidence = _check_hallucination(answer_text, top_chunks, domain)

    status = "✅ grounded" if not confidence["low_confidence"] else "⚠  low confidence"
    print(f"  Confidence: {confidence['score']} — {status}")

    return {
        "answer": answer_text,
        "sources": top_chunks,
        "confidence": confidence["score"],
        "low_confidence": confidence["low_confidence"],
        "warning": confidence["warning"],
    }


# Run from terminal 
if __name__ == "__main__":
    import sys
    question = sys.argv[1] if len(sys.argv) > 1 else "What medications were prescribed?"
    domain   = sys.argv[2] if len(sys.argv) > 2 else "medical"
    result   = answer(question, domain=domain)
    print(f"\nANSWER:\n{result['answer']}")
    print(f"\nConfidence: {result['confidence']}")
    if result["warning"]:
        print(result["warning"])
