"""
Integrates the vector-store retrieval pipeline with the Groq LLM.
Uses a Router pattern: LEGAL queries trigger RAG, GENERAL queries are pure chat.
"""
import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from config import GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS
from src.router import classify_query

load_dotenv()


def get_llm():
    """Create the ChatGroq LLM client using the GROQ_API_KEY from the environment."""
    groq_api_key = os.getenv("GROQ_API_KEY")

    llm = ChatGroq(
        model_name=GROQ_MODEL_NAME,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )
    return llm


def general_chat_response(query: str, llm, history=None) -> dict:
    """Natural conversation with the LLM — no retrieval, no legal grounding needed."""
    history_text = ""
    if history:
        history_lines = []
        for msg in history[-6:]:  # last 6 exchanges
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg['content']}")
        history_text = "\n\n".join(history_lines)
        history_text = f"\n\nPrevious conversation:\n{history_text}\n"

    prompt = f"""You are a helpful, friendly, and knowledgeable AI assistant. You are also a Nepali Legal Assistant, but right now the user is just having a general conversation with you.

Respond naturally and helpfully. If the user tells you their name, remembers it. If they ask a joke, tell one. If they ask about science, history, math, or general knowledge, answer from your training knowledge. Be warm and conversational.

If the user asks something that IS about Nepali law, politely mention that you can also help with legal questions about Nepal's Constitution, Civil Code, and Penal Code when needed.

{history_text}

User: {query}

Assistant:"""

    response = llm.invoke(prompt)

    return {
        'answer': response.content,
        'sources': [],
        'confidence': {
            'top_vector_score': 0.0,
            'top_bm25_score': 0.0,
            'top_rrf_score': 0.0,
            'retrieved_documents': 0,
            'agreement_count': 0,
            'agreement_ratio': 0.0
        }
    }


def rag_llm(query, retriever, llm, top_k=5, return_context=False):
    """Backward-compatible RAG pipeline."""
    return rag_llm_with_history(query, retriever, llm, history=[], top_k=top_k, return_context=return_context)


def rag_llm_with_history(query, retriever, llm, history=None, top_k=5, return_context=False):
    """
    Router RAG pipeline:
    1. Classify query as LEGAL or GENERAL
    2. GENERAL → pure LLM chat
    3. LEGAL → hybrid retrieval + grounded generation
    """
    history = history or []

    # ROUTER: Let the LLM decide if this needs legal retrieval
    classification = classify_query(query, llm)

    if classification == "GENERAL":
        return general_chat_response(query, llm, history)

    # --- LEGAL PATH: Full RAG ---
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {
            'answer': "I couldn't find the answer in the provided legal documents. Could you rephrase your question or ask about a specific legal topic?",
            'sources': [],
            'confidence': {
                'top_vector_score': 0.0,
                'top_bm25_score': 0.0,
                'top_rrf_score': 0.0,
                'retrieved_documents': 0,
                'agreement_count': 0,
                'agreement_ratio': 0.0
            },
            'context': ''
        }

    # Prepare context and sources
    context = "\n\n".join(
        f"""Document: {doc["document"].metadata.get("title", "Unknown")}
Part: {doc["document"].metadata.get("part", "")}
Chapter: {doc["document"].metadata.get("chapter", "")}
Section: {doc["document"].metadata.get("section_title", "")}

{doc["document"].page_content}"""
        for doc in results
    )

    sources = []
    for doc in results:
        metadata = doc["document"].metadata
        sources.append({
            "id": doc["id"],
            "source": metadata.get("source_file", metadata.get("title", "Unknown")),
            "title": metadata.get("title", "Unknown"),
            "page": metadata.get("page", "unknown"),
            "section": metadata.get("section_title", "unknown"),
            "vector_rank": doc["vector_rank"],
            "bm25_rank": doc["bm25_rank"],
            "retrieval_agreement": doc["retrieval_agreement"],
            "vector_score": doc["vector_score"],
            "bm25_score": doc["bm25_score"],
            "rrf_score": doc["rrf_score"],
            "rank": doc["rank"]
        })

    # Build history string
    history_text = ""
    if history:
        history_lines = []
        for msg in history:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            history_lines.append(f"{prefix}: {msg['content']}")
        history_text = "\n\n".join(history_lines)
        history_text = f"\n\nPrevious conversation:\n{history_text}\n"

    # Generate grounded legal answer
    prompt = f"""You are a legal assistant for Nepali law. Answer the user's question using ONLY the provided legal context.

When answering, always mention:
- the document name (Civil Code, Penal Code, Constitution)
- the section/article number whenever available.

If the context does not contain the answer, say: "I couldn't find the answer in the provided legal documents."

---{history_text}

Legal Context:
{context}

User's current question: {query}

Answer:"""

    # Retrieval confidence
    agreement_count = sum(1 for doc in results if doc["retrieval_agreement"])
    agreement_ratio = agreement_count / len(results) if results else 0
    top_result = results[0]

    confidence = {
        "top_vector_score": round(top_result["vector_score"], 4),
        "top_bm25_score": round(top_result["bm25_score"], 4),
        "top_rrf_score": round(top_result["rrf_score"], 6),
        "retrieved_documents": len(results),
        "agreement_count": agreement_count,
        "agreement_ratio": round(agreement_ratio, 2)
    }

    response = llm.invoke(prompt)

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output