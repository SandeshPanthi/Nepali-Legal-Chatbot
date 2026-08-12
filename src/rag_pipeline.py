"""
Integrates the vector-store retrieval pipeline with the Groq LLM to
produce grounded legal answers with sources and a confidence score.
"""

import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

from config import GROQ_MODEL_NAME, GROQ_TEMPERATURE, GROQ_MAX_TOKENS

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


def rag_llm(query, retriever, llm, top_k=5, return_context=False):
    """Rag pipeline with features:
        return answer, sources, confidence score
    """
    results = retriever.retrieve(query, top_k=top_k)
    if not results:
        return {'answer': "No relevant context found.", 'sources': [], 'confidence': 0.0, 'context': ''}

    ##Prepare context and sources
    context = "\n\n".join(
        f"""Document: {doc["document"].metadata.get("title", "Unknown")}
                Part: {doc["document"].metadata.get("part", "")}
                Chapter: {doc["document"].metadata.get("chapter", "")}
                Section: {doc["document"].metadata.get("section_title", "")}

                {doc["document"].page_content}
            """
        for doc in results
    )
    sources = []

    for doc in results:
        metadata = doc["document"].metadata

        sources.append({
        "id": doc["id"],
        "source": metadata.get(
            "source_file",
            metadata.get("title", "Unknown")
        ),
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

    # Generate answer

    prompt = f"""
    
        You are a legal assistant.

        Answer the user's question using ONLY the provided legal context.
        When answering, always mention:
        - the document name (Civil Code, Penal Code, Constitution)
        - the section/article number whenever available.

        If the context does not contain the answer, say:
        "I couldn't find the answer in the provided legal documents." \nContext: \n{context}\n\n Question: {query}\n\nAnswer:
            
        """

    # ---------------------------------------------------------
    # RETRIEVAL CONFIDENCE
    # ---------------------------------------------------------

    agreement_count = sum(
        1
        for doc in results
        if doc["retrieval_agreement"]
    )

    agreement_ratio = agreement_count / len(results)

    top_result = results[0]

    confidence = {
        "top_vector_score": round(
            top_result["vector_score"], 4
        ),

        "top_bm25_score": round(
            top_result["bm25_score"], 4
        ),

        "top_rrf_score": round(
            top_result["rrf_score"], 6
        ),

        "retrieved_documents": len(results),

        "agreement_count": agreement_count,

        "agreement_ratio": round(
            agreement_ratio, 2
        )
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
