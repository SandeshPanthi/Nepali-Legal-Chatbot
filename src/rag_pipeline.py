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


def rag_llm(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """Rag pipeline with features:
        return answer, sources, confidence score
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
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
            "source": metadata.get("title", metadata.get("source", "unknown")),
            "page": metadata.get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["document"].page_content[:120] + "......"
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
    confidence = max([doc['similarity_score'] for doc in results])
    response = llm.invoke(prompt)

    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output
