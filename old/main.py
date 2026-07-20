"""
Entry point for the Legal RAG pipeline.

Builds the vector store index from the legal PDFs (Civil Code, Penal
Code, Constitution) and then runs a query against it, printing the
answer, sources, confidence score, and a preview of the retrieved
context — matching the original notebook's example usage.

Usage:
    python main.py "Punishment for robbery"
    python main.py                     # will prompt for a query
"""

import sys

from config import DEFAULT_TOP_K, DEFAULT_MIN_SCORE
from src.ingest import build_index
from src.retriever import RAGRetriever
from src.rag_pipeline import rag_llm, get_llm


def main():
    # --- Build (load, parse, embed, store) the vector index ---
    vectorstore, embedding_manager = build_index()

    # --- Set up retriever and LLM ---
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = get_llm()

    # --- Get the query ---
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        query = input("Enter your legal query: ")

    # --- Run the RAG pipeline and show the result ---
    result = rag_llm(
        query,
        rag_retriever,
        llm,
        top_k=DEFAULT_TOP_K,
        min_score=DEFAULT_MIN_SCORE,
        return_context=True,
    )

    print("Answer:", result['answer'])
    print("Sources:", result['sources'])
    print("Confidence:", result['confidence'])
    if 'context' in result:
        print("Context Preview:", result['context'][:300])


if __name__ == "__main__":
    main()
