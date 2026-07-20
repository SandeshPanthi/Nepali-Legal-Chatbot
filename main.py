"""
Entry point for the Legal RAG pipeline.

Builds the vector store index from the legal PDFs (Civil Code, Penal
Code, Constitution) — reusing an existing index if one is already
persisted — and then runs a query against it, printing the answer,
sources, confidence score, and a preview of the retrieved context.

Usage:
    python main.py "Punishment for robbery"
    python main.py                          # will prompt for a query
    python main.py --rebuild "some query"    # force re-embedding first
    python main.py --min-score 0.0 "some query"   # loosen the score filter
"""

import argparse

from config import DEFAULT_TOP_K, DEFAULT_MIN_SCORE
from src.ingest import build_index
from src.retriever import RAGRetriever
from src.rag_pipeline import rag_llm, get_llm


def parse_args():
    parser = argparse.ArgumentParser(description="Query the Legal RAG pipeline.")
    parser.add_argument("query", nargs="*", help="The legal question to ask")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-embedding of all source PDFs even if the vector store already has data",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve")
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help="Minimum similarity score for a retrieved chunk to be kept",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Build (or reuse) the vector index ---
    vectorstore, embedding_manager = build_index(force_rebuild=args.rebuild)

    # --- Set up retriever and LLM ---
    rag_retriever = RAGRetriever(vectorstore, embedding_manager)
    llm = get_llm()

    # --- Get the query ---
    query = " ".join(args.query).strip()
    if not query:
        query = input("Enter your legal query: ")

    # --- Run the RAG pipeline and show the result ---
    result = rag_llm(
        query,
        rag_retriever,
        llm,
        top_k=args.top_k,
        min_score=args.min_score,
        return_context=True,
    )
    print("\n--------------------------------------------------------------------------------------\n")
    print("Answer:", result['answer'])
    print("\n--------------------------------------------------------------------------------------\n")
    print("Confidence:", result['confidence'])
    print("\n--------------------------------------------------------------------------------------\n")
    print("Sources:", result['sources'])
    # if 'context' in result:
    #     print("Context Preview:", result['context'][:300])


if __name__ == "__main__":
    main()
