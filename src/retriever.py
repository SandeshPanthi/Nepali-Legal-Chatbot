"""
Hybrid query-based retrieval using vector similarity and BM25.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document

from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager
from src.bm25 import BM25Retriever

from config import VECTOR_STORE_DIR


class RAGRetriever:
    """Handles hybrid retrieval from vector store and BM25."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager
    ):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        bm25_index_path = Path(VECTOR_STORE_DIR) / "bm25_index.pkl"
        self.bm25_retriever = BM25Retriever.load(bm25_index_path)

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        candidate_k = max(top_k *3, 10)

        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, CandidateK: {candidate_k}")

        # ---------------------------------------------------------
        # 1. VECTOR RETRIEVAL
        # ---------------------------------------------------------

        query_embedding = self.embedding_manager.generate_query_embedding(query)

        try:
            vector_results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=candidate_k
            )

            vector_retrieved_docs = []

            if vector_results["documents"] and vector_results["documents"][0]:

                documents = vector_results["documents"][0]
                metadatas = vector_results["metadatas"][0]
                distances = vector_results["distances"][0]
                ids = vector_results["ids"][0]

                for i, (doc_id, document, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):

                    similarity_score = 1 - distance

                    vector_retrieved_docs.append({
                        "document": Document(
                            page_content=document,
                            metadata=metadata
                        ),
                        "id": doc_id,
                        "similarity_score": similarity_score,
                        "rank": i + 1
                    })

            # ---------------------------------------------------------
            # 2. BM25 RETRIEVAL
            # ---------------------------------------------------------

            bm25_results = self.bm25_retriever.retrieve(
                query=query,
                top_k=candidate_k
            )

            # ---------------------------------------------------------
            # 3. DISPLAY RESULTS FOR DEBUGGING
            # ---------------------------------------------------------

            print("\nVECTOR RESULTS:")

            for result in vector_retrieved_docs:
                print(
                    f"Vector score: {result['similarity_score']:.4f} | "
                    f"Rank: {result['rank']} | "
                    f"Preview: "
                    f"{result['document'].page_content[:100]}"
                )

            print("\nBM25 RESULTS:")

            for result in bm25_results:
                print(
                    f"BM25 score: {result['bm25_score']:.4f} | "
                    f"Rank: {result['rank']} | "
                    f"Preview: "
                    f"{result['document'].page_content[:100]}"
                )

                    

            # ---------------------------------------------------------
            # 4. RECIPROCAL RANK FUSION (RRF)
            # ---------------------------------------------------------

            RRF_K = 60

            combined_documents = {}

            # Vector results
            for result in vector_retrieved_docs:
                doc_id = result["id"]

                combined_documents[doc_id] = {
                    "document": result["document"],
                    "id": doc_id,
                    "vector_rank": result["rank"],
                    "bm25_rank": None,
                    "vector_score": result["similarity_score"],
                    "bm25_score": 0.0,
                    "rrf_score": 1 / (RRF_K + result["rank"])
                }

            # BM25 results
            for result in bm25_results:
                doc_id = result["id"]

                if doc_id not in combined_documents:
                    combined_documents[doc_id] = {
                        "document": result["document"],
                        "id": doc_id,
                        "vector_rank": None,
                        "bm25_rank": result["rank"],
                        "vector_score": 0.0,
                        "bm25_score": result["bm25_score"],
                        "rrf_score": 1 / (RRF_K + result["rank"])
                    }

                else:
                    combined_documents[doc_id]["bm25_rank"] = result["rank"]
                    combined_documents[doc_id]["bm25_score"] = result["bm25_score"]

                    combined_documents[doc_id]["rrf_score"] += (
                        1 / (RRF_K + result["rank"])
                    )

            # ---------------------------------------------------------
            # 5. SORT BY RRF SCORE
            # ---------------------------------------------------------

            hybrid_results = sorted(
                combined_documents.values(),
                key=lambda x: x["rrf_score"],
                reverse=True
            )

            # Keep only final top_k results
            hybrid_results = hybrid_results[:top_k]

            # Add final rank
            for rank, result in enumerate(hybrid_results, start=1):
                result["rank"] = rank

                # Did both retrieval systems find this document?
                result["retrieval_agreement"] = (
                    result["vector_rank"] is not None
                    and result["bm25_rank"] is not None
                )

            return [
                {
                    "document": result["document"],
                    "id": result["id"],
                    "retrieval_score": result["rrf_score"],
                    "vector_score": result["vector_score"],
                    "bm25_score": result["bm25_score"],
                    "vector_rank": result["vector_rank"],
                    "bm25_rank": result["bm25_rank"],
                    "retrieval_agreement": result["retrieval_agreement"],
                    "rrf_score": result["rrf_score"],
                    "rank": result["rank"],
                }
                for result in hybrid_results
            ]

            

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []