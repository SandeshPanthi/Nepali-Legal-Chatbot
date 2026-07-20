"""
Query-based retrieval from the vector store.
"""

from typing import List, Dict, Any

from langchain_core.documents import Document

from src.vector_store import VectorStore
from src.embeddings import EmbeddingManager


class RAGRetriever:
    """Handles query-based retrieval from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever

        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query

        Returns:
            List of dictionaries containing retrieved documents and metadata
        """

        print(f"Retreiving documents for query: '{query}")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")

        # Generate query embedding
        query_embedding = self.embedding_manager.generate_query_embedding(query)

        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                raw_scores = []
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    raw_scores.append(similarity_score)

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "document": Document(
                                page_content=document,
                                metadata=metadata
                            ),
                            "id": doc_id,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1
                        })

                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")

                # DIAGNOSTIC: candidates were found in the collection but none
                # cleared score_threshold. Surface the raw scores so it's
                # obvious whether the threshold (not the retrieval itself) is
                # the problem.
                if not retrieved_docs and raw_scores:
                    print(
                        f"No candidates passed score_threshold={score_threshold}. "
                        f"Raw similarity scores of the top {len(raw_scores)} "
                        f"candidates were: {[round(s, 3) for s in raw_scores]}"
                    )
                    print(
                        "If these scores look reasonable, try a lower min_score "
                        "(e.g. 0.0-0.1) - cosine similarity from this embedding "
                        "model rarely exceeds ~0.5-0.6 even for good matches."
                    )
            else:
                # DIAGNOSTIC: nothing came back from the collection at all -
                # almost always means the collection is empty (ingestion
                # never ran / was skipped before anything was added).
                collection_count = self.vector_store.collection.count()
                print(
                    f"No documents found in the collection "
                    f"'{self.vector_store.collection_name}' "
                    f"(collection currently holds {collection_count} documents total). "
                    "If this is 0, run ingestion first (e.g. python main.py --rebuild) "
                    "and confirm the source PDFs are in data/acts/."
                )

            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
        return []
