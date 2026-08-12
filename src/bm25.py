import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


class BM25Retriever:
    """Keyword-based retrieval using BM25."""

    def __init__(
        self,
        documents: List[Document],
        ids: List[str]
    ):
        self.documents = documents
        self.ids = ids

        tokenized_documents = [
            self._tokenize(doc.page_content)
            for doc in documents
        ]

        self.bm25 = BM25Okapi(tokenized_documents)

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace-based tokenization."""
        return text.lower().split()

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1][:top_k]

        results = []

        for rank, index in enumerate(ranked_indices, start=1):
            results.append({
                "document": self.documents[index],
                "id": self.ids[index],
                "bm25_score": float(scores[index]),
                "rank": rank
            })

        return results

    def save(self, path: Path):
        """Save the BM25 index, documents, and IDs to disk."""

        with open(path, "wb") as file:
            pickle.dump(
                {
                    "documents": self.documents,
                    "ids": self.ids,
                    "bm25": self.bm25,
                },
                file
            )

    @classmethod
    def load(cls, path: Path):
        """Load a previously saved BM25 index."""

        with open(path, "rb") as file:
            data = pickle.load(file)

        instance = cls.__new__(cls)

        instance.documents = data["documents"]
        instance.ids = data["ids"]
        instance.bm25 = data["bm25"]

        return instance