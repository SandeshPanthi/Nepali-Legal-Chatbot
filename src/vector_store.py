"""
ChromaDB-backed vector store for persisting document embeddings.
"""

import os
from typing import List, Dict, Any, Tuple

import numpy as np
import chromadb
from chromadb.config import Settings

from config import COLLECTION_NAME, VECTOR_STORE_DIR


class VectorStore:
    """ Manages document embeddings in a ChromaDB vector store"""

    def __init__(self, collection_name: str = COLLECTION_NAME, persist_directory: str = str(VECTOR_STORE_DIR)):
        """"
        Initialize the vector store

        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""

        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(
        self,
        documents: List[Any],
        embeddings: np.ndarray,
        ids: List[str]
    ):
        """
        Add documents and their embeddings to the vector store.

        The provided IDs are preserved so that the same document IDs
        can be used by both ChromaDB and BM25.
        """

        if not (
            len(documents) == len(embeddings) == len(ids)
        ):
            raise ValueError(
                "Number of documents, embeddings, and IDs must match"
            )

        print(
            f"Adding {len(documents)} documents to vector store....."
        )

        metadatas = []
        documents_text = []
        embeddings_list = []

        for i, (doc, embedding, doc_id) in enumerate(
            zip(documents, embeddings, ids)
        ):
            # Prepare metadata
            metadata = {
                key: ("" if value is None else value)
                for key, value in doc.metadata.items()
            }

            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)

            metadatas.append(metadata)

            # Document content
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )

            print(
                f"Successfully added {len(documents)} documents "
                f"to vector store"
            )

            print(
                f"Total documents in collection: "
                f"{self.collection.count()}"
            )

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise