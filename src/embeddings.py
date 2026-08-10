"""
Embedding generation for document chunks and queries using SentenceTransformer.
"""
import os
from dotenv import load_dotenv
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document

from config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """
    handles document embedding gerneration using SentenceTransformer
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        """
        Initialize the embedding manager
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            load_dotenv()

            hugging_face_api_key = os.getenv("HUGGING_FACE_API_KEY")

            print(f"Loading embedding model: {self.model_name}")

            self.model = SentenceTransformer(
                self.model_name,
                token=hugging_face_api_key
            )

            print(
                f"Model loaded successfully. "
                f"Embedding dimension: {self.model.get_embedding_dimension()}"
            )

        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[Document]) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")

        # Extract text from each Document
        page_contents = [doc.page_content for doc in texts]

        print(f"Generating embeddings for {len(page_contents)} documents...")
        embeddings = self.model.encode(
            page_contents,
            show_progress_bar=True
        )

        print(f"Generated embeddings with shape: {embeddings.shape}")

        return embeddings

    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate an embedding for a single query string.
        """
        if self.model is None:
            raise ValueError("Model not loaded")

        return self.model.encode(query)
