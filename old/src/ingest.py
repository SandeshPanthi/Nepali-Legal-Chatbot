"""
Ingestion pipeline: load each legal PDF, parse it into structured chunks,
generate embeddings, and store everything in the vector store.

This mirrors the notebook's end-to-end flow (loading -> parsing ->
combining -> embedding -> storing) with no logic changes.
"""

from config import (
    CIVIL_CODE_PATH,
    PENAL_CODE_PATH,
    CONSTITUTION_PATH,
    VECTOR_STORE_DIR,
)
from src.document_loader import load_civil_code, load_penal_code, load_constitution
from src.parsers import parse_civil_code, parse_penal_code, parse_constitution
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


def build_index():
    """
    Loads, parses, embeds and stores the Civil Code, Penal Code and
    Constitution documents. Returns the populated VectorStore and the
    EmbeddingManager used to embed the query at retrieval time.
    """

    # --- Civil Code -------------------------------------------------------
    civil_documents = load_civil_code(CIVIL_CODE_PATH)
    civil_code_text = "\n".join(doc.page_content for doc in civil_documents)
    civil_code_chunks = parse_civil_code(civil_code_text)
    print(len(civil_code_chunks))

    # --- Penal Code ---------------------------------------------------------
    penal_documents = load_penal_code(PENAL_CODE_PATH)
    penal_code_text = "\n".join(doc.page_content for doc in penal_documents)
    penal_code_chunks = parse_penal_code(penal_code_text)
    print(len(penal_code_chunks))

    # --- Constitution ---------------------------------------------------------
    constitution_documents = load_constitution(CONSTITUTION_PATH)
    constitution_text = "\n".join(doc.page_content for doc in constitution_documents)
    constitution_chunks = parse_constitution(constitution_text)
    print(len(constitution_chunks))

    # Combine all chunks into a single flat list
    all_legal_chunks = civil_code_chunks + penal_code_chunks + constitution_chunks
    print(f"Total chunks unified: {len(all_legal_chunks)}")

    ### Generate embeddings
    embedding_manager = EmbeddingManager()
    embeddings = embedding_manager.generate_embeddings(all_legal_chunks)

    # Store in the vector database
    vectorstore = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
    try:
        vectorstore.client.delete_collection(vectorstore.collection_name)
        print("Old collection deleted.")
    except Exception:
        print("Collection did not exist.")
    vectorstore = VectorStore(persist_directory=str(VECTOR_STORE_DIR))

    vectorstore.add_documents(all_legal_chunks, embeddings)

    return vectorstore, embedding_manager
