"""
Ingestion pipeline: load each legal PDF, parse it into structured chunks,
generate embeddings, and store everything in the vector store.

The vector store is persisted to disk (ChromaDB persist directory), so by
default this SKIPS re-embedding if the collection already has documents in
it from a previous run. Pass force_rebuild=True (or run main.py --rebuild)
to wipe the collection and re-embed everything from scratch.
"""

from config import (
    CIVIL_CODE_PATH,
    PENAL_CODE_PATH,
    CONSTITUTION_PATH,
    DATA_DIR,
    VECTOR_STORE_DIR,
)
from src.document_loader import load_civil_code, load_penal_code, load_constitution
from src.parsers import parse_civil_code, parse_penal_code, parse_constitution
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


def _check_source_pdfs_exist():
    """Fail fast with a clear message if the source PDFs are missing."""
    missing = [
        path.name for path in (CIVIL_CODE_PATH, PENAL_CODE_PATH, CONSTITUTION_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing source PDF(s): {', '.join(missing)}. "
            f"Place civil_code.pdf, penal_code.pdf and constitution.pdf inside "
            f"{DATA_DIR} before running."
        )


def build_index(force_rebuild: bool = False):
    """
    Loads, parses, embeds and stores the Civil Code, Penal Code and
    Constitution documents. Returns the populated VectorStore and the
    EmbeddingManager used to embed the query at retrieval time.

    If the vector store already contains documents from a previous run,
    ingestion is skipped and the existing store is reused — unless
    force_rebuild=True, in which case the collection is deleted and
    everything is re-embedded from scratch.
    """

    vectorstore = VectorStore(persist_directory=str(VECTOR_STORE_DIR))
    embedding_manager = EmbeddingManager()

    existing_count = vectorstore.collection.count()

    if existing_count > 0 and not force_rebuild:
        print(
            f"Vector store already has {existing_count} documents — "
            f"skipping ingestion and reusing the existing index. "
            f"(pass --rebuild to force re-embedding)"
        )
        return vectorstore, embedding_manager

    if force_rebuild and existing_count > 0:
        try:
            vectorstore.client.delete_collection(vectorstore.collection_name)
            print("Old collection deleted.")
        except Exception:
            print("Collection did not exist.")
        vectorstore = VectorStore(persist_directory=str(VECTOR_STORE_DIR))

    _check_source_pdfs_exist()

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

    if not all_legal_chunks:
        print(
            "WARNING: 0 chunks were produced from the source PDFs. "
            "The regex-based parsers may not match this PDF's text layout "
            "(check the raw extracted text). The vector store will remain empty."
        )
        return vectorstore, embedding_manager

    ### Generate embeddings
    embeddings = embedding_manager.generate_embeddings(all_legal_chunks)

    # Store in the vector database
    vectorstore.add_documents(all_legal_chunks, embeddings)

    return vectorstore, embedding_manager
