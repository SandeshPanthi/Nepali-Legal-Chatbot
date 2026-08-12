from pathlib import Path

from src.bm25 import BM25Retriever

from config import (
    CIVIL_CODE_PATH,
    PENAL_CODE_PATH,
    CONSTITUTION_PATH,
    DATA_DIR,
    VECTOR_STORE_DIR,
)

from src.document_loader import (
    load_civil_code,
    load_penal_code,
    load_constitution,
)

from src.parsers import (
    parse_civil_code,
    parse_penal_code,
    parse_constitution,
)

from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore


def _check_source_pdfs_exist():
    """Fail fast with a clear message if the source PDFs are missing."""

    missing = [
        path.name
        for path in (
            CIVIL_CODE_PATH,
            PENAL_CODE_PATH,
            CONSTITUTION_PATH,
        )
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"Missing source PDF(s): {', '.join(missing)}. "
            f"Place civil_code.pdf, penal_code.pdf and constitution.pdf "
            f"inside {DATA_DIR} before running."
        )


def build_index(force_rebuild: bool = False):
    """
    Loads, parses, embeds and stores the Civil Code, Penal Code
    and Constitution documents.

    The same document IDs are used by both:
    - ChromaDB vector retrieval
    - BM25 retrieval

    This allows RRF to merge results using document IDs.
    """

    # ---------------------------------------------------------
    # 1. INITIALIZE VECTOR STORE + EMBEDDING MODEL
    # ---------------------------------------------------------

    vectorstore = VectorStore(
        persist_directory=str(VECTOR_STORE_DIR)
    )

    embedding_manager = EmbeddingManager()

    existing_count = vectorstore.collection.count()

    # ---------------------------------------------------------
    # 2. REUSE EXISTING INDEX
    # ---------------------------------------------------------

    if existing_count > 0 and not force_rebuild:
        print(
            f"Vector store already has {existing_count} documents — "
            f"skipping ingestion and reusing the existing index. "
            f"(pass --rebuild to force re-embedding)"
        )

        return vectorstore, embedding_manager

    # ---------------------------------------------------------
    # 3. DELETE OLD VECTOR STORE IF REBUILDING
    # ---------------------------------------------------------

    if force_rebuild and existing_count > 0:

        try:
            vectorstore.client.delete_collection(
                vectorstore.collection_name
            )

            print("Old collection deleted.")

        except Exception:
            print("Collection did not exist.")

        vectorstore = VectorStore(
            persist_directory=str(VECTOR_STORE_DIR)
        )

    # ---------------------------------------------------------
    # 4. CHECK SOURCE PDFs
    # ---------------------------------------------------------

    _check_source_pdfs_exist()

    # ---------------------------------------------------------
    # 5. LOAD + PARSE CIVIL CODE
    # ---------------------------------------------------------

    civil_documents = load_civil_code(
        CIVIL_CODE_PATH
    )

    civil_code_text = "\n".join(
        doc.page_content
        for doc in civil_documents
    )

    civil_code_chunks = parse_civil_code(
        civil_code_text
    )

    print(len(civil_code_chunks))

    # ---------------------------------------------------------
    # 6. LOAD + PARSE PENAL CODE
    # ---------------------------------------------------------

    penal_documents = load_penal_code(
        PENAL_CODE_PATH
    )

    penal_code_text = "\n".join(
        doc.page_content
        for doc in penal_documents
    )

    penal_code_chunks = parse_penal_code(
        penal_code_text
    )

    print(len(penal_code_chunks))

    # ---------------------------------------------------------
    # 7. LOAD + PARSE CONSTITUTION
    # ---------------------------------------------------------

    constitution_documents = load_constitution(
        CONSTITUTION_PATH
    )

    constitution_text = "\n".join(
        doc.page_content
        for doc in constitution_documents
    )

    constitution_chunks = parse_constitution(
        constitution_text
    )

    print(len(constitution_chunks))

    # ---------------------------------------------------------
    # 8. COMBINE ALL CHUNKS
    # ---------------------------------------------------------

    all_legal_chunks = (
        civil_code_chunks
        + penal_code_chunks
        + constitution_chunks
    )

    print(
        f"Total chunks unified: {len(all_legal_chunks)}"
    )

    # ---------------------------------------------------------
    # 9. CHECK WHETHER CHUNKS WERE CREATED
    # ---------------------------------------------------------

    if not all_legal_chunks:

        print(
            "WARNING: 0 chunks were produced from the source PDFs. "
            "The regex-based parsers may not match this PDF's text layout "
            "(check the raw extracted text). The vector store will remain empty."
        )

        return vectorstore, embedding_manager

    # ---------------------------------------------------------
    # 10. GENERATE SHARED DOCUMENT IDs
    # ---------------------------------------------------------
    #
    # These IDs are used by BOTH:
    #
    #       BM25
    #         +
    #      ChromaDB
    #
    # This allows RRF to merge results using IDs.
    # ---------------------------------------------------------

    document_ids = [
        f"doc_{i}"
        for i in range(len(all_legal_chunks))
    ]

    print(
        f"Generated {len(document_ids)} shared document IDs."
    )

    # ---------------------------------------------------------
    # 11. BUILD BM25 INDEX
    # ---------------------------------------------------------

    bm25_retriever = BM25Retriever(
        documents=all_legal_chunks,
        ids=document_ids,
    )

    bm25_index_path = (
        Path(VECTOR_STORE_DIR)
        / "bm25_index.pkl"
    )

    bm25_retriever.save(
        bm25_index_path
    )

    print(
        f"BM25 index saved to: {bm25_index_path}"
    )

    # ---------------------------------------------------------
    # 12. GENERATE EMBEDDINGS
    # ---------------------------------------------------------

    print(
        f"Generating embeddings for "
        f"{len(all_legal_chunks)} documents..."
    )

    embeddings = embedding_manager.generate_embeddings(
        all_legal_chunks
    )

    print(
        f"Generated embeddings with shape: "
        f"{embeddings.shape}"
    )

    # ---------------------------------------------------------
    # 13. STORE IN CHROMADB
    # ---------------------------------------------------------
    #
    # IMPORTANT:
    # The exact same document_ids used by BM25
    # are passed to ChromaDB.
    # ---------------------------------------------------------

    vectorstore.add_documents(
        documents=all_legal_chunks,
        embeddings=embeddings,
        ids=document_ids,
    )

    # ---------------------------------------------------------
    # 14. RETURN COMPONENTS
    # ---------------------------------------------------------

    return vectorstore, embedding_manager