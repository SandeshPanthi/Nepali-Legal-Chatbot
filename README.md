# Legal RAG Pipeline

A Retrieval-Augmented-Generation pipeline over three Nepali legal documents
(Civil Code, Penal Code, Constitution), converted from the original
`pdf_loader.ipynb` notebook into a proper Python project — no logic changed,
only reorganized into modules.

## Folder structure

```
legal_rag_pipeline/
├── main.py                  # Entry point: builds the index, runs a query, prints the result
├── config.py                # Central paths & settings (models, thresholds, file paths)
├── requirements.txt
├── .env.example              # Copy to .env and add your GROQ_API_KEY
├── data/
│   ├── acts/                 # Put civil_code.pdf, penal_code.pdf, constitution.pdf here
│   └── vector_store/         # ChromaDB persistent storage (auto-created)
└── src/
    ├── __init__.py
    ├── document_loader.py    # PDF loading + metadata cleanup (per document)
    ├── parsers.py             # Rule-based text -> chunk parsers (per document)
    ├── embeddings.py          # EmbeddingManager (SentenceTransformer)
    ├── vector_store.py        # VectorStore (ChromaDB)
    ├── retriever.py           # RAGRetriever (vector similarity search)
    ├── rag_pipeline.py        # rag_llm() + Groq LLM setup
    └── ingest.py               # Orchestrates load -> parse -> embed -> store
```

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the three source PDFs in `data/acts/`:
   - `civil_code.pdf`
   - `penal_code.pdf`
   - `constitution.pdf`
3. Copy `.env.example` to `.env` and set your `GROQ_API_KEY`.

## Usage

```bash
python main.py "Punishment for robbery"
```

or run without arguments to be prompted for a query:

```bash
python main.py
```

Each run builds the vector index from the PDFs (loads, parses, embeds, and
stores the documents — replacing any existing collection) and then answers
the given query, printing:

- `Answer` — the LLM's grounded answer
- `Sources` — the legal documents/sections used
- `Confidence` — the top similarity score
- `Context Preview` — first 300 characters of the retrieved context
