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

## Example Query

```text
Let's say a political party has a two-thirds majority in the House of Representatives but only around 10% of the seats in the National Assembly.

Can it pass ordinary bills, constitutional amendment bills, and other legislation?

If it cannot pass the bill, how will the government function if it wants to introduce new laws or amend existing legislation?

Explain all constitutional provisions, legal procedures, possible mechanisms, limitations, and exceptions. Provide a complete legal analysis with references to the relevant constitutional provisions and legal sources.
```

Answer: According to the Constitution of Nepal, Part-9 - Federal Legislative Procedures, Section 111, a Bill passed by one House of the Federal Parliament shall be transmitted to the other House as soon as possible. 

In this scenario, if the party has 2/3 seats in the lower house (House of Representatives) but only 10% of seats in the upper house (National Assembly), they can still pass a Bill in the lower house with a majority vote as per Section 99 of the Constitution of Nepal, Part-8 - Federal Legislature, which states that any motions submitted for decision in either House of the Federal Parliament shall be decided by a majority vote of the members present and voting.

However, when the Bill is transmitted to the upper house, it may face difficulties in passing due to the lack of majority seats. According to Section 111 (5) of the Constitution of Nepal, Part-9 - Federal Legislative Procedures, if the National Assembly does not return a Bill, except for a Money Bill, within two months from the date of receipt, the House of Representatives may, by a resolution passed by a majority of the total number of its existing members, present the Bill to the President for assent.

If the upper house rejects or passes the Bill with amendments, the lower house can still pass the Bill again with a majority vote, and present it to the President for assent, as per Section 111 (7) of the Constitution of Nepal, Part-9 - Federal Legislative Procedures.

In case of a disagreement between the two houses, the Bill can be referred to a joint sitting of both houses, and if the joint sitting passes the Bill, it can be presented to the President for assent, as per Section 111 (9) of the Constitution of Nepal, Part-9 - Federal Legislative Procedures.

If the government (PM) wants to introduce new amendments in laws, acts, or bills, they can do so by introducing a new Bill in the lower house, which can then be transmitted to the upper house for consideration.

In terms of the workings of the government, if the party is unable to pass the Bill, they can still continue to govern and make decisions through other means, such as executive orders or regulations, as long as they are within the scope of their constitutional powers.

However, if the government is unable to pass important legislation, it may face difficulties in implementing its policies and programs, which could lead to a constitutional crisis.

Sources:
- The Constitution of Nepal, Part-9 - Federal Legislative Procedures, Section 111
- The Constitution of Nepal, Part-8 - Federal Legislature, Section 99
- The Constitution of Nepal, Part-9 - Federal Legislative Procedures, Section 111 (5), (7), and (9)

It is worth noting that, according to the Constitution of Nepal, Part-9 - Federal Legislative Procedures, Section 111 (10), if the House of Representatives is dissolved or its term expires when any Bill introduced in the House of Representatives is under its consideration or when any Bill passed by the House of Representatives is under consideration in the National Assembly, such Bill shall lapse. 

Therefore, it is crucial for the government to manage its legislative agenda carefully and work with the upper house to pass important legislation, or to use other means to achieve its policy objectives.

--------------------------------------------------------------------------------------

Confidence: 0.07056301832199097

--------------------------------------------------------------------------------------

Sources: [{'source': 'THE CONSTITUTION OF NEPAL', 'page': 'unknown', 'score': 0.07056301832199097, 'preview': '199. Procedures for p assage of Bills: (1) A Bill passe d by the State Assembly shall be presented to the Chief of State......'}, {'source': 'THE CONSTITUTION OF NEPAL', 'page': 'unknown', 'score': 0.06221210956573486, 'preview': '111. Procedures for passage of Bills: (1) A Bill passed by one House of the Federal Parliament shall be transmitted to t......'}, {'source': 'THE CONSTITUTION OF NEPAL', 'page': 'unknown', 'score': 0.046923935413360596, 'preview': '99. Voting: Except as otherwise provided in this Constitution, any motions submitted for decision in either House o f th......'}]

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
