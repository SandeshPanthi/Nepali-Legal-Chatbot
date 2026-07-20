"""
PDF document loading utilities.

Each function loads a single legal document (Civil Code, Penal Code,
Constitution) via PyPDFLoader and normalizes its metadata, exactly as
was done in the original notebook.
"""

import os
from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_civil_code(pdf_path: Path):
    """Load the Civil Code PDF and normalize its metadata."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print(len(documents))

    for doc in documents:
        doc.metadata.pop("producer", None)
        doc.metadata.pop("creator", None)
        doc.metadata['title'] = "The National Civil (Code) Act, 2017 (2074)"
        doc.metadata['source_file'] = "Civil Code"
        doc.metadata['file_type'] = 'pdf'

    # Drop the first few front-matter pages (cover/title pages)
    documents = documents[3:]

    return documents


def load_penal_code(pdf_path: Path):
    """Load the Penal Code PDF and normalize its metadata."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(len(documents))

    for doc in documents:
        doc.metadata.pop("producer", None)
        doc.metadata.pop("creator", None)
        doc.metadata.pop("author")
        doc.metadata['title'] = "The National Penal (Code) Act, 2017"
        doc.metadata['source_file'] = "Penal Code"
        doc.metadata['file_type'] = 'pdf'

    return documents


def load_constitution(pdf_path: Path):
    """Load the Constitution PDF and normalize its metadata."""
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(len(documents))

    for doc in documents:
        doc.metadata.pop("producer", None)
        doc.metadata.pop("creator", None)
        doc.metadata['title'] = "THE CONSTITUTION OF NEPAL"
        doc.metadata['source_file'] = "Constitution"
        doc.metadata['file_type'] = 'pdf'

    # Drop the first few front-matter pages (cover/title/index pages)
    documents = documents[5:]

    return documents
