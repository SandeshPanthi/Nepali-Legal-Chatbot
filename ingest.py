"""
First time / documents haven't been indexed: uv run python ingest.py
If legal texts have been replaced/updated: uv run python ingest.py --rebuild
"""

import argparse

from src.ingest import build_index


def main():
    parser = argparse.ArgumentParser(
        description="Build the legal document vector index."
    )

    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete the existing vector store and re-embed all legal documents.",
    )

    args = parser.parse_args()

    build_index(force_rebuild=args.rebuild)


if __name__ == "__main__":
    main()