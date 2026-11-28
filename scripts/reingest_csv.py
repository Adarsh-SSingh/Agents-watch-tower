"""Re-ingest only the CSV files of data/real_documents/ (adds the dataset summary chunk).

Run once after updating loader.py. Other documents are not touched, so no
extra embedding cost. Usage:  python scripts\\reingest_csv.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_documents import REAL_DIR, safe_store_documents          # noqa: E402
from agentops.agents.document.chunker import chunk_documents          # noqa: E402
from agentops.agents.document.loader import load_uploaded_document    # noqa: E402
from agentops.agents.document.vector_store import delete_document     # noqa: E402


def main():
    files = sorted(REAL_DIR.glob("*.csv"))
    if not files:
        raise SystemExit("No CSV files in data/real_documents/")
    for path in files:
        document_id = "doc-" + path.stem
        documents = load_uploaded_document(path.read_bytes(), path.name)
        chunks = chunk_documents(documents)
        delete_document(document_id)
        safe_store_documents(chunks, document_id=document_id)
        print(f"{path.name}: {len(documents)} rows+summary -> {len(chunks)} chunks")


if __name__ == "__main__":
    main()