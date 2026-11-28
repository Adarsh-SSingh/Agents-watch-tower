import os
import re
import uuid
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from agentops.agents.document.embeddings import embeddings


VECTOR_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "agentops_documents"


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=VECTOR_DB_PATH,
)


_DOT_LEADERS = re.compile(r"\.{4,}|(?:\. ){4,}")
_ENDS_WITH_PAGE_NO = re.compile(r"\S\s+\d{1,4}\s*$")


def looks_like_table_of_contents(text):
    """True for chunks that only LIST section names and page numbers.

    Such chunks mention every heading, so they rank above the real text
    ("What does GOVERN 1.6 require?" matched the contents page, not the
    GOVERN 1.6 section). Rule: many short lines that end in a page number,
    or dotted leaders, or a "table of contents" title with some of them.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 4:
        return False
    page_lines = sum(1 for ln in lines if _ENDS_WITH_PAGE_NO.search(ln))
    dotted = sum(1 for ln in lines if _DOT_LEADERS.search(ln))
    share = page_lines / len(lines)
    if dotted / len(lines) >= 0.3:
        return True
    if "table of contents" in text.lower() and share >= 0.3:
        return True
    return share >= 0.6


def store_documents(chunks, document_id=None):

    if not chunks:
        raise ValueError("No document chunks provided.")

    document_id = document_id or str(uuid.uuid4())

    # Attach a unique document ID to every chunk
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id

    vector_store.add_documents(chunks)

    return document_id


def search_documents(query, document_id=None, k=4):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    search_filter = None

    if document_id:
        search_filter = {
            "document_id": document_id
        }

    results = vector_store.similarity_search(
        query,
        k=k,
        filter=search_filter
    )

    return results


class ScoredChromaRetriever(BaseRetriever):
    """Same top-k similarity search as `vector_store.as_retriever()`.

    The only difference: every returned chunk also carries its Chroma score
    in `metadata["retrieval_score"]`, so the monitoring callback can log it.
    The chunks and their order are unchanged, and no extra API call is made.

    Score meaning: Chroma returns a DISTANCE (a smaller number means the
    chunk is closer to the question) with its default settings.
    """

    k: int = 4
    document_id: Optional[str] = None

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:

        search_filter = (
            {"document_id": self.document_id} if self.document_id else None
        )

        # Ask for extra candidates, drop table-of-contents chunks, keep the
        # best k. If too few real chunks remain, fall back to the plain top k.
        results = vector_store.similarity_search_with_score(
            query,
            k=self.k * 5,
            filter=search_filter,
        )

        if os.environ.get("AGENTOPS_NO_FIXES") == "1":
            # Test switch: retrieval as it was BEFORE the fixes (plain top k,
            # no table-of-contents filter, no CSV dataset-summary chunks).
            plain = [(d, s) for d, s in results if not d.metadata.get("summary")]
            chosen = plain[: self.k]
        else:
            kept = [(d, s) for d, s in results
                    if not looks_like_table_of_contents(d.page_content)]
            chosen = kept[: self.k] if len(kept) >= self.k else results[: self.k]

        documents = []

        for document, score in chosen:
            document.metadata["retrieval_score"] = round(float(score), 6)
            documents.append(document)

        return documents


def get_retriever(document_id=None, k=4):
    """LangChain retriever, so the callback can log the retrieval step."""

    return ScoredChromaRetriever(k=k, document_id=document_id)


def delete_document(document_id):

    vector_store.delete(
        where={
            "document_id": document_id
        }
    )