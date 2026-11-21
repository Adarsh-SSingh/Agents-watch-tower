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

        results = vector_store.similarity_search_with_score(
            query,
            k=self.k,
            filter=search_filter,
        )

        documents = []

        for document, score in results:
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