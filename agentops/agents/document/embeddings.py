import os

from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings


load_dotenv()


EMBEDDING_MODEL = "nvidia/nemotron-3-embed-1b"


embeddings = NVIDIAEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=os.getenv("NVIDIA_API_KEY")
)


def generate_embeddings(texts):

    if not texts:
        raise ValueError("No text provided for embedding.")

    vectors = embeddings.embed_documents(texts)

    return vectors


def generate_query_embedding(query):

    if not query.strip():
        raise ValueError("Query cannot be empty.")

    vector = embeddings.embed_query(query)

    return vector