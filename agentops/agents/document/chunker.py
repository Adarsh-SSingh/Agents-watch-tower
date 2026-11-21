from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_documents(
    documents,
    chunk_size=1000,
    chunk_overlap=150
):

    if not documents:
        raise ValueError("No documents provided for chunking.")

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "Chunk overlap must be non-negative and smaller than chunk size."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True
    )

    chunks = splitter.split_documents(documents)

    return chunks