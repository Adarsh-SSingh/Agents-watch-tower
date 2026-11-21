from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)


LOADER_REGISTRY = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".txt": TextLoader,
    ".csv": CSVLoader,
    ".xlsx": UnstructuredExcelLoader,
    ".pptx": UnstructuredPowerPointLoader,
}


def load_uploaded_document(file_bytes: bytes, filename: str):

    extension = Path(filename).suffix.lower()

    loader_class = LOADER_REGISTRY.get(extension)

    if loader_class is None:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    if not file_bytes:
        raise ValueError("Uploaded file is empty.")

    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=extension,
            delete=False
        ) as temp_file:

            temp_file.write(file_bytes)
            temp_path = temp_file.name

        if extension == ".txt":
            loader = loader_class(
                temp_path,
                encoding="utf-8",
                autodetect_encoding=True
            )

        elif extension in {".xlsx", ".pptx"}:
            loader = loader_class(
                temp_path,
                mode="elements"
            )

        else:
            loader = loader_class(temp_path)

        documents = loader.load()

        if not documents:
            raise ValueError(
                "No readable content found in document."
            )

        for document in documents:
            document.metadata.update({
                "source_file": Path(filename).name,
                "file_type": extension,
            })

        return documents

    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()