from pathlib import Path
from tempfile import NamedTemporaryFile

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
)


def build_csv_summary(path, filename):
    """One extra "dataset summary" Document for a CSV file.

    A CSV is stored one row per chunk and only the top 4 chunks reach the
    LLM, so "how many passengers survived" or "average age" can never be
    answered from 4 rows. This summary carries the whole-file numbers
    (computed with pandas, not by the LLM) and is retrieved like any chunk.
    Returns None if the file cannot be summarised.
    """
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:
        return None

    lines = [f"DATASET SUMMARY of {filename}: {len(df)} rows, {len(df.columns)} columns.",
             "Columns: " + ", ".join(str(c) for c in df.columns) + "."]

    numeric = df.select_dtypes("number")
    if not numeric.empty:
        lines.append("Numeric columns (count of non-missing values, mean, min, max; missing values ignored):")
        for col in numeric.columns:
            s = numeric[col].dropna()
            if len(s):
                lines.append(f"- {col}: count {len(s)}, mean {s.mean():.4g}, min {s.min():.4g}, max {s.max():.4g}")

    small = [c for c in df.columns if 1 < df[c].nunique() <= 8]
    for col in small:
        counts = df[col].value_counts(dropna=False).to_dict()
        lines.append(f"Count of rows by {col}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    binary = [c for c in numeric.columns if set(numeric[c].dropna().unique()) <= {0, 1}]
    for flag in binary:
        for col in small:
            if col == flag:
                continue
            g = df.groupby(col)[flag].agg(["mean", "sum", "count"])
            lines.append(f"{flag} by {col} (share of 1s, number of 1s, rows): " + "; ".join(
                f"{k}: {r['mean'] * 100:.1f}%, {int(r['sum'])}, {int(r['count'])}" for k, r in g.iterrows()))

    return Document(page_content="\n".join(lines),
                    metadata={"source": str(filename), "row": -1, "summary": True})


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

        if extension == ".csv":
            summary = build_csv_summary(temp_path, Path(filename).name)
            if summary is not None:
                documents.append(summary)

        for document in documents:
            document.metadata.update({
                "source_file": Path(filename).name,
                "file_type": extension,
            })

        return documents

    finally:
        if temp_path and Path(temp_path).exists():
            Path(temp_path).unlink()