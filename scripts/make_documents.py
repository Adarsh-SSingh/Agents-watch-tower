"""Create the realistic synthetic documents and the question set.

Usage (from project root):
    python scripts/make_documents.py

Needs:  uv pip install python-docx reportlab pypdf

Writes:
  data/documents/*.pdf|docx|txt|csv   the documents (14 types)
  data/documents/qa_set.csv           every question with its known answer
  data/documents/doc_index.csv        one row per document (type, format, chunks)

After writing, every document is read back and every expected answer is
checked against the real file text. Unanswerable questions are checked to make
sure the document really does not contain the answer.
"""

import argparse
import csv
import re
from pathlib import Path

from agentops.config import DATA_DIR
from agentops.datagen.document_corpus import build_corpus

DOC_DIR = Path(DATA_DIR) / "documents"
QA_FILE = DOC_DIR / "qa_set.csv"
INDEX_FILE = DOC_DIR / "doc_index.csv"

QA_FIELDS = [
    "Question No", "Doc File", "Doc Type", "Format", "Question Type",
    "Question Style", "Difficulty", "Question", "Expected Answer", "Absent Words",
]
INDEX_FIELDS = ["Doc File", "Doc Type", "Format", "Title", "Characters", "Chunks", "Questions"]


# ------------------------------------------------------------------ writers

def block_text(blocks: list) -> str:
    parts = []
    for block in blocks:
        kind = block[0]
        if kind in ("h", "p"):
            parts.append(block[1])
        else:
            header, rows = block[1], block[2]
            parts.append("\n".join([", ".join(map(str, header))] +
                                   [", ".join(map(str, r)) for r in rows]))
    return "\n\n".join(parts)


def write_txt(path: Path, title: str, blocks: list) -> None:
    path.write_text(title + "\n\n" + block_text(blocks), encoding="utf-8")


def write_csv(path: Path, title: str, blocks: list) -> None:
    _, header, rows = blocks[0]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def write_docx(path: Path, title: str, blocks: list) -> None:
    from docx import Document

    document = Document()
    document.add_heading(title, level=1)
    for block in blocks:
        if block[0] == "h":
            document.add_heading(block[1], level=2)
        elif block[0] == "p":
            document.add_paragraph(block[1])
        else:
            header, rows = block[1], block[2]
            table = document.add_table(rows=1, cols=len(header))
            table.style = "Table Grid"
            for i, name in enumerate(header):
                table.rows[0].cells[i].text = str(name)
            for row in rows:
                cells = table.add_row().cells
                for i, value in enumerate(row):
                    cells[i].text = str(value)
    document.save(path)


def write_pdf(path: Path, title: str, blocks: list) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    story = [Paragraph(escape(title), styles["Heading1"]), Spacer(1, 10)]
    for block in blocks:
        if block[0] == "h":
            story.append(Paragraph(escape(block[1]), styles["Heading2"]))
        elif block[0] == "p":
            story.append(Paragraph(escape(block[1]), styles["BodyText"]))
            story.append(Spacer(1, 6))
        else:
            data = [list(map(str, block[1]))] + [list(map(str, r)) for r in block[2]]
            table = Table(data)
            table.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]))
            story += [table, Spacer(1, 8)]
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


WRITERS = {"txt": write_txt, "csv": write_csv, "docx": write_docx, "pdf": write_pdf}


# --------------------------------------------------------------- read back

def read_back(path: Path) -> str:
    """Read the written file the same way the project loader would."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".csv"):
        return path.read_text(encoding="utf-8")
    if suffix == ".docx":
        from docx import Document

        document = Document(path)
        text = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                text.append(", ".join(cell.text for cell in row.cells))
        return "\n".join(text)
    from pypdf import PdfReader

    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def normalise(text: str) -> str:
    text = text.lower().replace(",", "")
    return re.sub(r"\s+", " ", text)


def count_chunks(text: str) -> int:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    return len(splitter.split_text(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-type", type=int, default=3,
                        help="Documents per type (14 types). Default 3 = about 45 documents.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_type < 1:
        parser.error("--per-type must be 1 or more")

    DOC_DIR.mkdir(parents=True, exist_ok=True)
    for old in DOC_DIR.glob("*"):
        if old.suffix.lower() in (".pdf", ".docx", ".txt", ".csv"):
            old.unlink()

    corpus = build_corpus(seed=args.seed, docs_per_type=args.per_type)

    qa_rows, index_rows, problems = [], [], []
    for doc in corpus:
        path = DOC_DIR / doc["file_name"]
        WRITERS[doc["format"]](path, doc["title"], doc["blocks"])

        text = read_back(path)
        flat = normalise(text)
        chunks = count_chunks(text)

        for item in doc["questions"]:
            if item["question_type"] == "Unanswerable":
                for word in item["absent"]:
                    if word.lower() in flat:
                        problems.append(f"{doc['file_name']}: '{word}' IS in the file -> {item['question']}")
            elif item["question_type"] in ("Fact", "Multi", "Table Lookup"):
                for expected in item["expected"]:
                    if normalise(expected) not in flat:
                        problems.append(
                            f"{doc['file_name']}: '{expected}' NOT in the file -> {item['question']}")
            qa_rows.append({
                "Question No": len(qa_rows) + 1,
                "Doc File": doc["file_name"],
                "Doc Type": doc["doc_type"],
                "Format": doc["format"],
                "Question Type": item["question_type"],
                "Question Style": item["style"],
                "Difficulty": item["difficulty"],
                "Question": item["question"],
                "Expected Answer": "|".join(item["expected"]),
                "Absent Words": "|".join(item["absent"]),
            })

        index_rows.append({
            "Doc File": doc["file_name"], "Doc Type": doc["doc_type"], "Format": doc["format"],
            "Title": doc["title"], "Characters": len(text), "Chunks": chunks,
            "Questions": len(doc["questions"]),
        })

    with open(QA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(qa_rows)
    with open(INDEX_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=INDEX_FIELDS)
        writer.writeheader()
        writer.writerows(index_rows)

    chunk_counts = [r["Chunks"] for r in index_rows]
    print(f"Documents : {len(index_rows)}")
    print(f"Questions : {len(qa_rows)}")
    print(f"Chunks    : min {min(chunk_counts)}, max {max(chunk_counts)}, total {sum(chunk_counts)}")
    print(f"Folder    : {DOC_DIR}")
    if problems:
        print(f"\nVALIDATION PROBLEMS ({len(problems)}):")
        for line in problems[:40]:
            print("  -", line)
    else:
        print("Validation: every expected answer is in its file; every unanswerable is absent.")


if __name__ == "__main__":
    main()