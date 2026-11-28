"""Phase 2: ask questions about REAL documents (papers, reports, books, datasets).

Usage (project root):
    python scripts/run_real_documents.py --limit 3     # smoke test
    python scripts/run_real_documents.py               # everything left

Put the real files in  data/real_documents/  (exact names as in
agentops/datagen/real_questions.py). The question list lives in code, NOT in
that folder, so the agent never sees the answers.

It reuses run_document_bulk.py (ingest once, ask, judge) and writes
separate files, so the synthetic results stay untouched:
  data/real_qa_set.csv  data/real_document_data.csv
  data/real_document_trace_map.csv  data/real_document_failures.csv
  data/real_ingest_manifest.csv
Each question is asked in 3 styles (Plain / Polite / Hinglish).
Then run:  python scripts/import_labels.py
"""

import csv
import sys
from pathlib import Path

import run_document_bulk as base
from agentops.config import DATA_DIR
from agentops.datagen import document_judge
from agentops.datagen.document_corpus import STYLES, style_question
from agentops.datagen.real_hinglish_questions import HINGLISH
from agentops.datagen.real_hinglish_2 import HINGLISH_2
from agentops.datagen.real_questions import DIFFICULTY, QUESTIONS
from agentops.datagen.real_questions_2 import QUESTIONS_2

HINGLISH_ALL = {**HINGLISH, **HINGLISH_2}
ACTIVE = list(QUESTIONS)   # --tag heldout switches to the 250 held-out questions
ACTIVE_STYLES = list(STYLES)

REAL_DIR = Path(DATA_DIR) / "real_documents"
QA_FIELDS = ["Question No", "Doc File", "Doc Type", "Format", "Question Type",
             "Question Style", "Difficulty", "Question", "Expected Answer", "Location"]

_original_judge = document_judge.answer_correct


def real_answer_correct(question_type, expected, answer):
    """Same judge, plus alternatives: a piece 'a~b' passes if a OR b is found."""
    if question_type in ("Unanswerable", "Yes No") or not expected:
        return _original_judge(question_type, expected, answer)
    for piece in expected:
        if not any(_original_judge(question_type, [alt], answer) for alt in piece.split("~")):
            return False
    return True


SKIPPED_FILE = Path(DATA_DIR) / "real_skipped_chunks.csv"


def append_skipped(document_id: str, chunk, error: Exception) -> None:
    is_new = not SKIPPED_FILE.exists()
    with open(SKIPPED_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if is_new:
            writer.writerow(["Document ID", "Page", "Error", "Chunk Start"])
        writer.writerow([document_id, chunk.metadata.get("page", ""),
                         str(error)[:120].replace("\n", " "),
                         chunk.page_content[:150].replace("\n", " ")])


def safe_store_documents(chunks, document_id=None, batch_size=16):
    """Store chunks in small batches. The embedding API screens content (422);
    one rejected chunk must not lose the whole document: retry that batch one
    chunk at a time, skip only the rejected chunk and log it."""
    from agentops.agents.document.vector_store import vector_store

    if not chunks:
        raise ValueError("No document chunks provided.")
    for chunk in chunks:
        chunk.metadata["document_id"] = document_id
    skipped = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        try:
            vector_store.add_documents(batch)
        except Exception as error:
            if "422" not in str(error) and "content safety" not in str(error).lower():
                raise  # rate limit / network: let the normal retry handle it
            for chunk in batch:
                try:
                    vector_store.add_documents([chunk])
                except Exception as single_error:
                    if "422" not in str(single_error) and "content safety" not in str(single_error).lower():
                        raise
                    append_skipped(document_id, chunk, single_error)
                    skipped += 1
    if skipped:
        print(f"    ! {skipped} chunk(s) rejected by content screening (see data/real_skipped_chunks.csv)")
    return document_id


def build_qa_file() -> list:
    rows = []
    for doc_file, doc_type, q_type, question, expected, location in ACTIVE:
        suffix = Path(doc_file).suffix.lstrip(".").upper()
        for style in ACTIVE_STYLES:
            rows.append({
                "Question No": len(rows) + 1, "Doc File": doc_file,
                "Doc Type": "Real " + doc_type, "Format": suffix,
                "Question Type": q_type, "Question Style": style,
                "Difficulty": DIFFICULTY[q_type],
                "Question": HINGLISH_ALL[question] if style == "Hinglish" else style_question(question, style),
                "Expected Answer": expected, "Location": location,
            })
    with open(base.QA_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    global ACTIVE, ACTIVE_STYLES
    if any(a.startswith("heldout") for a in sys.argv):
        ACTIVE = list(QUESTIONS_2)   # all 3 styles: plain, polite, Hinglish
    missing = sorted({q[0] for q in ACTIVE if not (REAL_DIR / q[0]).exists()})
    if missing:
        print("These files are missing in data/real_documents/ :")
        for name in missing:
            print("  -", name)
        raise SystemExit("\nAdd them (exact names) and run again.")

    # --tag after  -> results go to real_after_*.csv, so the first ("before")
    # results are never touched. No tag = the original real_*.csv files.
    prefix = "real_"
    if "--tag" in sys.argv:
        at = sys.argv.index("--tag")
        tag = sys.argv[at + 1]
        del sys.argv[at:at + 2]
        prefix = f"real_{tag}_"

    data = Path(DATA_DIR)
    base.DOC_DIR = REAL_DIR
    base.QA_FILE = data / f"{prefix}qa_set.csv"
    base.MANIFEST_FILE = data / "real_ingest_manifest.csv"
    base.TABLE_FILE = data / f"{prefix}document_data.csv"
    base.MAP_FILE = data / f"{prefix}document_trace_map.csv"
    base.FAIL_FILE = data / f"{prefix}document_failures.csv"
    base.answer_correct = real_answer_correct
    base.store_documents = safe_store_documents

    rows = build_qa_file()
    print(f"Real documents: {len({r['Doc File'] for r in rows})} | questions x styles: {len(rows)}")
    base.main()


if __name__ == "__main__":
    main()