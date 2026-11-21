"""Run the document question set through the Document Agent.

Usage (from project root, after scripts/make_documents.py):
    python scripts/run_document_bulk.py --limit 3     # smoke test
    python scripts/run_document_bulk.py               # run everything left

Steps it does:
  1. Ingest every document ONCE (load -> chunk -> embed -> Chroma) with a fixed
     document id. Already ingested documents are skipped, so there are no
     duplicate chunks. Use --reingest to delete and ingest again.
  2. Ask every question (shuffled, same order every time) with the whole
     corpus searchable, like a real user (use --scoped to search only the
     question's own document).
  3. Judge each answer against the known answer (no LLM needed).

Writes:
  data/document_data.csv         one readable row per question
  data/document_trace_map.csv    Run No -> Trace ID (join with the database)
  data/document_failures.csv     runs lost to API problems (kept as data)
  data/documents/ingest_manifest.csv   document id, chunks, ingest time
Running again continues from the first question not yet answered.
"""

import argparse
import csv
import hashlib
import random
import time
from datetime import datetime
from pathlib import Path

import psutil

from agentops.agents.document.agent import run_document_agent
from agentops.agents.document.chunker import chunk_documents
from agentops.agents.document.loader import load_uploaded_document
from agentops.agents.document.vector_store import delete_document, store_documents
from agentops.config import DATA_DIR
from agentops.datagen.document_judge import answer_correct, refused, source_correct

DOC_DIR = Path(DATA_DIR) / "documents"
QA_FILE = DOC_DIR / "qa_set.csv"
MANIFEST_FILE = DOC_DIR / "ingest_manifest.csv"
TABLE_FILE = Path(DATA_DIR) / "document_data.csv"
MAP_FILE = Path(DATA_DIR) / "document_trace_map.csv"
FAIL_FILE = Path(DATA_DIR) / "document_failures.csv"

MANIFEST_FIELDS = ["Doc File", "Document ID", "Characters", "Chunks", "Content Hash", "Ingested At"]
MAP_FIELDS = ["Run No", "Trace ID"]
FIELDS = [
    "Run No", "Status", "Answer Correct", "Source Correct", "Answer Refused",
    "Doc Type", "Format", "Question Type", "Question Style", "Difficulty",
    "Question Length", "Word Count", "Doc Chunks", "Doc Characters", "Scope",
    "Retrieved Chunks", "Answer Length",
    "CPU Usage (%)", "RAM Usage (%)", "Free RAM (GB)", "Hour", "Weekday",
    "Run Time (s)", "Retrieval Latency (s)", "Generation Latency (s)",
    "Doc File", "Question", "Expected Answer",
]
FAIL_FIELDS = [
    "Run No", "Failure Type", "Failure Stage", "Error Message", "Trace ID",
    "Doc Type", "Format", "Question Type", "Question Style", "Difficulty",
    "Question Length", "Word Count", "CPU Usage (%)", "RAM Usage (%)",
    "Free RAM (GB)", "Hour", "Weekday", "Run Time (s)", "Doc File", "Question",
]


def nice(value) -> str:
    return str(value).replace("_", " ").title()


def failure_type(error_text: str) -> str:
    text = (error_text or "").lower()
    if "429" in text or "rate limit" in text or "ratelimit" in text:
        return "Rate Limit"
    if "timed out" in text or "timeout" in text:
        return "Timeout"
    if "connection" in text or "502" in text or "503" in text or "504" in text:
        return "Network Error"
    if "401" in text or "unauthorized" in text or "api key" in text or "quota" in text:
        return "Auth Or Quota"
    return "Other Error"


def machine_state() -> dict:
    memory = psutil.virtual_memory()
    now = datetime.now()
    return {
        "CPU Usage (%)": psutil.cpu_percent(interval=0.3),
        "RAM Usage (%)": memory.percent,
        "Free RAM (GB)": round(memory.available / 1024 ** 3, 2),
        "Hour": now.hour,
        "Weekday": now.strftime("%A"),
    }


def read_csv(path: Path) -> list:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def append_csv(path: Path, fields: list, row: dict) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def document_id_for(doc_file: str) -> str:
    return "doc-" + Path(doc_file).stem


def ingest_documents(doc_files: list, reingest: bool) -> dict:
    """Ingest each file once. Returns {doc_file: manifest row}."""
    manifest = {row["Doc File"]: row for row in read_csv(MANIFEST_FILE)}

    for number, doc_file in enumerate(doc_files, start=1):
        path = DOC_DIR / doc_file
        file_bytes = path.read_bytes()
        content_hash = hashlib.sha1(file_bytes).hexdigest()[:12]

        # Skip only if this exact file content was already ingested.
        old = manifest.get(doc_file)
        if old and not reingest and old.get("Content Hash") == content_hash:
            continue

        document_id = document_id_for(doc_file)
        print(f"  ingest [{number}/{len(doc_files)}] {doc_file}")

        documents = load_uploaded_document(file_bytes, doc_file)
        chunks = chunk_documents(documents)
        characters = sum(len(doc.page_content) for doc in documents)

        # Embedding API can fail briefly (502, rate limit): retry, then stop.
        for attempt in range(1, 5):
            try:
                # Never create duplicates: remove old chunks of this id first.
                delete_document(document_id)
                store_documents(chunks, document_id=document_id)
                break
            except Exception as error:
                if attempt == 4:
                    raise
                wait = 10 * attempt
                print(f"    ! {type(error).__name__}: {str(error)[:80]} - retry in {wait}s")
                time.sleep(wait)

        row = {
            "Doc File": doc_file, "Document ID": document_id, "Characters": characters,
            "Chunks": len(chunks), "Content Hash": content_hash, "Ingested At": datetime.now().isoformat(timespec="seconds"),
        }
        manifest[doc_file] = row

        # Rewrite the manifest after every document (safe if stopped midway).
        with open(MANIFEST_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=MANIFEST_FIELDS)
            writer.writeheader()
            writer.writerows(manifest.values())

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max questions to run in this session (must be >= 1).")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="Seconds to wait between runs (must be >= 0).")
    parser.add_argument("--scoped", action="store_true",
                        help="Search only the question's own document (default: all documents).")
    parser.add_argument("--reingest", action="store_true",
                        help="Delete and ingest all documents again.")
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    if args.pause < 0:
        parser.error("--pause cannot be negative")
    if args.top_k < 1:
        parser.error("--top-k must be 1 or more")

    questions = read_csv(QA_FILE)
    if not questions:
        raise SystemExit("data/documents/qa_set.csv not found. Run scripts/make_documents.py first.")

    # Same shuffled order every time -> Run No always means the same question.
    order = list(range(len(questions)))
    random.Random(42).shuffle(order)
    items = [questions[i] for i in order]

    done = {int(row["Run No"]) - 1 for row in read_csv(TABLE_FILE)}
    pending = [(i, item) for i, item in enumerate(items) if i not in done]
    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Questions total: {len(items)} | already done: {len(done)} | running now: {len(pending)}")
    if not pending:
        return

    # 1. Ingest (only what is not ingested yet)
    doc_files = sorted({item["Doc File"] for item in questions})
    print("Checking documents in the vector store...")
    try:
        manifest = ingest_documents(doc_files, args.reingest)
    except Exception as error:
        raise SystemExit(f"\nIngest failed: {type(error).__name__}: {error}\n"
                         "Nothing is lost. Fix the problem and run again; finished documents are skipped.")

    started = time.perf_counter()
    failures = 0

    try:
        for position, (index, item) in enumerate(pending, start=1):
            question = item["Question"]
            doc_file = item["Doc File"]
            print(f"\n[{position}/{len(pending)}] #{index} ({item['Question Type']}) {question[:90]}")

            state = machine_state()
            run_started = time.perf_counter()
            document_id = manifest[doc_file]["Document ID"] if args.scoped else None

            error_text = ""
            result = {}
            try:
                result = run_document_agent(question, document_id=document_id, top_k=args.top_k)
                trace_id = result.get("trace_id") or ""
                status = result.get("status") or "unknown"
            except Exception as error:
                trace_id, status = "", "script_error"
                error_text = f"{type(error).__name__}: {error}"

            run_time = round(time.perf_counter() - run_started, 3)

            # API / infrastructure problems are logged, not saved as data rows.
            if status in ("error", "script_error", "unknown") or not trace_id:
                error_text = error_text or str(result.get("answer") or status)
                kind = failure_type(error_text)
                print(f"  ! {kind}: {error_text[:200]}")
                append_csv(FAIL_FILE, FAIL_FIELDS, {
                    "Run No": index + 1,
                    "Failure Type": kind,
                    "Failure Stage": "Run" if status == "script_error" else "Agent",
                    "Error Message": error_text[:300].replace("\n", " "),
                    "Trace ID": trace_id,
                    "Doc Type": nice(item["Doc Type"]), "Format": item["Format"].upper(),
                    "Question Type": item["Question Type"], "Question Style": item["Question Style"],
                    "Difficulty": nice(item["Difficulty"]),
                    "Question Length": len(question), "Word Count": len(question.split()),
                    **state, "Run Time (s)": run_time,
                    "Doc File": doc_file, "Question": question,
                })
                failures += 1
                print(f"  ! Failed ({failures} in a row). Not saved to the main table; will retry later.")
                if failures >= 3:
                    print("\nStopping: 3 failures in a row (API limit or network problem). "
                          "Wait 15-30 minutes and run again.")
                    break
                time.sleep(30 * failures)
                continue

            failures = 0

            answer = result.get("answer") or ""
            expected = [e for e in item["Expected Answer"].split("|") if e]
            correct = answer_correct(item["Question Type"], expected, answer)
            src_ok = source_correct(doc_file, result.get("sources"))

            append_csv(MAP_FILE, MAP_FIELDS, {"Run No": index + 1, "Trace ID": trace_id})
            append_csv(TABLE_FILE, FIELDS, {
                "Run No": index + 1,
                "Status": nice(status),
                "Answer Correct": "Yes" if correct else "No",
                "Source Correct": "Yes" if src_ok else "No",
                "Answer Refused": "Yes" if refused(answer) else "No",
                "Doc Type": nice(item["Doc Type"]),
                "Format": item["Format"].upper(),
                "Question Type": item["Question Type"],
                "Question Style": item["Question Style"],
                "Difficulty": nice(item["Difficulty"]),
                "Question Length": len(question),
                "Word Count": len(question.split()),
                "Doc Chunks": manifest[doc_file]["Chunks"],
                "Doc Characters": manifest[doc_file]["Characters"],
                "Scope": "Single Document" if args.scoped else "All Documents",
                "Retrieved Chunks": result.get("retrieved_chunks", 0),
                "Answer Length": len(answer),
                **state,
                "Run Time (s)": run_time,
                "Retrieval Latency (s)": result.get("retrieval_latency", 0),
                "Generation Latency (s)": result.get("generation_latency", 0),
                "Doc File": doc_file,
                "Question": question,
                "Expected Answer": item["Expected Answer"],
            })
            print(f"  -> correct={'Yes' if correct else 'No'} source={'Yes' if src_ok else 'No'} "
                  f"in {round(run_time, 1)} s")

            if args.pause:
                time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\nStopped by user. Run again to continue.")

    print(f"\nSession time: {round((time.perf_counter() - started) / 60, 1)} min")


if __name__ == "__main__":
    main()