"""Import quality labels and failure causes into the database (table run_labels).

Usage (project root):
    python scripts/import_labels.py

Safe to run again: labels are replaced, never duplicated. Only the new
run_labels table is written; no other table is touched.

Labels written
  Document runs (data/document_data.csv):
      answer_correct      Yes / No   (judged against the known answer)
      source_correct      Yes / No
      retrieval_hit       Yes / No   (did a retrieved chunk contain the answer?)
      rag_outcome         Correct / Retrieval miss / Generation miss /
                          Wrong answer (no answer existed)
      question_type, doc_type
  Failed runs (data/research_failures.csv, data/document_failures.csv):
      failure_type        Rate Limit / Timeout / ...
      failure_stage
"""

import csv
import json
from datetime import datetime, timezone

from agentops.config import DATA_DIR
from agentops.datagen.document_judge import contains, normalise, with_number_words
from agentops.db import ExecutionEvent, RunLabel, SessionLocal, init_db


def read_csv(name):
    path = DATA_DIR / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def retrieved_texts(session, trace_id):
    event = (session.query(ExecutionEvent)
             .filter(ExecutionEvent.trace_id == trace_id,
                     ExecutionEvent.event_name == "document_search").first())
    if not event or not event.output_data:
        return None
    try:
        data = json.loads(event.output_data) if isinstance(event.output_data, str) else event.output_data
        return [c.get("text", "") for c in data.get("chunks", [])]
    except Exception:
        return None


def chunks_have_answer(expected, texts):
    if texts is None:
        return None
    joined = normalise(" ".join(texts))
    joined_n = with_number_words(joined)
    return all(contains(joined, normalise(e)) or contains(joined_n, with_number_words(normalise(e)))
               for e in expected)


def main():
    init_db()
    session = SessionLocal()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    labels = {}   # (trace_id, name) -> (value, source)

    # ---- document runs
    trace = {r["Run No"]: r["Trace ID"] for r in read_csv("document_trace_map.csv")}
    for row in read_csv("document_data.csv"):
        trace_id = trace.get(row["Run No"])
        if not trace_id:
            continue
        expected = [e for e in row["Expected Answer"].split("|") if e]
        correct = row["Answer Correct"] == "Yes"

        if row["Question Type"] == "Unanswerable":
            hit, outcome = "n/a", ("Correct" if correct else "Wrong answer (no answer existed)")
        else:
            found = chunks_have_answer(expected, retrieved_texts(session, trace_id))
            hit = {True: "Yes", False: "No", None: "Unknown"}[found]
            if correct:
                outcome = "Correct"
            elif found is False:
                outcome = "Retrieval miss"
            else:
                outcome = "Generation miss"

        for name, value in [
            ("answer_correct", row["Answer Correct"]),
            ("source_correct", row["Source Correct"]),
            ("retrieval_hit", hit),
            ("rag_outcome", outcome),
            ("question_type", row["Question Type"]),
            ("doc_type", row["Doc Type"]),
        ]:
            labels[(trace_id, name)] = (value, "document_data.csv")

    # ---- failed runs (research + document)
    for file_name in ("research_failures.csv", "document_failures.csv"):
        for row in read_csv(file_name):
            trace_id = row.get("Trace ID")
            if not trace_id:
                continue
            labels[(trace_id, "failure_type")] = (row["Failure Type"], file_name)
            labels[(trace_id, "failure_stage")] = (row["Failure Stage"], file_name)

    for (trace_id, name), (value, source) in labels.items():
        session.merge(RunLabel(trace_id=trace_id, label_name=name, label_value=value,
                               source=source, created_at=now))
    session.commit()

    names = {}
    for (_, name) in labels:
        names[name] = names.get(name, 0) + 1
    print(f"Labels written: {len(labels)}")
    for name, count in sorted(names.items()):
        print(f"  {name:<16} {count}")


if __name__ == "__main__":
    main()