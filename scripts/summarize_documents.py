"""Summary of the document run + a review file of the wrong answers.

Usage (project root):
    python scripts/summarize_documents.py

Prints accuracy by question type / document type / format, and for every
wrong answer finds out WHY using the retrieval logs in the database:
  Retrieval miss  - the retrieved chunks did not contain the answer
  Generation miss - the chunks had the answer, the agent still answered wrong
  (Unanswerable)  - the agent gave an answer when the document had none

Writes data/document_wrong_review.txt (question, expected, answer) so the
judge can be checked by reading it.
"""

import csv
import json
from collections import Counter, defaultdict

from agentops.config import DATA_DIR
from agentops.datagen.document_judge import contains, normalise, with_number_words
from agentops.db import AgentRun, ExecutionEvent, SessionLocal

TABLE = DATA_DIR / "document_data.csv"
MAP = DATA_DIR / "document_trace_map.csv"
REVIEW = DATA_DIR / "document_wrong_review.txt"


def chunk_texts(session, trace_id):
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


def retrieval_hit(expected, texts):
    if texts is None:
        return None
    joined = normalise(" ".join(texts))
    joined_n = with_number_words(joined)
    return all(contains(joined, normalise(e)) or contains(joined_n, with_number_words(normalise(e)))
               for e in expected)


def table(title, rows, key):
    groups = defaultdict(lambda: [0, 0])
    for row in rows:
        groups[row[key]][0] += 1
        groups[row[key]][1] += row["Answer Correct"] == "Yes"
    print(f"\n{title}")
    for name, (total, ok) in sorted(groups.items(), key=lambda x: x[1][1] / x[1][0]):
        print(f"  {name:<20} {ok:>3}/{total:<3} {ok / total * 100:5.1f}%")


def main():
    rows = list(csv.DictReader(open(TABLE, encoding="utf-8")))
    trace = {r["Run No"]: r["Trace ID"] for r in csv.DictReader(open(MAP, encoding="utf-8"))}
    ok = sum(r["Answer Correct"] == "Yes" for r in rows)
    print(f"Rows: {len(rows)} | correct: {ok} ({ok / len(rows) * 100:.1f}%)")

    table("By question type", rows, "Question Type")
    table("By document type", rows, "Doc Type")
    table("By format", rows, "Format")
    table("By question style", rows, "Question Style")

    session = SessionLocal()
    reasons = Counter()
    lines = []
    for row in rows:
        if row["Answer Correct"] == "Yes":
            continue
        run = session.query(AgentRun).filter(AgentRun.trace_id == trace[row["Run No"]]).first()
        answer = (run.final_output if run else "?") or ""
        expected = [e for e in row["Expected Answer"].split("|") if e]

        if row["Question Type"] == "Unanswerable":
            reason = "Gave an answer (or judge missed refusal wording)"
        else:
            hit = retrieval_hit(expected, chunk_texts(session, trace[row["Run No"]]))
            reason = {True: "Generation miss (chunks had the answer)",
                      False: "Retrieval miss (chunks lacked the answer)",
                      None: "Unknown (no chunk log)"}[hit]
        reasons[reason] += 1
        lines.append(f"#{row['Run No']} [{row['Question Type']}] {reason}\n"
                     f"Q: {row['Question']}\nExpected: {row['Expected Answer']}\n"
                     f"Answer: {answer[:600]}\n{'-' * 70}")

    print(f"\nWrong answers: {len(lines)}")
    for reason, count in reasons.most_common():
        print(f"  {count:>3}  {reason}")
    REVIEW.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReview file: {REVIEW}")


if __name__ == "__main__":
    main()