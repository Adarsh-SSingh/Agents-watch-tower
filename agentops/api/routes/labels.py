"""Quality labels and failures.

Run status only says "the agent finished without crashing". These endpoints
show what status cannot: wrong answers (silent failures), retrieval misses,
and the cause of failed runs.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Query
from sqlalchemy import func

from agentops.db import AgentRun, RunLabel, SessionLocal

router = APIRouter()

OK_STATUSES = {"success"}


def _labels_by_trace(session, names):
    rows = session.query(RunLabel).filter(RunLabel.label_name.in_(names)).all()
    out = defaultdict(dict)
    for row in rows:
        out[row.trace_id][row.label_name] = row.label_value
    return out


@router.get("/labels/summary")
def labels_summary():
    session = SessionLocal()
    try:
        labels = _labels_by_trace(session, ["answer_correct", "retrieval_hit", "rag_outcome",
                                            "question_type", "doc_type", "source_correct", "run_set"])
        runs = {r.trace_id: r for r in session.query(AgentRun).filter(
            AgentRun.trace_id.in_(list(labels))).all()} if labels else {}

        total = correct = 0
        by_type, by_doc = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        outcomes, hits = Counter(), Counter()
        silent = 0   # run status success, answer wrong

        # Synthetic documents vs real public documents (doc_type starts with "Real ")
        parts = {name: {"total": 0, "correct": 0, "silent_failures": 0,
                        "by_question_type": defaultdict(lambda: [0, 0]),
                        "by_doc_type": defaultdict(lambda: [0, 0]),
                        "outcomes": Counter()}
                 for name in ("synthetic", "real", "real_after", "real_heldout", "real_heldout_nofix")}

        for trace_id, lab in labels.items():
            if "answer_correct" not in lab:
                continue
            ok = lab["answer_correct"] == "Yes"
            total += 1
            correct += ok
            by_type[lab.get("question_type", "?")][0] += 1
            by_type[lab.get("question_type", "?")][1] += ok
            by_doc[lab.get("doc_type", "?")][0] += 1
            by_doc[lab.get("doc_type", "?")][1] += ok
            outcomes[lab.get("rag_outcome", "?")] += 1
            hits[lab.get("retrieval_hit", "?")] += 1
            run = runs.get(trace_id)
            is_silent = (not ok) and run is not None and run.status in OK_STATUSES
            if is_silent:
                silent += 1

            if not str(lab.get("doc_type", "")).startswith("Real "):
                part = parts["synthetic"]
            elif lab.get("run_set") == "after":
                part = parts["real_after"]
            elif lab.get("run_set") == "heldout_nofix":
                part = parts["real_heldout_nofix"]
            elif lab.get("run_set") == "heldout":
                part = parts["real_heldout"]
            else:
                part = parts["real"]
            part["total"] += 1
            part["correct"] += ok
            part["silent_failures"] += is_silent
            part["by_question_type"][lab.get("question_type", "?")][0] += 1
            part["by_question_type"][lab.get("question_type", "?")][1] += ok
            part["by_doc_type"][lab.get("doc_type", "?")][0] += 1
            part["by_doc_type"][lab.get("doc_type", "?")][1] += ok
            part["outcomes"][lab.get("rag_outcome", "?")] += 1

        failure_rows = session.query(RunLabel).filter(RunLabel.label_name == "failure_type").all()
        failure_types = Counter(r.label_value for r in failure_rows)

        def pack(groups):
            return [{"name": k, "total": v[0], "correct": v[1]} for k, v in groups.items()]

        def pack_part(part):
            return {
                "total": part["total"],
                "correct": part["correct"],
                "silent_failures": part["silent_failures"],
                "by_question_type": pack(part["by_question_type"]),
                "by_doc_type": pack(part["by_doc_type"]),
                "outcomes": dict(part["outcomes"]),
            }

        return {
            "synthetic": pack_part(parts["synthetic"]),
            "real": pack_part(parts["real"]),
            "real_after": pack_part(parts["real_after"]),
            "real_heldout": pack_part(parts["real_heldout"]),
            "real_heldout_nofix": pack_part(parts["real_heldout_nofix"]),
            "document": {
                "total": total,
                "correct": correct,
                "silent_failures": silent,
                "by_question_type": pack(by_type),
                "by_doc_type": pack(by_doc),
                "outcomes": dict(outcomes),
                "retrieval_hit": dict(hits),
            },
            "failure_types": [{"failure_type": k, "runs": v} for k, v in failure_types.most_common()],
        }
    finally:
        session.close()


@router.get("/run-labels/{trace_id}")
def run_labels(trace_id: str):
    session = SessionLocal()
    try:
        rows = session.query(RunLabel).filter(RunLabel.trace_id == trace_id).all()
        return {r.label_name: r.label_value for r in rows}
    finally:
        session.close()


@router.get("/failures")
def failures(limit: int = Query(default=60, ge=1, le=300)):
    """Latest problem runs of EVERY agent (up to `limit` per agent).

    kind = "Run failed"        status error / failed
    kind = "Flagged"           run finished but flagged (partial success, needs
                               review, insufficient sources, max rounds ...)
    kind = "Silent failure"    status success, but the answer was wrong
    """
    session = SessionLocal()
    try:
        names = [r[0] for r in session.query(AgentRun.agent_name).distinct().all() if r[0]]
        problem_runs = []   # (run, kind)

        for name in names:
            rows = (session.query(AgentRun)
                    .filter(AgentRun.agent_name == name)
                    .filter(AgentRun.status.notin_(list(OK_STATUSES) + ["running", "interrupted"]))
                    .order_by(AgentRun.timestamp.desc()).limit(limit).all())
            for run in rows:
                kind = "Run failed" if run.status in ("error", "failed", "failure") else "Flagged"
                problem_runs.append((run, kind))

        wrong_ids = [r[0] for r in session.query(RunLabel.trace_id).filter(
            RunLabel.label_name == "answer_correct", RunLabel.label_value == "No").all()]
        if wrong_ids:
            silent = (session.query(AgentRun)
                      .filter(AgentRun.trace_id.in_(wrong_ids), AgentRun.status.in_(list(OK_STATUSES)))
                      .order_by(AgentRun.timestamp.desc()).limit(limit).all())
            problem_runs += [(run, "Silent failure") for run in silent]

        ids = [run.trace_id for run, _ in problem_runs]
        lab = _labels_by_trace(session, ["failure_type", "failure_stage", "rag_outcome"]) if ids else {}

        def row(run, kind):
            tags = lab.get(run.trace_id, {})
            return {
                "trace_id": run.trace_id,
                "timestamp": run.timestamp,
                "agent_name": run.agent_name,
                "status": run.status,
                "kind": kind,
                "cause": (tags.get("failure_type") or tags.get("rag_outcome")
                          or run.error_type or (run.status or "").replace("_", " ").title()),
                "stage": tags.get("failure_stage"),
                "error_message": (run.error_message or "")[:300],
                "user_input": (run.user_input or "")[:200],
            }

        items = [row(run, kind) for run, kind in problem_runs]
        items.sort(key=lambda x: x["timestamp"] or "", reverse=True)
        return {"total": len(items), "failures": items}
    finally:
        session.close()


@router.get("/labels/document-outcomes")
def document_outcomes(hours: int | None = Query(default=None, ge=1)):
    """Splits the document agent's "success" runs by answer quality.

    The dashboard uses this so "100% success" is not shown for an agent whose
    answers are wrong: success = correct answers only; the rest are
    retrieval miss / generation miss / wrong answer, plus runs not judged
    (test runs without a known answer).
    """
    session = SessionLocal()
    try:
        base = [AgentRun.agent_name == "document_agent", AgentRun.status == "success"]
        if hours:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            base.append(AgentRun.timestamp >= cutoff)

        success_total = session.query(func.count(AgentRun.trace_id)).filter(*base).scalar() or 0

        rows = (session.query(RunLabel.label_value, func.count(RunLabel.trace_id))
                .join(AgentRun, AgentRun.trace_id == RunLabel.trace_id)
                .filter(RunLabel.label_name == "rag_outcome", *base)
                .group_by(RunLabel.label_value).all())
        outcomes = {value: count for value, count in rows}
        judged = sum(outcomes.values())

        return {
            "hours": hours,
            "success_total": success_total,
            "correct": outcomes.get("Correct", 0),
            "outcomes": {k: v for k, v in outcomes.items() if k != "Correct"},
            "not_judged": max(0, success_total - judged),
        }
    finally:
        session.close()