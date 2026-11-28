"""One table for the resume / README: how every agent did, from data/agentops.db.

Usage:  python scripts\\agent_summary.py
"""

from collections import Counter, defaultdict
from statistics import median

from agentops.db import AgentRun, RunLabel, SessionLocal


def main():
    session = SessionLocal()
    runs = session.query(AgentRun).all()
    labels = defaultdict(dict)
    for row in session.query(RunLabel).filter(RunLabel.label_name.in_(
            ["answer_correct", "failure_type"])).all():
        labels[row.trace_id][row.label_name] = row.label_value

    by_agent = defaultdict(list)
    for run in runs:
        by_agent[run.agent_name].append(run)

    print(f"Total traced runs: {len(runs)}\n")
    for agent, items in sorted(by_agent.items()):
        n = len(items)
        ok = [r for r in items if r.status == "success"]
        lat = [r.total_latency for r in items if r.total_latency]
        tokens = sum((r.total_input_tokens or 0) + (r.total_output_tokens or 0) for r in items)
        judged = [labels[r.trace_id]["answer_correct"] for r in items if "answer_correct" in labels[r.trace_id]]
        causes = Counter(labels[r.trace_id]["failure_type"] for r in items if "failure_type" in labels[r.trace_id])
        statuses = Counter(r.status for r in items)

        print(f"== {agent}")
        print(f"   runs {n} | status success {len(ok)} ({len(ok) / n * 100:.1f}%) | all statuses {dict(statuses)}")
        if lat:
            print(f"   latency: median {median(lat):.1f}s | max {max(lat):.1f}s")
        print(f"   tokens used: {tokens:,}")
        if judged:
            right = judged.count("Yes")
            print(f"   answers judged {len(judged)} | correct {right} ({right / len(judged) * 100:.1f}%)")
        if causes:
            print("   failure causes: " + ", ".join(f"{k} {v}" for k, v in causes.most_common(5)))
        print()


if __name__ == "__main__":
    main()