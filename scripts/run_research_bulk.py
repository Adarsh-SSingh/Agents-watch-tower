"""Run the research topic set through the Research Agent.

Usage (from project root):
    python scripts/run_research_bulk.py --limit 3     # smoke test
    python scripts/run_research_bulk.py               # run everything left

Needs TAVILY_API_KEY in .env. One run takes about 1-3 minutes.
Writes:
  data/research_data.csv        one readable row per run
  data/research_trace_map.csv   Run No -> Trace ID (join with the database)
Running again continues from the first topic not yet run.
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import psutil

from agentops.agents.research import pipeline as research_pipeline
from agentops.agents.research.pipeline import run_research_pipeline
from agentops.config import DATA_DIR
from agentops.datagen.research_prompts import build_research_prompts

TABLE_FILE = Path(DATA_DIR) / "research_data.csv"
MAP_FILE = Path(DATA_DIR) / "research_trace_map.csv"

FIELDS = [
    "Run No", "Status",
    "Topic Category", "Difficulty", "Topic Style",
    "Topic Length", "Word Count",
    "CPU Usage (%)", "RAM Usage (%)", "Free RAM (GB)",
    "Hour", "Weekday",
    "Run Time (s)",
    "Topic",
]
MAP_FIELDS = ["Run No", "Trace ID"]

# Runs that did NOT produce a usable row because of the API (not the topic).
# Kept as data: they show what failed and why.
FAIL_FILE = Path(DATA_DIR) / "research_failures.csv"
FAIL_FIELDS = [
    "Run No", "Failure Type", "Failure Stage", "Error Message", "Trace ID",
    "Topic Category", "Difficulty", "Topic Style", "Topic Length", "Word Count",
    "CPU Usage (%)", "RAM Usage (%)", "Free RAM (GB)", "Hour", "Weekday",
    "Run Time (s)", "Topic",
]


def nice(value) -> str:
    return str(value).replace("_", " ").title()


QUOTA_WORDS = (
    "usage limit", "quota", "limit", "exceeded", "429", "432", "433",
    "401", "unauthorized", "invalid api key", "api key", "credits", "plan",
)


def search_problem(result: dict):
    """Return the Tavily error text if any search query failed, else None.

    The pipeline swallows failed searches and can still finish with
    "insufficient_sources". Such a row would be fake data (caused by the API,
    not by the topic), so the runner must not save it.
    """
    search = result.get("search") or {}
    failed = search.get("failed_queries") or []
    if failed:
        return str(failed[0].get("error", "unknown search error"))
    return None


API_ERROR_WORDS = (
    "429", "rate limit", "ratelimit", "timed out", "timeout", "connection",
    "502", "503", "504", "overloaded", "temporarily",
)


def critic_api_problem(result: dict):
    """Return the critic error text if the critic failed because of the API.

    In the pipeline, status "needs_review" only means the critic did not
    produce a result. When the reason is a rate limit or timeout, the row says
    nothing about the report quality, so it must not be saved.
    """
    if result.get("status") != "needs_review":
        return None
    error = str(result.get("critic_error") or "")
    if any(word in error.lower() for word in API_ERROR_WORDS):
        return error
    return None


def failure_type(error_text: str) -> str:
    """Name the root cause of an infrastructure failure from its error text."""
    text = (error_text or "").lower()
    if "429" in text or "rate limit" in text or "ratelimit" in text:
        return "LLM Rate Limit"
    if "timed out" in text or "timeout" in text:
        return "Timeout"
    if "connection" in text or "502" in text or "503" in text or "504" in text:
        return "Network Error"
    if any(word in text for word in QUOTA_WORDS):
        return "Search Quota"
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


def completed_indexes() -> set:
    if not TABLE_FILE.exists():
        return set()
    with open(TABLE_FILE, newline="", encoding="utf-8") as file:
        return {int(row["Run No"]) - 1 for row in csv.DictReader(file)}


def append_csv(path: Path, fields: list, row: dict) -> None:
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Max topics to run in this session (must be >= 1).")
    parser.add_argument("--pause", type=float, default=2.0,
                        help="Seconds to wait between runs (must be >= 0).")
    parser.add_argument("--target", type=int, default=400,
                        help="Total number of topics in the set.")
    parser.add_argument("--queries", type=int, default=5,
                        help="Tavily searches per run (1 credit each). Default 5 "
                             "(same as the dashboard agent).")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be 1 or more")
    if args.pause < 0:
        parser.error("--pause cannot be negative")
    if args.target < 1:
        parser.error("--target must be 1 or more")
    if not 1 <= args.queries <= 5:
        parser.error("--queries must be between 1 and 5")

    # Bulk runs only: limit Tavily searches per run (agent code is not changed).
    research_pipeline.MAX_SEARCH_QUERIES = args.queries
    print(f"Tavily searches per run: {args.queries} "
          f"(about {args.queries * args.target} credits for {args.target} topics)")

    items = build_research_prompts(target=args.target)
    done = completed_indexes()
    pending = [(i, it) for i, it in enumerate(items) if i not in done]

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Topics total: {len(items)} | already done: {len(done)} | "
          f"running now: {len(pending)}")

    started = time.perf_counter()
    failures = 0

    try:
        for position, (index, item) in enumerate(pending, start=1):
            topic = item["topic"]
            print(f"\n[{position}/{len(pending)}] #{index} ({item['category']}) {topic}")

            state = machine_state()
            run_started = time.perf_counter()

            error_text = ""
            result = {}

            try:
                result = run_research_pipeline(topic)
                trace_id = result.get("trace_id") or ""
                status = result.get("status") or "unknown"
            except Exception as error:
                trace_id = ""
                status = "script_error"
                error_text = f"{type(error).__name__}: {error}"

            # A run is NOT saved to the main table when the API caused the
            # problem (not the topic). It is logged in research_failures.csv
            # with its cause, and retried the next time the script runs.
            # "insufficient_sources" is a real outcome and IS saved.
            reason = None   # (stage, error text)

            if status == "script_error":
                reason = ("script", error_text)
            else:
                problem = search_problem(result)
                critic_problem = critic_api_problem(result)

                if problem:
                    reason = ("search", problem)
                elif critic_problem:
                    reason = ("critic", critic_problem)
                elif status in ("failed", "unknown") or not trace_id:
                    reason = ("pipeline", str(result.get("error") or status))

            if reason:
                stage, error_text = reason
                kind = failure_type(error_text)
                print(f"  ! {kind} at {stage}: {error_text[:200]}")

                append_csv(FAIL_FILE, FAIL_FIELDS, {
                    "Run No": index + 1,
                    "Failure Type": kind,
                    "Failure Stage": stage.title(),
                    "Error Message": error_text[:300].replace("\n", " "),
                    "Trace ID": trace_id,
                    "Topic Category": nice(item["category"]),
                    "Difficulty": nice(item["difficulty"]),
                    "Topic Style": nice(item["style"]),
                    "Topic Length": len(topic),
                    "Word Count": len(topic.split()),
                    **state,
                    "Run Time (s)": round(time.perf_counter() - run_started, 3),
                    "Topic": topic,
                })

                if kind == "Search Quota":
                    print(
                        "\nSTOPPED: Tavily credits/limit finished (or key invalid).\n"
                        "Nothing was lost: this topic was logged in research_failures.csv "
                        "and all earlier rows are safe.\n"
                        "Put the new TAVILY_API_KEY in .env, then run the same command again."
                    )
                    break

                failures += 1
                print(f"  ! Run failed ({failures} in a row). Not saved to the main table; will retry later.")
                if failures >= 3:
                    print("\nStopping: 3 failures in a row (LLM API limit or network problem). "
                          "Wait 15-30 minutes and run again.")
                    break
                time.sleep(30 * failures)
                continue

            failures = 0

            append_csv(MAP_FILE, MAP_FIELDS, {"Run No": index + 1, "Trace ID": trace_id})
            append_csv(TABLE_FILE, FIELDS, {
                "Run No": index + 1,
                "Status": nice(status),
                "Topic Category": nice(item["category"]),
                "Difficulty": nice(item["difficulty"]),
                "Topic Style": nice(item["style"]),
                "Topic Length": len(topic),
                "Word Count": len(topic.split()),
                **state,
                "Run Time (s)": round(time.perf_counter() - run_started, 3),
                "Topic": topic,
            })
            print(f"  -> {nice(status)} in {round(time.perf_counter() - run_started, 1)} s")

            if args.pause:
                time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\nStopped by user. Run again to continue.")

    print(f"\nSession time: {round((time.perf_counter() - started) / 60, 1)} min")


if __name__ == "__main__":
    main()