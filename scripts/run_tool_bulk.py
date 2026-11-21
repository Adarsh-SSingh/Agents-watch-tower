"""Run a prompt set through an agent to generate AgentOps data.

Usage (from project root):
    python scripts/run_bulk.py --limit 5      # small smoke test
    python scripts/run_bulk.py                # run everything left

Every run is recorded in the AgentOps database by the agent itself.
This script also writes:
  data/realtable.csv      one readable row per run (features + results)
  data/run_trace_map.csv  Run No -> Trace ID, to join the table with the database
Running the script again continues from the first prompt not yet run.
"""

import argparse
import csv
import re
import time
from datetime import datetime
from pathlib import Path

import psutil

from agentops.agents.tool.agent import run_tool_agent
from agentops.config import DATA_DIR
from agentops.datagen.tool_prompts import build_tool_prompts

TABLE_FILE = Path(DATA_DIR) / "tool_data.csv"
MAP_FILE = Path(DATA_DIR) / "tool_trace_map.csv"

# Readable tool names for the table.
TOOL_LABELS = {
    "calculator": "Calculator",
    "get_system_info": "System Info",
    "get_weather": "Weather",
    "wikipedia_lookup": "Wikipedia",
    "flaky_tool": "Flaky Tool",
}

# Column names are written the way they should read in the table.
# Order: run id, status, task description, prompt features,
# machine state, run time, and the long prompt text last.
FIELDS = [
    "Run No", "Status",
    "Task Type", "Difficulty", "Prompt Style",
    "Expected Tools", "Expected Tool Count",
    "Prompt Length", "Word Count", "Digit Count",
    "CPU Usage (%)", "RAM Usage (%)", "Free RAM (GB)",
    "Hour", "Weekday",
    "Run Time (s)",
    "Prompt",
]
MAP_FIELDS = ["Run No", "Trace ID"]


def nice(value) -> str:
    """multi_tool -> Multi Tool, partial_success -> Partial Success."""
    return str(value).replace("_", " ").title()


def tools_label(tools: list) -> str:
    if not tools:
        return "None"
    return " + ".join(TOOL_LABELS.get(tool, tool) for tool in tools)


def prompt_features(prompt: str) -> dict:
    return {
        "Prompt Length": len(prompt),
        "Word Count": len(prompt.split()),
        "Digit Count": len(re.findall(r"\d", prompt)),
    }


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
        # Run No starts at 1; the prompt list index starts at 0.
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
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Maximum number of prompts to run in this session."
    )
    parser.add_argument(
        "--pause", type=float, default=1.0,
        help="Seconds to wait between runs (rate-limit friendly)."
    )
    parser.add_argument(
        "--target", type=int, default=1000,
        help="Total number of prompts in the set (default 1000)."
    )
    args = parser.parse_args()

    prompts = build_tool_prompts(target=args.target)
    done = completed_indexes()
    pending = [
        (index, item) for index, item in enumerate(prompts)
        if index not in done
    ]

    if args.limit:
        pending = pending[:args.limit]

    print(
        f"Prompts total: {len(prompts)} | already done: {len(done)} | "
        f"running now: {len(pending)}"
    )

    started = time.perf_counter()
    consecutive_failures = 0

    try:
        for position, (index, item) in enumerate(pending, start=1):
            print(
                f"\n[{position}/{len(pending)}] #{index} "
                f"({item['category']}) {item['prompt']}"
            )

            features = prompt_features(item["prompt"])
            state = machine_state()
            run_started = time.perf_counter()

            try:
                result = run_tool_agent(item["prompt"])
                trace_id = result.get("trace_id", "")
                status = result.get("status", "unknown")

            except Exception as error:
                # The agent normally records its own failures; this only
                # catches problems outside the agent itself.
                trace_id = ""
                status = f"script_error: {type(error).__name__}"
                print(f"  ! {status}: {error}")

            # In the Tool Agent, status "error" means the LLM request itself
            # failed (rate limit, timeout, network). Do not save such a row:
            # the prompt will be retried the next time the script runs.
            if status == "error" or status.startswith("script_error"):
                consecutive_failures += 1
                print(
                    f"  ! API/network failure ({consecutive_failures} in a row). "
                    "Not saved; this prompt will be retried later."
                )

                if consecutive_failures >= 3:
                    print(
                        "\nStopping: 3 failures in a row (probably the API limit). "
                        "Wait 15-30 minutes, then run the script again."
                    )
                    break

                time.sleep(30 * consecutive_failures)
                continue

            consecutive_failures = 0

            # Trace ids live in their own file so the table stays readable.
            append_csv(MAP_FILE, MAP_FIELDS, {
                "Run No": index + 1,
                "Trace ID": trace_id,
            })
            append_csv(TABLE_FILE, FIELDS, {
                "Run No": index + 1,
                "Status": nice(status),
                "Task Type": nice(item["category"]),
                "Difficulty": nice(item["difficulty"]),
                "Prompt Style": nice(item.get("style", "plain")),
                "Expected Tools": tools_label(item["expected_tools"]),
                "Expected Tool Count": len(item["expected_tools"]),
                **features,
                **state,
                "Run Time (s)": round(time.perf_counter() - run_started, 3),
                "Prompt": item["prompt"],
            })
            print(f"  -> {status}")

            time.sleep(args.pause)

    except KeyboardInterrupt:
        print("\nStopped. Run the script again to continue from where it stopped.")

    minutes = (time.perf_counter() - started) / 60
    print(f"\nFinished this session in {minutes:.1f} min. Table: {TABLE_FILE}")


if __name__ == "__main__":
    main()