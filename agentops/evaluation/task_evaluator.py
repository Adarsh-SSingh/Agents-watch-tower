"""
Task outcome evaluator.

Separates runtime execution health from whether the agent fulfilled the
user's request. The judge call goes through the same callback handler as
every other LLM call, so it shows up as its own `llm` event.
"""

import json
import re
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agentops.llm import get_chat_model


EVALUATOR_SYSTEM_PROMPT = """
You are an independent evaluator for an AI agent observability platform.

Evaluate whether the agent's final answer actually satisfies the user's request.
Do not assume a task succeeded merely because tools ran without runtime errors.
Compare the request, final answer, and available tool results.

Return ONLY one valid JSON object with exactly these keys:
{
  "task_status": "success" | "partial_success" | "failed" | "needs_review",
  "completed_tasks": ["..."],
  "incomplete_tasks": ["..."],
  "reasoning": "brief explanation",
  "confidence": 0.0
}

Rules:
- success: all meaningful requested tasks are answered with relevant, supported results.
- partial_success: at least one meaningful task is completed, but one or more are incomplete,
  irrelevant, unsupported, or only partially answered.
- failed: none of the meaningful requested tasks were completed.
- needs_review: the evidence is too ambiguous to make a reliable judgment.
- A tool returning successfully does NOT mean the user's task was completed.
- Do not reward hallucinated details or irrelevant search results.
- If the user asks for current information but the available evidence is not current,
  mark that part incomplete.
- Keep completed_tasks and incomplete_tasks specific and concise.
- confidence must be a number between 0 and 1.
"""


def _extract_json(text: str) -> Dict[str, Any]:
    """Parse JSON from a plain or fenced model response."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Evaluator returned an empty response.")

    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError("Evaluator response did not contain a JSON object.")
        result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Evaluator response must be a JSON object.")
    return result


def _normalise_evaluation(data: Dict[str, Any]) -> Dict[str, Any]:
    allowed_statuses = {"success", "partial_success", "failed", "needs_review"}
    status = data.get("task_status")

    if status not in allowed_statuses:
        status = "needs_review"

    confidence = data.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    completed = data.get("completed_tasks", [])
    incomplete = data.get("incomplete_tasks", [])

    if not isinstance(completed, list):
        completed = []
    if not isinstance(incomplete, list):
        incomplete = []

    return {
        "task_status": status,
        "completed_tasks": [str(item) for item in completed],
        "incomplete_tasks": [str(item) for item in incomplete],
        "reasoning": str(data.get("reasoning", "No explanation provided.")),
        "confidence": confidence,
    }


EVALUATION_EVENT_NAME = "task_evaluation"

evaluator_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
    (
        "human",
        "Evaluate this agent run using the required JSON format.\n\n{payload}",
    ),
])


def evaluate_task_outcome(
    user_input: str,
    final_output: str,
    tool_results: Optional[List[Dict[str, Any]]] = None,
    execution_status: str = "success",
    recorder=None,
) -> Dict[str, Any]:
    """
    Evaluate task completion independently of runtime execution status.

    If the evaluation itself fails, task_status becomes needs_review rather
    than incorrectly reporting task success.
    """
    payload = {
        "user_request": user_input,
        "agent_final_answer": final_output,
        "tool_results": tool_results or [],
        "runtime_execution_status": execution_status,
    }

    start_time = time.perf_counter()

    try:
        chain = evaluator_prompt | get_chat_model() | StrOutputParser()

        raw_output = chain.invoke(
            {"payload": json.dumps(payload, ensure_ascii=False, default=str)},
            config=(
                recorder.config(EVALUATION_EVENT_NAME) if recorder else None
            ),
        )

        evaluation = _normalise_evaluation(_extract_json(raw_output))

        return {
            "execution_status": execution_status,
            **evaluation,
            "evaluator_status": "success",
            "evaluator_latency": round(time.perf_counter() - start_time, 3),
            "evaluator_error": None,
        }

    except Exception as exc:
        return {
            "execution_status": execution_status,
            "task_status": "needs_review",
            "completed_tasks": [],
            "incomplete_tasks": ["Automatic task evaluation could not be completed."],
            "reasoning": f"Evaluation failed: {type(exc).__name__}: {exc}",
            "confidence": 0.0,
            "evaluator_status": "error",
            "evaluator_latency": round(time.perf_counter() - start_time, 3),
            "evaluator_error": str(exc),
        }