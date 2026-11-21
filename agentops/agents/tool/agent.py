"""
Tool Agent: a LangChain tool-calling loop.

Same behaviour as v1 (round limit, tool-call limit, duplicate-call
prevention, independent task evaluation). What changed:
  * tools are LangChain StructuredTools with validated argument schemas
  * the model is bound to the tools with `bind_tools`
  * LLM calls and tool calls are recorded by the callback handler
"""

import json
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agentops.agents.tool.prompts import TOOL_AGENT_SYSTEM_PROMPT
from agentops.agents.tool.tools import TOOLS, TOOLS_BY_NAME
from agentops.evaluation import evaluate_task_outcome
from agentops.llm import get_chat_model
from agentops.monitoring import RunRecorder

AGENT_NAME = "tool_agent"

MAX_TOOL_ROUNDS = 12
MAX_TOOL_CALLS = 20


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, default=str)


def fault_label(tool_name, arguments):
    """v1 behaviour: the flaky_tool mode is recorded as the injected fault."""

    if tool_name != "flaky_tool":
        return None

    return arguments.get("mode") if isinstance(arguments, dict) else None


def run_tool_agent(user_query: str):

    if not isinstance(user_query, str) or not user_query.strip():
        raise ValueError("User query cannot be empty.")

    user_query = user_query.strip()

    recorder = RunRecorder(AGENT_NAME, user_query)

    llm = get_chat_model().bind_tools(
        TOOLS,
        tool_choice="auto",
        parallel_tool_calls=False,
    )

    messages = [
        SystemMessage(content=TOOL_AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]

    executed_tools = []
    tool_results = []

    llm_round_count = 0
    tool_call_count = 0

    seen_tool_calls = set()

    # --------------------------------------------------
    # FINISH RUN + EVALUATION
    # --------------------------------------------------

    def finish_run(answer, status, error_type=None, error_message=None):

        # v1 behaviour: run totals count only the agent's own LLM calls.
        agent_input_tokens = recorder.total_input_tokens
        agent_output_tokens = recorder.total_output_tokens

        evaluation = evaluate_task_outcome(
            user_input=user_query,
            final_output=answer,
            tool_results=tool_results,
            execution_status=status,
            recorder=recorder,
        )

        evaluation_ok = evaluation.get("evaluator_status") == "success"

        # Tokens and latency of the judge call live in its own `llm` event
        # (task_evaluation). This event only carries the verdict.
        recorder.log_event(
            event_type="evaluation",
            event_name="task_completion_evaluation",
            input_data={
                "user_request": user_query,
                "execution_status": status,
                "tool_count": tool_call_count,
            },
            output_data=evaluation,
            latency=evaluation.get("evaluator_latency"),
            status="success" if evaluation_ok else "error",
            error_type=None if evaluation_ok else "EvaluatorError",
            error_message=evaluation.get("evaluator_error"),
        )

        recorder.total_input_tokens = agent_input_tokens
        recorder.total_output_tokens = agent_output_tokens

        # v1 behaviour: total latency is measured after the evaluation.
        recorder.complete(
            final_output=answer,
            status=status,
            total_latency=recorder.elapsed(),
            error_type=error_type,
            error_message=error_message,
        )

        return {
            "trace_id": recorder.trace_id,
            "answer": answer,

            # Runtime execution status
            "status": status,
            "execution_status": status,

            # Independent task evaluation
            "task_status": evaluation.get("task_status"),
            "evaluation": evaluation,

            # Execution details
            "executed_tools": executed_tools,
            "tool_results": tool_results,
            "tool_call_count": tool_call_count,
            "llm_rounds": llm_round_count,
        }

    # --------------------------------------------------
    # MAIN AGENT LOOP
    # --------------------------------------------------

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):

        llm_round_count = round_number

        # ----------------------------------------------
        # LLM CALL
        # ----------------------------------------------

        try:
            ai_message = llm.invoke(
                messages,
                config=recorder.config(
                    "tool_selection", input_data={"round": round_number}
                ),
            )

        except Exception as error:

            error_type = type(error).__name__
            error_message = str(error)

            answer = (
                "I couldn't complete the request because the "
                "language model request failed: "
                f"{error_message}"
            )

            return finish_run(
                answer,
                status="error",
                error_type=error_type,
                error_message=error_message,
            )

        messages.append(ai_message)

        pending_calls = [
            {"name": c["name"], "args": c["args"], "id": c["id"], "error": None}
            for c in ai_message.tool_calls
        ] + [
            # The model produced a call whose arguments are not valid JSON.
            {
                "name": c.get("name"),
                "args": {},
                "id": c.get("id"),
                "error": c.get("error")
                or "Tool arguments must be a JSON object.",
            }
            for c in ai_message.invalid_tool_calls
        ]

        # ----------------------------------------------
        # FINAL RESPONSE
        # ----------------------------------------------

        if not pending_calls:

            answer = (
                ai_message.content
                or "The model returned an empty response."
            )

            execution_status = (
                "success"
                if all(item["status"] == "success" for item in executed_tools)
                else "partial_success"
            )

            return finish_run(answer, status=execution_status)

        # ----------------------------------------------
        # TOOL EXECUTION
        # ----------------------------------------------

        for call in pending_calls:

            tool_name = call["name"]
            arguments = call["args"]

            tool_start = time.perf_counter()
            invoked = False

            tool_result = None
            tool_status = "error"
            error_type = None
            error_message = None

            try:

                if call["error"]:
                    raise ValueError(call["error"])

                fingerprint = (tool_name, safe_json(arguments))

                if fingerprint in seen_tool_calls:
                    raise RuntimeError(
                        "Duplicate tool call prevented: "
                        "this tool was already called "
                        "with the same arguments."
                    )

                if tool_call_count >= MAX_TOOL_CALLS:
                    raise RuntimeError(
                        f"Maximum tool-call limit ({MAX_TOOL_CALLS}) reached."
                    )

                seen_tool_calls.add(fingerprint)
                tool_call_count += 1

                tool = TOOLS_BY_NAME.get(tool_name)

                if tool is None:
                    raise ValueError(f"Unknown tool: {tool_name}")

                # The callback handler records the tool event
                # (latency, input, output, error, injected fault).
                invoked = True
                tool_result = tool.invoke(
                    arguments,
                    config=recorder.config(
                        injected_fault=fault_label(tool_name, arguments)
                    ),
                )

                tool_status = "success"

            except Exception as error:

                error_type = type(error).__name__
                error_message = str(error)

                tool_result = {
                    "error": error_message,
                    "error_type": error_type,
                }

                # Failures before the tool ran (bad JSON, duplicate, limit,
                # unknown tool) never reach the handler, so log them here.
                if not invoked:
                    recorder.log_event(
                        event_type="tool",
                        event_name=tool_name or "unknown_tool",
                        input_data=arguments,
                        output_data=tool_result,
                        latency=round(time.perf_counter() - tool_start, 6),
                        status="error",
                        error_type=error_type,
                        error_message=error_message,
                        injected_fault=fault_label(tool_name, arguments),
                    )

            tool_results.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "result": tool_result,
                "status": tool_status,
                "error": error_message,
            })

            executed_tools.append({
                "tool_name": tool_name,
                "status": tool_status,
                "error": error_message,
            })

            # Every tool call must be answered, or the next LLM call is invalid.
            messages.append(
                ToolMessage(
                    content=safe_json(tool_result),
                    tool_call_id=call["id"] or "unknown",
                )
            )

            if tool_call_count >= MAX_TOOL_CALLS:

                answer = (
                    "The request is only partially completed "
                    f"because the maximum limit of {MAX_TOOL_CALLS} "
                    "tool calls was reached. The successful and "
                    "failed tool executions are listed in the "
                    "execution trace. Please split the remaining "
                    "tasks into a follow-up request."
                )

                return finish_run(answer, status="max_tool_calls_reached")

    # --------------------------------------------------
    # MAX LLM ROUNDS
    # --------------------------------------------------

    answer = (
        "The request could not be fully completed because "
        f"the agent reached the maximum of {MAX_TOOL_ROUNDS} "
        "LLM tool-selection rounds. Some requested tasks may "
        "remain unfinished. Review the execution trace and "
        "submit the remaining tasks as a follow-up."
    )

    return finish_run(answer, status="max_rounds_reached")