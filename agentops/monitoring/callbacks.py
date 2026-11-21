"""
LangChain callback handler that turns every LLM call, tool call and
retrieval into one `execution_event` row.

Agents never time or log these calls by hand. They pass
`recorder.config(...)` to `invoke()` and the handler does the rest.
"""

import time
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

EVENT_NAME_KEY = "agentops_event"
EVENT_INPUT_KEY = "agentops_input"
EVENT_FAULT_KEY = "agentops_fault"

MAX_LOGGED_TEXT_CHARS = 20000


def _count_input_chars(messages) -> int:
    """Characters in message content (a pre-run ML feature)."""

    total = 0

    for batch in messages or []:
        for message in batch:
            content = getattr(message, "content", "")

            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                total += sum(len(str(item)) for item in content)
            elif content is not None:
                total += len(str(content))

    return total


def _truncate(text: Optional[str]) -> Optional[str]:
    if text is not None and len(text) > MAX_LOGGED_TEXT_CHARS:
        return text[:MAX_LOGGED_TEXT_CHARS] + "...[truncated]"
    return text


class AgentOpsCallbackHandler(BaseCallbackHandler):
    """Writes execution events through a RunRecorder."""

    # A monitoring failure must never break the agent.
    raise_error = False

    def __init__(self, recorder):
        self.recorder = recorder
        self._calls: Dict[UUID, Dict[str, Any]] = {}

    # --------------------------------------------------
    # LLM CALLS
    # --------------------------------------------------

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        params = kwargs.get("invocation_params") or {}
        metadata = metadata or {}

        self._calls[run_id] = {
            "start": time.perf_counter(),
            "event_name": metadata.get(EVENT_NAME_KEY) or "llm_call",
            "input_data": metadata.get(EVENT_INPUT_KEY),
            "model": params.get("model") or params.get("model_name"),
            "max_tokens": (
                params.get("max_tokens")
                or params.get("max_completion_tokens")
                or (params.get("extra_body") or {}).get("max_tokens")
            ),
            "temperature": params.get("temperature"),
            "input_chars": _count_input_chars(messages),
        }

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        latency = round(time.perf_counter() - call["start"], 4)

        generation = response.generations[0][0]
        message = getattr(generation, "message", None)

        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")

        if input_tokens is None and response.llm_output:
            token_usage = response.llm_output.get("token_usage") or {}
            input_tokens = token_usage.get("prompt_tokens")
            output_tokens = token_usage.get("completion_tokens")

        finish_reason = (
            (generation.generation_info or {}).get("finish_reason")
            or (getattr(message, "response_metadata", None) or {}).get(
                "finish_reason"
            )
        )

        output_data: Dict[str, Any] = {"content": _truncate(generation.text)}

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            output_data["tool_calls"] = tool_calls

        self._record_llm(
            call,
            latency=latency,
            status="success",
            output_data=output_data,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        self._record_llm(
            call,
            latency=round(time.perf_counter() - call["start"], 4),
            status="error",
            output_data=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    def _record_llm(self, call, *, latency, status, output_data, **extra):
        try:
            self.recorder.log_llm_event(
                event_name=call["event_name"],
                input_data=call["input_data"],
                output_data=output_data,
                model=call["model"],
                latency=latency,
                status=status,
                max_tokens=call["max_tokens"],
                input_chars=call["input_chars"],
                temperature=call["temperature"],
                **extra,
            )
        except Exception as error:  # noqa: BLE001
            print(f"[AgentOps] Could not record LLM event: {error}")

    # --------------------------------------------------
    # TOOL CALLS
    # --------------------------------------------------

    def on_tool_start(
        self,
        serialized,
        input_str,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        inputs=None,
        **kwargs,
    ):
        metadata = metadata or {}

        self._calls[run_id] = {
            "start": time.perf_counter(),
            "event_name": (
                (serialized or {}).get("name") or kwargs.get("name") or "tool"
            ),
            "input_data": inputs if inputs is not None else {"input": input_str},
            "injected_fault": metadata.get(EVENT_FAULT_KEY),
        }

    def on_tool_end(self, output, *, run_id, parent_run_id=None, **kwargs):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        content = getattr(output, "content", output)

        self._record_simple(
            "tool", call, status="success", output_data=content
        )

    def on_tool_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        self._record_simple(
            "tool",
            call,
            status="error",
            output_data={
                "error": str(error),
                "error_type": type(error).__name__,
            },
            error_type=type(error).__name__,
            error_message=str(error),
        )

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    def on_retriever_start(
        self,
        serialized,
        query,
        *,
        run_id,
        parent_run_id=None,
        tags=None,
        metadata=None,
        **kwargs,
    ):
        metadata = metadata or {}

        self._calls[run_id] = {
            "start": time.perf_counter(),
            "event_name": "document_search",
            "input_data": metadata.get(EVENT_INPUT_KEY) or {"query": query},
        }

    def on_retriever_end(
        self, documents: List[Any], *, run_id, parent_run_id=None, **kwargs
    ):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        sources = list(dict.fromkeys(
            doc.metadata.get("source_file", "Unknown") for doc in documents
        ))

        chunks = [
            {
                "source": doc.metadata.get("source_file", "Unknown"),
                "document_id": doc.metadata.get("document_id"),
                "page": doc.metadata.get("page"),
                "start_index": doc.metadata.get("start_index"),
                "score": doc.metadata.get("retrieval_score"),
                "text": doc.page_content,
            }
            for doc in documents
        ]

        self._record_simple(
            "retrieval",
            call,
            status="success",
            output_data={
                "retrieved_chunks": len(documents),
                "sources": sources,
                "chunks": chunks,
            },
        )

    def on_retriever_error(self, error, *, run_id, parent_run_id=None, **kwargs):
        call = self._calls.pop(run_id, None)

        if call is None:
            return

        self._record_simple(
            "retrieval",
            call,
            status="error",
            output_data=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    # --------------------------------------------------

    def _record_simple(self, event_type, call, *, status, output_data, **extra):
        try:
            self.recorder.log_event(
                event_type=event_type,
                event_name=call["event_name"],
                input_data=call.get("input_data"),
                output_data=output_data,
                latency=round(time.perf_counter() - call["start"], 6),
                status=status,
                injected_fault=call.get("injected_fault"),
                **extra,
            )
        except Exception as error:  # noqa: BLE001
            print(f"[AgentOps] Could not record {event_type} event: {error}")