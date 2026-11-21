"""One RunRecorder per agent run: owns the trace_id, event order and totals."""

import threading
import time

from agentops.monitoring.callbacks import (
    EVENT_FAULT_KEY,
    EVENT_INPUT_KEY,
    EVENT_NAME_KEY,
    AgentOpsCallbackHandler,
)
from agentops.monitoring.store import (
    complete_agent_run,
    log_execution_event,
    start_agent_run,
)


class RunRecorder:
    def __init__(self, agent_name: str, user_input: str):
        self.agent_name = agent_name
        self.trace_id = start_agent_run(
            agent_name=agent_name, user_input=user_input
        )

        self.started_at = time.perf_counter()
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # One entry per LLM call, in order. Used for per-stage metrics.
        self.llm_calls = []

        self._sequence = 0
        self._lock = threading.Lock()
        self._completed = False

        self.handler = AgentOpsCallbackHandler(self)

    # --------------------------------------------------
    # LANGCHAIN CONFIG
    # --------------------------------------------------

    def config(self, event_name=None, input_data=None, injected_fault=None):
        """Pass this as `config=` to any LangChain invoke()."""

        metadata = {}

        if event_name:
            metadata[EVENT_NAME_KEY] = event_name
        if input_data is not None:
            metadata[EVENT_INPUT_KEY] = input_data
        if injected_fault is not None:
            metadata[EVENT_FAULT_KEY] = injected_fault

        return {"callbacks": [self.handler], "metadata": metadata}

    # --------------------------------------------------
    # EVENTS
    # --------------------------------------------------

    def log_event(self, **kwargs):
        with self._lock:
            self._sequence += 1
            sequence = self._sequence

        return log_execution_event(
            trace_id=self.trace_id,
            sequence_number=sequence,
            **kwargs,
        )

    def log_llm_event(self, event_name, **kwargs):
        """LLM events also feed the run totals and the per-call history."""

        input_tokens = kwargs.get("input_tokens")
        output_tokens = kwargs.get("output_tokens")

        with self._lock:
            self.total_input_tokens += input_tokens or 0
            self.total_output_tokens += output_tokens or 0

            self.llm_calls.append({
                "event_name": event_name,
                "latency": kwargs.get("latency"),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "finish_reason": kwargs.get("finish_reason"),
                "status": kwargs.get("status"),
            })

        return self.log_event(
            event_type="llm", event_name=event_name, **kwargs
        )

    def llm_stats(self, event_name):
        """Summed stats of every LLM call with this event name."""

        calls = [c for c in self.llm_calls if c["event_name"] == event_name]

        if not calls:
            return {}

        return {
            "latency": round(sum(c["latency"] or 0 for c in calls), 4),
            "tokens": sum(c["output_tokens"] or 0 for c in calls),
            "calls": len(calls),
        }

    # --------------------------------------------------
    # RUN COMPLETION
    # --------------------------------------------------

    def elapsed(self, digits=3):
        return round(time.perf_counter() - self.started_at, digits)

    def complete(
        self,
        final_output,
        status,
        total_latency=None,
        error_type=None,
        error_message=None,
    ):
        if self._completed:
            return

        complete_agent_run(
            trace_id=self.trace_id,
            final_output=final_output,
            status=status,
            total_latency=(
                total_latency if total_latency is not None else self.elapsed()
            ),
            total_input_tokens=self.total_input_tokens,
            total_output_tokens=self.total_output_tokens,
            error_type=error_type,
            error_message=error_message,
        )

        self._completed = True