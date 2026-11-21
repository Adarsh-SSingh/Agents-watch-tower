"""Database tables. Same tables as v1, plus three pre-run feature columns
on execution_events: max_tokens, input_chars, temperature, plus the new
run_labels table (quality labels and failure causes per run)."""

from sqlalchemy import Column, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Trace(Base):
    """Legacy one-row-per-run table, kept for backward compatibility."""

    __tablename__ = "traces"

    trace_id = Column(String, primary_key=True)
    timestamp = Column(String, nullable=False)

    input = Column(Text)
    output = Column(Text)
    model = Column(String)
    category = Column(String)
    agent_name = Column(String, default="legacy")

    source = Column(String, default="legacy")
    prompt_length = Column(Integer)
    estimated_input_tokens = Column(Integer)
    max_tokens = Column(Integer)
    temperature = Column(Float)
    thinking_mode = Column(String)

    input_tokens = Column(Integer)
    output_tokens = Column(Integer)
    latency = Column(Float)
    status = Column(String)
    error_type = Column(String)

    injected_fault = Column(String)
    tokens = Column(Integer)
    finish_reason = Column(String)


class AgentRun(Base):
    """One row per agent run."""

    __tablename__ = "agent_runs"

    trace_id = Column(String, primary_key=True)
    timestamp = Column(String, nullable=False)

    agent_name = Column(String, nullable=False)
    user_input = Column(Text)
    final_output = Column(Text)

    status = Column(String)
    total_latency = Column(Float)

    total_input_tokens = Column(Integer, default=0)
    total_output_tokens = Column(Integer, default=0)

    error_type = Column(String)
    error_message = Column(Text)


class ExecutionEvent(Base):
    """One row per LLM call, tool call or pipeline stage."""

    __tablename__ = "execution_events"

    event_id = Column(String, primary_key=True)
    trace_id = Column(String, nullable=False, index=True)

    sequence_number = Column(Integer, nullable=False)
    timestamp = Column(String, nullable=False)

    event_type = Column(String, nullable=False)
    event_name = Column(String, nullable=False)

    input_data = Column(Text)
    output_data = Column(Text)

    model = Column(String)
    latency = Column(Float)

    # Pre-run features (known before the call is made)
    max_tokens = Column(Integer)
    input_chars = Column(Integer)
    temperature = Column(Float)

    # Post-run measurements
    input_tokens = Column(Integer)
    output_tokens = Column(Integer)

    status = Column(String)
    error_type = Column(String)
    error_message = Column(Text)

    injected_fault = Column(String)
    finish_reason = Column(String)


class RunLabel(Base):
    """A label attached to one run: answer_correct, retrieval_hit,
    rag_failure_type, failure_type ... One row per (run, label)."""

    __tablename__ = "run_labels"

    trace_id = Column(String, primary_key=True)
    label_name = Column(String, primary_key=True)
    label_value = Column(String)
    source = Column(String)
    created_at = Column(String)