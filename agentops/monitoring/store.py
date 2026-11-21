"""Low-level persistence of runs, events and legacy traces."""

import json
import uuid
from datetime import datetime, timezone

from agentops.db import AgentRun, ExecutionEvent, SessionLocal, Trace


def current_timestamp():
    return datetime.now(timezone.utc).isoformat()


def serialize_data(data):
    if data is None:
        return None

    return json.dumps(data, ensure_ascii=False, default=str)


def start_agent_run(agent_name, user_input):
    trace_id = str(uuid.uuid4())

    run = AgentRun(
        trace_id=trace_id,
        timestamp=current_timestamp(),
        agent_name=agent_name,
        user_input=user_input,
        status="running",
        total_latency=0,
        total_input_tokens=0,
        total_output_tokens=0,
    )

    session = SessionLocal()

    try:
        session.add(run)
        session.commit()
        return trace_id

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


def log_execution_event(
    trace_id,
    sequence_number,
    event_type,
    event_name,
    input_data=None,
    output_data=None,
    model=None,
    latency=None,
    input_tokens=None,
    output_tokens=None,
    status="success",
    error_type=None,
    error_message=None,
    injected_fault=None,
    finish_reason=None,
    max_tokens=None,
    input_chars=None,
    temperature=None,
):
    event = ExecutionEvent(
        event_id=str(uuid.uuid4()),
        trace_id=trace_id,
        sequence_number=sequence_number,
        timestamp=current_timestamp(),
        event_type=event_type,
        event_name=event_name,
        input_data=serialize_data(input_data),
        output_data=serialize_data(output_data),
        model=model,
        latency=latency,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        status=status,
        error_type=error_type,
        error_message=error_message,
        injected_fault=injected_fault,
        finish_reason=finish_reason,
        max_tokens=max_tokens,
        input_chars=input_chars,
        temperature=temperature,
    )

    session = SessionLocal()

    try:
        session.add(event)
        session.commit()
        return event

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


def complete_agent_run(
    trace_id,
    final_output,
    status,
    total_latency,
    total_input_tokens=0,
    total_output_tokens=0,
    error_type=None,
    error_message=None,
):
    session = SessionLocal()

    try:
        run = session.get(AgentRun, trace_id)

        if run is None:
            raise ValueError(f"Agent run not found: {trace_id}")

        run.final_output = final_output
        run.status = status
        run.total_latency = total_latency
        run.total_input_tokens = total_input_tokens
        run.total_output_tokens = total_output_tokens
        run.error_type = error_type
        run.error_message = error_message

        session.commit()
        return run

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()


def log_trace(
    input_text,
    output_text,
    model,
    latency,
    status="success",
    tokens=0,
    *,
    source="real",
    category=None,
    agent_name="legacy",
    prompt_length=None,
    estimated_input_tokens=None,
    input_tokens=None,
    output_tokens=None,
    max_tokens=None,
    temperature=None,
    thinking_mode=None,
    error_type=None,
    injected_fault=None,
    finish_reason=None,
):
    """Write one row to the legacy `traces` table."""

    trace = Trace(
        trace_id=str(uuid.uuid4()),
        timestamp=current_timestamp(),
        input=input_text,
        output=output_text,
        model=model,
        source=source,
        category=category,
        agent_name=agent_name,
        prompt_length=prompt_length,
        estimated_input_tokens=estimated_input_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking_mode=thinking_mode,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency=latency,
        status=status,
        error_type=error_type,
        injected_fault=injected_fault,
        tokens=tokens,
        finish_reason=finish_reason,
    )

    session = SessionLocal()

    try:
        session.add(trace)
        session.commit()
        return trace

    except Exception:
        session.rollback()
        raise

    finally:
        session.close()