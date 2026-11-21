import json
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import and_, case, func

from agentops.db import AgentRun, ExecutionEvent, SessionLocal

router = APIRouter()


def parse_json_field(value):
    if value is None:
        return None

    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return value


# AGENT RUNS

@router.get("/agent-runs")
def get_agent_runs(
    agent_name: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):
    session = SessionLocal()

    try:
        query = session.query(AgentRun)

        if agent_name:
            query = query.filter(AgentRun.agent_name == agent_name)

        if status:
            query = query.filter(AgentRun.status == status)

        total = query.count()

        runs = (
            query
            .order_by(AgentRun.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "runs": [
                {
                    "trace_id": run.trace_id,
                    "timestamp": run.timestamp,
                    "agent_name": run.agent_name,
                    "user_input": run.user_input,
                    "final_output": run.final_output,
                    "status": run.status,
                    "total_latency": run.total_latency,
                    "total_input_tokens": run.total_input_tokens,
                    "total_output_tokens": run.total_output_tokens,
                    "error_type": run.error_type,
                    "error_message": run.error_message
                }
                for run in runs
            ]
        }

    finally:
        session.close()


# AGENT STATS (true totals for the whole database)

# A run still "running" after this many hours was stopped half way
# (for example with Ctrl+C) and will never finish.
STALE_RUNNING_HOURS = 1


@router.get("/agent-stats")
def get_agent_stats(hours: int | None = Query(default=None, ge=1)):
    """Counts, latency sums and token sums per agent and status.

    Light-weight: no run text is returned, so the dashboard can show the
    real totals of the whole database without loading every run.

    Optional `hours` limits the numbers to the last N hours. The value is
    echoed back in the response so the dashboard can tell that the filter
    was really applied.
    """
    session = SessionLocal()

    try:
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(hours=STALE_RUNNING_HOURS)).isoformat()

        status_expr = case(
            (
                and_(AgentRun.status == "running", AgentRun.timestamp < cutoff),
                "interrupted",
            ),
            else_=AgentRun.status,
        )

        query = session.query(
            AgentRun.agent_name,
            status_expr.label("status"),
            func.count(AgentRun.trace_id),
            func.coalesce(func.sum(AgentRun.total_latency), 0.0),
            func.count(AgentRun.total_latency),
            func.coalesce(func.sum(AgentRun.total_input_tokens), 0),
            func.coalesce(func.sum(AgentRun.total_output_tokens), 0),
        )

        if hours:
            since = (now - timedelta(hours=hours)).isoformat()
            query = query.filter(AgentRun.timestamp >= since)

        rows = query.group_by(AgentRun.agent_name, status_expr).all()

        groups = [
            {
                "agent_name": agent_name,
                "status": status,
                "runs": runs,
                "latency_sum": float(latency_sum or 0),
                "latency_count": latency_count,
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
            }
            for (agent_name, status, runs, latency_sum,
                 latency_count, input_tokens, output_tokens) in rows
        ]

        return {
            "hours": hours,
            "total_runs": sum(group["runs"] for group in groups),
            "groups": groups,
        }

    finally:
        session.close()


# AGENT RUN DETAIL + EXECUTION TIMELINE

@router.get("/agent-runs/{trace_id}")
def get_agent_run_detail(trace_id: str):
    session = SessionLocal()

    try:
        run = session.get(AgentRun, trace_id)

        if run is None:
            raise HTTPException(
                status_code=404,
                detail="Agent run not found."
            )

        events = (
            session.query(ExecutionEvent)
            .filter(ExecutionEvent.trace_id == trace_id)
            .order_by(ExecutionEvent.sequence_number.asc())
            .all()
        )

        return {
            "run": {
                "trace_id": run.trace_id,
                "timestamp": run.timestamp,
                "agent_name": run.agent_name,
                "user_input": run.user_input,
                "final_output": run.final_output,
                "status": run.status,
                "total_latency": run.total_latency,
                "total_input_tokens": run.total_input_tokens,
                "total_output_tokens": run.total_output_tokens,
                "error_type": run.error_type,
                "error_message": run.error_message
            },
            "events": [
                {
                    "event_id": event.event_id,
                    "sequence_number": event.sequence_number,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "event_name": event.event_name,
                    "input_data": parse_json_field(event.input_data),
                    "output_data": parse_json_field(event.output_data),
                    "model": event.model,
                    "latency": event.latency,
                    "input_tokens": event.input_tokens,
                    "output_tokens": event.output_tokens,
                    "status": event.status,
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                    "injected_fault": event.injected_fault,
                    "finish_reason": event.finish_reason,
                    "max_tokens": event.max_tokens,
                    "input_chars": event.input_chars,
                    "temperature": event.temperature
                }
                for event in events
            ]
        }

    finally:
        session.close()