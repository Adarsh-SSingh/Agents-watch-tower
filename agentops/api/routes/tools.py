from fastapi import APIRouter, Query
from sqlalchemy import case, func

from agentops.api.routes.runs import parse_json_field
from agentops.db import AgentRun, ExecutionEvent, SessionLocal

router = APIRouter()

# --------------------------------------------------
# TOOL AGENT ANALYTICS
# --------------------------------------------------

@router.get("/tool-analytics")
def get_tool_analytics():

    session = SessionLocal()

    try:
        # Only executions belonging to Tool Agent
        tool_events = (
            session.query(ExecutionEvent)
            .join(
                AgentRun,
                ExecutionEvent.trace_id == AgentRun.trace_id
            )
            .filter(
                AgentRun.agent_name == "tool_agent",
                ExecutionEvent.event_type == "tool"
            )
        )

        # Overall metrics
        total_calls = tool_events.count()

        successful_calls = tool_events.filter(
            ExecutionEvent.status == "success"
        ).count()

        failed_calls = tool_events.filter(
            ExecutionEvent.status == "error"
        ).count()

        average_latency = (
            tool_events.with_entities(
                func.avg(ExecutionEvent.latency)
            ).scalar() or 0
        )

        # Tool-wise aggregation
        tool_stats = (
            session.query(
                ExecutionEvent.event_name.label("tool_name"),
                func.count(ExecutionEvent.event_id).label("total_calls"),
                func.sum(
                    case(
                        (ExecutionEvent.status == "success", 1),
                        else_=0
                    )
                ).label("successful_calls"),
                func.sum(
                    case(
                        (ExecutionEvent.status == "error", 1),
                        else_=0
                    )
                ).label("failed_calls"),
                func.avg(ExecutionEvent.latency).label("average_latency")
            )
            .join(
                AgentRun,
                ExecutionEvent.trace_id == AgentRun.trace_id
            )
            .filter(
                AgentRun.agent_name == "tool_agent",
                ExecutionEvent.event_type == "tool"
            )
            .group_by(ExecutionEvent.event_name)
            .order_by(func.count(ExecutionEvent.event_id).desc())
            .all()
        )

        tools = []

        for tool in tool_stats:

            total = tool.total_calls or 0
            successful = tool.successful_calls or 0
            failed = tool.failed_calls or 0

            tools.append({
                "tool_name": tool.tool_name,
                "total_calls": total,
                "successful_calls": successful,
                "failed_calls": failed,
                "success_rate": round(
                    successful / total * 100, 2
                ) if total else 0,
                "failure_rate": round(
                    failed / total * 100, 2
                ) if total else 0,
                "average_latency": round(
                    tool.average_latency or 0, 4
                )
            })

        return {
            "total_tool_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": round(
                successful_calls / total_calls * 100, 2
            ) if total_calls else 0,
            "failure_rate": round(
                failed_calls / total_calls * 100, 2
            ) if total_calls else 0,
            "average_latency": round(average_latency, 4),
            "tools": tools
        }

    finally:
        session.close()


# --------------------------------------------------
# RECENT TOOL EXECUTION EVENTS
# --------------------------------------------------

@router.get("/tool-events")
def get_tool_events(
    tool_name: str | None = None,
    status: str | None = None,
    limit: int = Query(default=20, ge=1, le=200),
    offset: int = Query(default=0, ge=0)
):

    session = SessionLocal()

    try:
        query = (
            session.query(ExecutionEvent, AgentRun)
            .join(
                AgentRun,
                ExecutionEvent.trace_id == AgentRun.trace_id
            )
            .filter(
                AgentRun.agent_name == "tool_agent",
                ExecutionEvent.event_type == "tool"
            )
        )

        if tool_name:
            query = query.filter(
                ExecutionEvent.event_name == tool_name
            )

        if status:
            query = query.filter(
                ExecutionEvent.status == status
            )

        total = query.count()

        events = (
            query
            .order_by(ExecutionEvent.timestamp.desc())
            .offset(offset)
            .limit(limit)
            .all()
        )

        return {
            "total": total,
            "limit": limit,
            "offset": offset,
            "events": [
                {
                    "event_id": event.event_id,
                    "trace_id": event.trace_id,
                    "timestamp": event.timestamp,
                    "tool_name": event.event_name,
                    "status": event.status,
                    "latency": event.latency,
                    "input_data": parse_json_field(
                        event.input_data
                    ),
                    "output_data": parse_json_field(
                        event.output_data
                    ),
                    "error_type": event.error_type,
                    "error_message": event.error_message,
                    "injected_fault": event.injected_fault,
                    "user_input": run.user_input
                }
                for event, run in events
            ]
        }

    finally:
        session.close()