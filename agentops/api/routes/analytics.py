from fastapi import APIRouter
from sqlalchemy import func

from agentops.db import SessionLocal, Trace

router = APIRouter()


@router.get("/traces")
def get_traces():
    session = SessionLocal()

    try:
        traces = session.query(Trace).order_by(
            Trace.timestamp.desc()
        ).all()

        return [
            {
                "trace_id": t.trace_id,
                "timestamp": t.timestamp,
                "input": t.input,
                "output": t.output,
                "model": t.model,
                "agent_name": t.agent_name,
                "latency": t.latency,
                "tokens": t.tokens,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "status": t.status,
                "error_type": t.error_type
            }
            for t in traces
        ]

    finally:
        session.close()


@router.get("/analytics")
def get_analytics():
    session = SessionLocal()

    try:
        total = session.query(Trace).count()

        successful = session.query(Trace).filter(
            Trace.status == "success"
        ).count()

        failed = session.query(Trace).filter(
            Trace.status == "error"
        ).count()

        avg_latency = session.query(
            func.avg(Trace.latency)
        ).scalar() or 0

        total_tokens = session.query(
            func.sum(Trace.tokens)
        ).scalar() or 0

        return {
            "total_executions": total,
            "successful": successful,
            "failed": failed,
            "error_rate": round((failed / total) * 100, 2) if total else 0,
            "average_latency": round(avg_latency, 3),
            "total_tokens": total_tokens
        }

    finally:
        session.close()