import os
import tempfile

# Must happen before `agentops` is imported: use a throw-away database.
_TMP = tempfile.mkdtemp(prefix="agentops-test-")
os.environ["AGENTOPS_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ.setdefault("NVIDIA_API_KEY", "test-key")

import pytest  # noqa: E402

from agentops.db import AgentRun, ExecutionEvent, SessionLocal, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def clean_db():
    init_db()
    session = SessionLocal()
    session.query(ExecutionEvent).delete()
    session.query(AgentRun).delete()
    session.commit()
    session.close()
    yield


@pytest.fixture
def db():
    session = SessionLocal()
    yield session
    session.close()


def events_of(session, trace_id):
    return (
        session.query(ExecutionEvent)
        .filter(ExecutionEvent.trace_id == trace_id)
        .order_by(ExecutionEvent.sequence_number)
        .all()
    )