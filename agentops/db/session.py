"""Engine, session factory, schema creation and safe migration."""

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from agentops.config import DB_PATH
from agentops.db.models import Base

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(bind=engine)


def migrate_schema(bind=None) -> list:
    """Add any model column that is missing from an existing table.

    Works for every table, so an old v1 database can be copied into v2 and
    upgraded without losing rows. Returns the list of added columns.
    """
    bind = bind or engine
    inspector = inspect(bind)
    added = []

    with bind.begin() as connection:
        for table in Base.metadata.sorted_tables:
            if not inspector.has_table(table.name):
                continue

            existing = {c["name"] for c in inspector.get_columns(table.name)}

            for column in table.columns:
                if column.name in existing:
                    continue
                connection.execute(text(
                    f"ALTER TABLE {table.name} "
                    f"ADD COLUMN {column.name} {column.type.compile(bind.dialect)}"
                ))
                added.append(f"{table.name}.{column.name}")

    return added


def init_db() -> list:
    Base.metadata.create_all(bind=engine)
    return migrate_schema()


def migrate_jsonl() -> int:
    """Import old v1 JSONL traces into the legacy `traces` table."""
    import json

    from agentops.config import LEGACY_TRACE_FILE
    from agentops.db.models import Trace

    if not LEGACY_TRACE_FILE.exists():
        print("No existing JSONL file found.")
        return 0

    migrated = 0
    with SessionLocal() as session:
        with open(LEGACY_TRACE_FILE, "r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                data = json.loads(line)
                if session.get(Trace, data["trace_id"]):
                    continue
                data.setdefault("source", "legacy")
                session.add(Trace(**data))
                migrated += 1
        session.commit()

    print(f"{migrated} old traces migrated.")
    return migrated