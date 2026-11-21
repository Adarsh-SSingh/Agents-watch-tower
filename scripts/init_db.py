"""Create tables, add missing columns and import old JSONL traces."""

from agentops.db import init_db, migrate_jsonl

if __name__ == "__main__":
    init_db()
    migrate_jsonl()
    print("Database ready.")

    