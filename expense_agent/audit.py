"""Append-only audit store + human review queue (SQLite).
Every run — auto-decided or human-decided — ends as one immutable record
containing everything needed to reconstruct the decision."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_runs (
    run_id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    employee_id TEXT NOT NULL,
    vendor TEXT NOT NULL,
    amount REAL NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,            -- completed | pending_human
    final_decision TEXT,
    decided_by TEXT,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_dup
    ON audit_runs (employee_id, vendor, amount, created_at);

CREATE TABLE IF NOT EXISTS review_queue (
    thread_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    request_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',  -- pending | resolved
    resolved_by TEXT,
    resolved_decision TEXT,
    resolved_at TEXT
);
"""


@contextmanager
def _conn():
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def write_run(record: dict) -> None:
    req = record["request"]
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO audit_runs
               (run_id, request_id, employee_id, vendor, amount, created_at,
                status, final_decision, decided_by, record_json)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                record["run_id"], req["request_id"], req["employee_id"],
                req["vendor"], req["amount"], record["created_at"],
                record["status"], record.get("final_decision"),
                record.get("decided_by"), json.dumps(record, default=str),
            ),
        )


def get_run(request_id: str) -> dict | None:
    with _conn() as c:
        row = c.execute(
            "SELECT record_json FROM audit_runs WHERE request_id = ? ORDER BY created_at DESC",
            (request_id,),
        ).fetchone()
    return json.loads(row["record_json"]) if row else None


def duplicate_exists(employee_id: str, vendor: str, amount: float, exclude_request_id: str) -> bool:
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=config.DUPLICATE_WINDOW_HOURS)
    ).isoformat()
    with _conn() as c:
        row = c.execute(
            """SELECT 1 FROM audit_runs
               WHERE employee_id = ? AND vendor = ? AND amount = ?
                 AND request_id != ? AND created_at >= ?
               LIMIT 1""",
            (employee_id, vendor, amount, exclude_request_id, cutoff),
        ).fetchone()
    return row is not None


# --- Human review queue ---

def add_pending(thread_id: str, run_id: str, request: dict, reason: str) -> None:
    with _conn() as c:
        c.execute(
            """INSERT OR REPLACE INTO review_queue
               (thread_id, run_id, request_json, reason, created_at, status)
               VALUES (?,?,?,?,?, 'pending')""",
            (thread_id, run_id, json.dumps(request, default=str), reason,
             datetime.now(timezone.utc).isoformat()),
        )


def list_pending() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM review_queue WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
    return [
        {**dict(r), "request": json.loads(r["request_json"])} for r in rows
    ]


def resolve_pending(thread_id: str, decision: str, reviewer: str) -> None:
    with _conn() as c:
        c.execute(
            """UPDATE review_queue
               SET status='resolved', resolved_by=?, resolved_decision=?, resolved_at=?
               WHERE thread_id=?""",
            (reviewer, decision, datetime.now(timezone.utc).isoformat(), thread_id),
        )
