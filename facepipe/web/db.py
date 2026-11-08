"""SQLite persistence for the web application (stdlib only).

Tables: projects, clips, jobs, activity. Files live under <data_dir>/projects/<project_id>/.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'new',          -- new | uploaded | ready | analyzing | analyzed | processing | error
    source_filename TEXT,
    source_info TEXT,                              -- JSON probe_media result
    transcript_filename TEXT,
    transcript_info TEXT,                          -- JSON {sentences, duration_ms, speakers}
    settings TEXT NOT NULL DEFAULT '{}',           -- JSON ProjectSettings
    error TEXT
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rank INTEGER NOT NULL,
    selected INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'candidate',    -- candidate | processing | rendered | error
    moment TEXT NOT NULL,                          -- JSON 5.5 moment document
    created_at REAL NOT NULL,
    rendered_at REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,                            -- analyze | process | render
    status TEXT NOT NULL,                          -- queued | running | done | failed | cancelled | interrupted
    params TEXT NOT NULL DEFAULT '{}',
    progress TEXT NOT NULL DEFAULT '{}',           -- JSON live progress snapshot
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS activity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_project ON clips(project_id, rank);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at);
CREATE INDEX IF NOT EXISTS idx_activity_at ON activity(at);
"""


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)  # executescript manages its own transaction

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                yield self._conn
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def one(self, sql: str, params: tuple = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def execute(self, sql: str, params: tuple = ()) -> None:
        with self.tx() as c:
            c.execute(sql, params)

    # -- helpers -----------------------------------------------------------------
    def log(self, project_id: str | None, kind: str, message: str) -> None:
        self.execute("INSERT INTO activity(project_id, at, kind, message) VALUES (?,?,?,?)",
                     (project_id, time.time(), kind, message))

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def loads(s: str | None, default: Any = None) -> Any:
    if not s:
        return default if default is not None else {}
    return json.loads(s)


def dumps(o: Any) -> str:
    return json.dumps(o, separators=(",", ":"))
