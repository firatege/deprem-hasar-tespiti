import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


VALID_STATUSES = {"queued", "running", "succeeded", "failed", "canceled"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Database:
    db_path: Path
    schema_path: Path

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            schema_sql = self.schema_path.read_text(encoding="utf-8")
            conn.executescript(schema_sql)
            conn.commit()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def create_job(self, job_id: str, payload: dict[str, Any], status: str = "queued") -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs(id, status, payload_json, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (job_id, status, json.dumps(payload), now, now),
            )
            conn.commit()

    def update_job_status(
        self,
        job_id: str,
        status: str,
        *,
        result_path: str | None = None,
        error_message: str | None = None,
    ) -> None:
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE jobs
                SET status = ?, result_path = COALESCE(?, result_path),
                    error_message = COALESCE(?, error_message), updated_at = ?
                WHERE id = ?
                """,
                (status, result_path, error_message, now, job_id),
            )
            conn.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if not row:
                return None
            payload = json.loads(row["payload_json"])
            return {
                "id": row["id"],
                "status": row["status"],
                "payload": payload,
                "result_path": row["result_path"],
                "error_message": row["error_message"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

    def upsert_tile(
        self,
        tile_id: str,
        file_path: str,
        *,
        source_version: str | None = None,
        pga_value: float | None = None,
        binary_label: int | None = None,
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tiles(tile_id, file_path, source_version, pga_value, binary_label, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tile_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    source_version = excluded.source_version,
                    pga_value = excluded.pga_value,
                    binary_label = excluded.binary_label,
                    updated_at = excluded.updated_at
                """,
                (tile_id, file_path, source_version, pga_value, binary_label, now),
            )
            conn.commit()

    def list_tiles(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM tiles").fetchall()
            return [dict(row) for row in rows]

    def soft_lock_tile(self, tile_id: str, reason: str) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE tiles
                SET is_soft_locked = 1,
                    flagged_for_fix = 1,
                    lock_reason = ?,
                    updated_at = ?
                WHERE tile_id = ?
                """,
                (reason, now, tile_id),
            )
            conn.commit()

    def add_integrity_event(
        self,
        *,
        run_id: str,
        tile_id: str | None,
        file_path: str | None,
        issue_type: str,
        action: str,
        details: str = "",
    ) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO integrity_events(
                    run_id, tile_id, file_path, issue_type, action, details, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, tile_id, file_path, issue_type, action, details, now),
            )
            conn.commit()
