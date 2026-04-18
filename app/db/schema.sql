PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    result_path TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tiles (
    tile_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    source_version TEXT,
    pga_value REAL,
    binary_label INTEGER,
    is_soft_locked INTEGER NOT NULL DEFAULT 0,
    flagged_for_fix INTEGER NOT NULL DEFAULT 0,
    lock_reason TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS integrity_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    tile_id TEXT,
    file_path TEXT,
    issue_type TEXT NOT NULL,
    action TEXT NOT NULL,
    details TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_tiles_soft_locked ON tiles(is_soft_locked);
CREATE INDEX IF NOT EXISTS idx_integrity_run_id ON integrity_events(run_id);
