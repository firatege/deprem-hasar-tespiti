import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class Settings:
    base_dir: Path = Path(os.getenv("APP_BASE_DIR", Path(__file__).resolve().parents[2]))
    sqlite_path: Path = _env_path("SQLITE_PATH", base_dir / "data" / "index.db")
    schema_path: Path = _env_path("SCHEMA_PATH", base_dir / "app" / "db" / "schema.sql")
    storage_root: Path = _env_path("STORAGE_ROOT", base_dir / "storage")
    tiles_dir: Path = _env_path("TILES_DIR", storage_root / "tiles")
    cache_dir: Path = _env_path("CACHE_DIR", storage_root / "cache")
    quarantine_dir: Path = _env_path("QUARANTINE_DIR", storage_root / "quarantine")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    rq_queue_name: str = os.getenv("RQ_QUEUE_NAME", "tile-jobs")
    sync_fallback_when_queue_unavailable: bool = _env_bool(
        "SYNC_FALLBACK_WHEN_QUEUE_UNAVAILABLE", True
    )
    batch_quarantine_limit: int = _env_int("BATCH_QUARANTINE_LIMIT", 50)


settings = Settings()