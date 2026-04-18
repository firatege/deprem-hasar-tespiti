import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def cache_result(cache_dir: Path, job_id: str, payload: dict[str, Any]) -> Path:
    ensure_dir(cache_dir)
    out_path = cache_dir / f"{job_id}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_cached_result(result_path: Path) -> dict[str, Any]:
    return json.loads(result_path.read_text(encoding="utf-8"))
