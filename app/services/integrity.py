import shutil
import uuid
from pathlib import Path
from typing import Any

from app.db.repository import Database


def _is_zero_byte(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size == 0


def run_integrity_check(
    db: Database,
    *,
    tiles_dir: Path,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    tiles_dir.mkdir(parents=True, exist_ok=True)

    db_tiles = db.list_tiles()
    db_by_tile_id = {row["tile_id"]: row for row in db_tiles}

    missing_on_disk: list[dict[str, str]] = []
    zero_byte_on_disk: list[dict[str, str]] = []
    for tile_id, row in db_by_tile_id.items():
        file_path = Path(row["file_path"])
        if not file_path.exists():
            db.soft_lock_tile(tile_id, "missing_on_disk")
            db.add_integrity_event(
                run_id=run_id,
                tile_id=tile_id,
                file_path=str(file_path),
                issue_type="missing_on_disk",
                action="soft_lock",
            )
            missing_on_disk.append({"tile_id": tile_id, "file_path": str(file_path)})
        elif _is_zero_byte(file_path):
            db.soft_lock_tile(tile_id, "zero_byte")
            db.add_integrity_event(
                run_id=run_id,
                tile_id=tile_id,
                file_path=str(file_path),
                issue_type="zero_byte",
                action="soft_lock",
            )
            zero_byte_on_disk.append({"tile_id": tile_id, "file_path": str(file_path)})

    disk_files = [p for p in tiles_dir.rglob("*") if p.is_file()]
    known_paths = {str(Path(row["file_path"])) for row in db_tiles}
    missing_in_index: list[str] = []
    for file_path in disk_files:
        if str(file_path) not in known_paths:
            db.add_integrity_event(
                run_id=run_id,
                tile_id=None,
                file_path=str(file_path),
                issue_type="missing_in_index",
                action="report",
                details="candidate_for_batch_quarantine",
            )
            missing_in_index.append(str(file_path))

    return {
        "run_id": run_id,
        "missing_on_disk": missing_on_disk,
        "zero_byte_on_disk": zero_byte_on_disk,
        "missing_in_index": missing_in_index,
    }


def batch_move_to_quarantine(
    report: dict[str, Any],
    *,
    quarantine_dir: Path,
    limit: int = 50,
) -> dict[str, Any]:
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    moved: list[dict[str, str]] = []

    candidates: list[str] = []
    candidates.extend(report.get("missing_in_index", []))
    candidates.extend([item["file_path"] for item in report.get("zero_byte_on_disk", [])])

    for source_str in candidates[:limit]:
        source = Path(source_str)
        if not source.exists() or not source.is_file():
            continue
        target = quarantine_dir / source.name
        if target.exists():
            target = quarantine_dir / f"{source.stem}_{uuid.uuid4().hex[:8]}{source.suffix}"
        shutil.move(str(source), str(target))
        moved.append({"from": str(source), "to": str(target)})

    return {"moved": moved, "moved_count": len(moved)}
