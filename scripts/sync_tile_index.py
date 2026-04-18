import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.db.repository import Database


def build_stable_tile_id(*, tiles_dir: Path, file_path: Path, used_ids: set[str]) -> str:
    rel_posix = file_path.relative_to(tiles_dir).as_posix()
    base_id = f"fs::{rel_posix.replace('/', '__')}"
    if base_id not in used_ids:
        return base_id

    digest = hashlib.sha1(rel_posix.encode("utf-8")).hexdigest()[:10]
    candidate = f"{base_id}__{digest}"
    if candidate not in used_ids:
        return candidate

    suffix = 1
    while True:
        candidate = f"{base_id}__{digest}_{suffix}"
        if candidate not in used_ids:
            return candidate
        suffix += 1


def scan_files(tiles_dir: Path, extensions: set[str]) -> list[Path]:
    files = [p for p in tiles_dir.rglob("*") if p.is_file()]
    if not extensions:
        return sorted(files)
    return sorted([p for p in files if p.suffix.lower() in extensions])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync disk files under tiles_dir into tiles index")
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=settings.tiles_dir,
        help="Root directory to scan (defaults to settings.tiles_dir)",
    )
    parser.add_argument(
        "--include-ext",
        default=".tif,.json",
        help="Comma-separated file extensions to include (default: .tif,.json)",
    )
    parser.add_argument(
        "--source-version",
        default="xview2_geotiff",
        help="source_version stored in tiles rows for newly indexed files",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Persist missing files into DB. Without this flag, script runs as dry-run.",
    )
    args = parser.parse_args()

    tiles_dir = args.tiles_dir.resolve()
    tiles_dir.mkdir(parents=True, exist_ok=True)

    extensions = {
        ext.strip().lower() if ext.strip().startswith(".") else f".{ext.strip().lower()}"
        for ext in args.include_ext.split(",")
        if ext.strip()
    }

    db = Database(settings.sqlite_path, settings.schema_path)
    db.init_db()
    rows = db.list_tiles()

    known_paths = {str(Path(row["file_path"]).resolve()): row for row in rows}
    used_ids = {row["tile_id"] for row in rows}

    disk_files = scan_files(tiles_dir, extensions)
    missing_paths = [p for p in disk_files if str(p.resolve()) not in known_paths]

    added = 0
    for file_path in missing_paths:
        tile_id = build_stable_tile_id(tiles_dir=tiles_dir, file_path=file_path, used_ids=used_ids)
        used_ids.add(tile_id)
        if args.apply:
            db.upsert_tile(
                tile_id,
                str(file_path.resolve()),
                source_version=args.source_version,
            )
        added += 1

    summary = {
        "mode": "apply" if args.apply else "dry-run",
        "tiles_dir": str(tiles_dir),
        "include_ext": sorted(extensions),
        "disk_files_scanned": len(disk_files),
        "already_indexed": len(disk_files) - len(missing_paths),
        "would_add_or_added": added,
        "db_rows_before": len(rows),
        "db_rows_after_estimate": len(rows) + added,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

