import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config.settings import settings
from app.db.repository import Database
from app.services.integrity import batch_move_to_quarantine, run_integrity_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tile index/disk integrity checks")
    parser.add_argument(
        "--batch-move",
        action="store_true",
        help="Move orphan files to quarantine after soft-lock/report phase",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=settings.batch_quarantine_limit,
        help="Maximum files to move per batch run",
    )
    args = parser.parse_args()

    db = Database(settings.sqlite_path, settings.schema_path)
    db.init_db()

    report = run_integrity_check(db, tiles_dir=settings.tiles_dir)
    output = {"report": report}

    if args.batch_move:
        output["batch"] = batch_move_to_quarantine(
            report,
            quarantine_dir=settings.quarantine_dir,
            limit=args.limit,
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()