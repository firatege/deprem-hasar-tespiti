from fastapi import APIRouter

from app.config.settings import settings
from app.db.repository import Database
from app.services.integrity import batch_move_to_quarantine, run_integrity_check

router = APIRouter(prefix="/data", tags=["data"])


db = Database(settings.sqlite_path, settings.schema_path)
db.init_db()


@router.post("/integrity/run")
def run_integrity(batch_move: bool = False) -> dict:
    report = run_integrity_check(db, tiles_dir=settings.tiles_dir)
    if not batch_move:
        return report

    moved = batch_move_to_quarantine(
        report,
        quarantine_dir=settings.quarantine_dir,
        limit=settings.batch_quarantine_limit,
    )
    return {"report": report, "batch": moved}
