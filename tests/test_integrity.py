from pathlib import Path

from app.db.repository import Database
from app.services.integrity import batch_move_to_quarantine, run_integrity_check


def _setup_db(tmp_path: Path) -> Database:
    db_path = tmp_path / "test.db"
    schema_path = Path(__file__).resolve().parents[1] / "app" / "db" / "schema.sql"
    db = Database(db_path, schema_path)
    db.init_db()
    return db


def test_integrity_soft_lock_and_batch_move(tmp_path: Path) -> None:
    tiles_dir = tmp_path / "tiles"
    quarantine_dir = tmp_path / "quarantine"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    db = _setup_db(tmp_path)

    # in db and present
    good_file = tiles_dir / "tile_good.bin"
    good_file.write_bytes(b"abc")
    db.upsert_tile("tile_good", str(good_file))

    # in db but missing on disk
    missing_file = tiles_dir / "tile_missing.bin"
    db.upsert_tile("tile_missing", str(missing_file))

    # in db but zero byte
    zero_file = tiles_dir / "tile_zero.bin"
    zero_file.write_bytes(b"")
    db.upsert_tile("tile_zero", str(zero_file))

    # on disk but not in db
    orphan_file = tiles_dir / "tile_orphan.bin"
    orphan_file.write_bytes(b"orphan")

    report = run_integrity_check(db, tiles_dir=tiles_dir)

    assert len(report["missing_on_disk"]) == 1
    assert report["missing_on_disk"][0]["tile_id"] == "tile_missing"
    assert len(report["zero_byte_on_disk"]) == 1
    assert report["zero_byte_on_disk"][0]["tile_id"] == "tile_zero"
    assert str(orphan_file) in report["missing_in_index"]

    moved = batch_move_to_quarantine(report, quarantine_dir=quarantine_dir, limit=10)
    moved_from = {item["from"] for item in moved["moved"]}
    assert str(orphan_file) in moved_from
    assert str(zero_file) in moved_from
    assert not orphan_file.exists()
    assert not zero_file.exists()

    tiles = {row["tile_id"]: row for row in db.list_tiles()}
    assert tiles["tile_missing"]["is_soft_locked"] == 1
    assert tiles["tile_missing"]["flagged_for_fix"] == 1
    assert tiles["tile_zero"]["is_soft_locked"] == 1
