import time
from typing import Any

from app.config.settings import settings
from app.db.repository import Database
from app.services.cache import cache_result
from app.services.classification import normalize_damage_label, xview2_damage_to_binary


def process_tile_job(job_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    db = Database(settings.sqlite_path, settings.schema_path)
    db.init_db()

    job = db.get_job(job_id)
    if not job:
        raise ValueError(f"Job not found: {job_id}")
    if job["status"] == "canceled":
        return {"job_id": job_id, "status": "canceled"}

    db.update_job_status(job_id, "running")

    try:
        # Placeholder for heavy geospatial preprocessing and model inference.
        time.sleep(0.2)
        damage_label = payload.get("damage_label", "major-damage")
        normalized_damage_label = normalize_damage_label(damage_label)
        binary_damage = xview2_damage_to_binary(damage_label)
        pga_value = payload.get("pga_value")

        result = {
            "job_id": job_id,
            "tile_id": payload.get("tile_id"),
            "event_id": payload.get("event_id"),
            "damage_label": normalized_damage_label,
            "binary_damage": binary_damage,
            "pga_value": pga_value,
            "policy": "xview2_major_destroyed_is_one",
        }
        result_path = cache_result(settings.cache_dir, job_id, result)
        db.update_job_status(job_id, "succeeded", result_path=str(result_path))
        return result
    except Exception as exc:
        db.update_job_status(job_id, "failed", error_message=str(exc))
        raise
