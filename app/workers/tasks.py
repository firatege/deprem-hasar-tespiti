from pathlib import Path
from typing import Any

from app.config.settings import settings
from app.db.repository import Database
from app.ml.inference import DamageInferenceModel
from app.services.cache import cache_result

_model: DamageInferenceModel | None = None


def _get_model() -> DamageInferenceModel:
    global _model
    if _model is None:
        _model = DamageInferenceModel(settings.model_path, settings.inference_config_path)
    return _model


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
        prediction = _get_model().predict(
            pre_image=Path(payload["pre_image_path"]),
            post_image=Path(payload["post_image_path"]),
            pga_value=float(payload.get("pga_value") or 0.0),
            magnitude=float(payload.get("magnitude") or 0.0),
            depth_km=float(payload.get("depth_km") or 0.0),
            acquisition_delta_days=float(payload.get("acquisition_delta_days") or 0.0),
            building_density=float(payload.get("building_density") or 0.0),
        )
        result = {
            "job_id": job_id,
            "tile_id": payload.get("tile_id"),
            "event_id": payload.get("event_id"),
            **prediction,
        }
        result_path = cache_result(settings.cache_dir, job_id, result)
        db.update_job_status(job_id, "succeeded", result_path=str(result_path))
        return result
    except Exception as exc:
        db.update_job_status(job_id, "failed", error_message=str(exc))
        raise
