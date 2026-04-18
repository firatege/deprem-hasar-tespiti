import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.config.settings import settings
from app.db.repository import Database
from app.queue import get_queue
from app.services.cache import load_cached_result
from app.services.classification import normalize_damage_label
from app.workers.tasks import process_tile_job

router = APIRouter(prefix="/jobs", tags=["jobs"])


db = Database(settings.sqlite_path, settings.schema_path)
db.init_db()


class SubmitJobRequest(BaseModel):
    event_id: str = Field(..., min_length=1)
    tile_id: str = Field(..., min_length=1)
    pga_value: float | None = None
    damage_label: str = "major-damage"

    @field_validator("damage_label")
    @classmethod
    def validate_damage_label(cls, value: str) -> str:
        return normalize_damage_label(value)


@router.post("/submit")
def submit_job(payload: SubmitJobRequest) -> dict[str, Any]:
    job_id = str(uuid.uuid4())
    payload_dict = payload.model_dump()
    db.create_job(job_id, payload_dict, status="queued")

    try:
        queue = get_queue()
        queue.enqueue(process_tile_job, job_id, payload_dict, job_id=job_id)
    except Exception as exc:
        if not settings.sync_fallback_when_queue_unavailable:
            raise HTTPException(status_code=503, detail=f"Queue unavailable: {exc}") from exc
        process_tile_job(job_id, payload_dict)

    job = db.get_job(job_id)
    return {"job_id": job_id, "status": job["status"] if job else "queued"}


@router.get("/status/{job_id}")
def get_job_status(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job_id,
        "status": job["status"],
        "error_message": job["error_message"],
        "updated_at": job["updated_at"],
    }


@router.get("/result/{job_id}")
def get_job_result(job_id: str) -> dict[str, Any]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] != "succeeded":
        raise HTTPException(status_code=409, detail=f"Job status is {job['status']}")
    if not job["result_path"]:
        raise HTTPException(status_code=500, detail="Result path missing")

    result = load_cached_result(Path(job["result_path"]))
    return result


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, str]:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] in {"succeeded", "failed", "canceled"}:
        return {"job_id": job_id, "status": job["status"]}

    db.update_job_status(job_id, "canceled")
    return {"job_id": job_id, "status": "canceled"}
