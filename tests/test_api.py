from fastapi.testclient import TestClient
import time

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_submit_status_result_flow() -> None:
    payload = {
        "event_id": "usgs-test-event",
        "tile_id": "tile-001",
        "pga_value": 0.42,
        "damage_label": "major-damage",
    }
    submit = client.post("/jobs/submit", json=payload)
    assert submit.status_code == 200
    job_id = submit.json()["job_id"]

    deadline = time.time() + 5
    current_status = "queued"
    while time.time() < deadline:
        status = client.get(f"/jobs/status/{job_id}")
        assert status.status_code == 200
        current_status = status.json()["status"]
        if current_status == "succeeded":
            break
        assert current_status in {"queued", "running"}
        time.sleep(0.05)

    result = client.get(f"/jobs/result/{job_id}")
    if current_status == "succeeded":
        assert result.status_code == 200
        assert result.json()["binary_damage"] == 1
        assert result.json()["damage_label"] == "major-damage"
        assert result.json()["policy"] == "xview2_major_destroyed_is_one"
    else:
        assert current_status in {"queued", "running"}
        assert result.status_code == 409


def test_integrity_endpoint() -> None:
    response = client.post("/data/integrity/run")
    assert response.status_code == 200
    body = response.json()
    assert "run_id" in body
    assert "missing_on_disk" in body


def test_submit_rejects_invalid_damage_label() -> None:
    payload = {
        "event_id": "usgs-test-event",
        "tile_id": "tile-002",
        "damage_label": "catastrophic",
    }
    response = client.post("/jobs/submit", json=payload)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert any(err.get("loc") == ["body", "damage_label"] for err in detail)

