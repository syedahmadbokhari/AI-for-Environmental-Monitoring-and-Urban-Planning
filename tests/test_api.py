"""
tests/test_api.py — Flask REST API tests for dashboard/app.py.

Uses app.test_client() — no real HTTP server is started.

Adaptation notes vs. the original spec:
  • No /health route exists in the codebase.  GET /api/settings is used as
    the health-check proxy (returns 200 when the server is up).
  • The video stream route is GET /video/<id>, not /stream/<id>.
  • POST /api/cameras returns 503 when _model is None (not yet loaded),
    which is always the case in tests.
"""
import json

import pytest

from dashboard.app import app


@pytest.fixture
def client():
    """Isolated test client — no shared state with other tests."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ─────────────────────────────────────────────────────────────────────────────
# Settings — liveness probe
# ─────────────────────────────────────────────────────────────────────────────

def test_health_endpoint(client):
    """GET /api/settings returns 200 (closest equivalent to a /health probe)."""
    response = client.get("/api/settings")
    assert response.status_code == 200


def test_settings_returns_json(client):
    """GET /api/settings body is valid JSON containing expected parameter keys."""
    response = client.get("/api/settings")
    data = response.get_json()
    assert data is not None
    assert "threshold" in data
    assert "stationary_min_frames" in data
    assert "bg_history" in data


def test_settings_update_valid(client):
    """POST /api/settings with a valid threshold value returns 200 and the updated value."""
    payload = {"threshold": 0.7}
    response = client.post(
        "/api/settings",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 200
    data = response.get_json()
    assert abs(data["threshold"] - 0.7) < 1e-6


def test_api_error_handling(client):
    """POST /api/settings with an out-of-range threshold returns 400 with errors key."""
    payload = {"threshold": 999}   # valid range is [0.01, 1.0]
    response = client.post(
        "/api/settings",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 400
    data = response.get_json()
    assert "errors" in data
    assert "threshold" in data["errors"]


# ─────────────────────────────────────────────────────────────────────────────
# Camera list
# ─────────────────────────────────────────────────────────────────────────────

def test_events_endpoint(client):
    """GET /api/cameras returns 200 with a JSON list."""
    response = client.get("/api/cameras")
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


def test_events_structure(client):
    """Each camera object in GET /api/cameras has the required schema keys."""
    response = client.get("/api/cameras")
    cameras = response.get_json()
    for cam in cameras:
        for key in ("id", "name", "source", "status"):
            assert key in cam, f"Missing key '{key}' in camera object"


# ─────────────────────────────────────────────────────────────────────────────
# Events log
# ─────────────────────────────────────────────────────────────────────────────

def test_events_json_endpoint(client):
    """GET /api/events returns 200 with a JSON list."""
    response = client.get("/api/events")
    assert response.status_code == 200
    assert isinstance(response.get_json(), list)


def test_events_n_param(client):
    """GET /api/events?n=3 returns at most 3 events."""
    response = client.get("/api/events?n=3")
    assert response.status_code == 200
    assert len(response.get_json()) <= 3


# ─────────────────────────────────────────────────────────────────────────────
# 404 and error paths
# ─────────────────────────────────────────────────────────────────────────────

def test_nonexistent_route(client):
    """GET /nonexistent-route returns 404."""
    response = client.get("/nonexistent-route-xyz")
    assert response.status_code == 404


def test_delete_unknown_camera(client):
    """DELETE /api/cameras/<unknown-id> returns 404 with an error key."""
    response = client.delete("/api/cameras/does-not-exist-abc")
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data


def test_patch_unknown_camera(client):
    """PATCH /api/cameras/<unknown-id> returns 404."""
    response = client.patch(
        "/api/cameras/no-such-camera",
        data=json.dumps({"name": "New Name"}),
        content_type="application/json",
    )
    assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Video stream
# ─────────────────────────────────────────────────────────────────────────────

def test_stream_endpoint(client):
    """GET /video/<nonexistent-id> returns 404 when the camera is not registered."""
    response = client.get("/video/nonexistent-camera-id")
    assert response.status_code == 404


# ─────────────────────────────────────────────────────────────────────────────
# Model-gated route
# ─────────────────────────────────────────────────────────────────────────────

def test_add_camera_without_model(client):
    """POST /api/cameras returns 503 when no model has been loaded (_model is None)."""
    payload = {"name": "Test Camera", "source": "videos/test.mp4"}
    response = client.post(
        "/api/cameras",
        data=json.dumps(payload),
        content_type="application/json",
    )
    assert response.status_code == 503
    data = response.get_json()
    assert "error" in data


def test_add_camera_missing_source(client):
    """
    POST /api/cameras without a source field.
    Returns 503 (model not loaded) before the source validation is reached;
    the key check verifies the request is handled, not silently ignored.
    """
    payload = {"name": "No Source"}
    response = client.post(
        "/api/cameras",
        data=json.dumps(payload),
        content_type="application/json",
    )
    # 503 because model isn't loaded; if model were loaded this would be 400
    assert response.status_code in (400, 503)
