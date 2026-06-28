"""
tests/test_config.py — Tests for config.yaml (the project's central configuration).

config.yaml is loaded by main.py and dashboard/app.py to populate default
pipeline parameters.  These tests validate structure, value ranges, and that
the referenced model file exists on disk.

Validation of runtime settings (Flask /api/settings) is tested here too,
as that is the canonical enforcement path for parameter bounds.
"""
import os

import pytest
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

REQUIRED_KEYS = {
    "model_path",
    "threshold",
    "video_path",
    "bg_history",
    "bg_var_threshold",
    "min_area",
    "stationary_min_frames",
    "stationary_distance",
    "max_missed",
    "process_every_n",
}


@pytest.fixture
def config():
    """Parse config.yaml and return the dict."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Structure
# ─────────────────────────────────────────────────────────────────────────────

def test_config_loads(config):
    """config.yaml parses without error and returns a non-empty dict."""
    assert isinstance(config, dict)
    assert len(config) > 0


def test_required_keys(config):
    """config.yaml contains all required top-level keys."""
    for key in REQUIRED_KEYS:
        assert key in config, f"Required key '{key}' is missing from config.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# Value ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_confidence_threshold_range(config):
    """Confidence threshold is a float in (0, 1]."""
    threshold = config["threshold"]
    assert isinstance(threshold, (int, float))
    assert 0.0 < threshold <= 1.0, (
        f"threshold={threshold} not in (0, 1]"
    )


def test_bg_history_positive(config):
    """bg_history must be a positive integer."""
    assert isinstance(config["bg_history"], int)
    assert config["bg_history"] > 0


def test_stationary_min_frames_positive(config):
    """stationary_min_frames must be >= 1."""
    val = config["stationary_min_frames"]
    assert isinstance(val, int)
    assert val >= 1


def test_stationary_distance_positive(config):
    """stationary_distance must be > 0 (zero would make every track stationary)."""
    val = config["stationary_distance"]
    assert isinstance(val, (int, float))
    assert val > 0


def test_process_every_n_valid(config):
    """process_every_n must be >= 1 (1 = every frame; higher = subsampling)."""
    val = config["process_every_n"]
    assert isinstance(val, int)
    assert val >= 1


def test_model_path_exists(config):
    """The model file referenced by model_path exists on disk."""
    model_abs = os.path.join(PROJECT_ROOT, config["model_path"])
    assert os.path.exists(model_abs), (
        f"Model checkpoint not found: {model_abs}\n"
        "Download or train best_model.pth before running this test."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runtime validation (Flask settings schema)
# ─────────────────────────────────────────────────────────────────────────────

def test_invalid_threshold_raises():
    """
    The Flask /api/settings endpoint rejects thresholds outside [0.01, 1.0].
    This is the canonical validation path — config.yaml itself has no schema.
    """
    import json
    from dashboard.app import app

    app.config["TESTING"] = True
    with app.test_client() as client:
        for bad_value in [-1, 0, 999, "not_a_number"]:
            response = client.post(
                "/api/settings",
                data=json.dumps({"threshold": bad_value}),
                content_type="application/json",
            )
            assert response.status_code == 400, (
                f"Expected 400 for threshold={bad_value!r}, "
                f"got {response.status_code}"
            )
            data = response.get_json()
            assert "errors" in data
