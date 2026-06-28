"""
tests/conftest.py — Shared pytest fixtures for the detection system test suite.

All fixtures are independent and stateless — no shared mutable state between tests.
"""
import os
import queue
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torchvision import models

# Ensure the project root is importable as the top-level package namespace.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ── Image fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image_bgr():
    """224×224 BGR uint8 ndarray built from a fixed seed — no real image needed."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)


# ── Model fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def model_and_classes():
    """
    MobileNetV3-large with the same custom head used in production,
    initialised with random weights.  No checkpoint file required.
    """
    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, 2),
    )
    m.eval()
    return m, ["no_trash", "trash"], torch.device("cpu")


# ── Config fixture ────────────────────────────────────────────────────────────

@pytest.fixture
def loaded_config():
    """Parse config.yaml from the project root and return the dict."""
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Flask fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def flask_client():
    """
    Flask test client.  _model is None at import time so read-only routes work
    without any model weights present.
    """
    from dashboard.app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# ── Misc fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path):
    """Temporary directory auto-cleaned after each test."""
    return tmp_path


@pytest.fixture
def event_queue():
    """Bounded queue that mirrors what CameraPipeline receives from the app."""
    return queue.Queue(maxsize=50)
