"""
tests/test_classifier.py — Unit tests for core/classifier.py.

Tests use random-weight models built inline (torch.randn equivalent).
No checkpoint file is required for any test in this module.
"""
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import patch
from torchvision import models

from core.classifier import _preprocess_cv2, predict, draw_label, load_model


CLASS_NAMES = ["no_trash", "trash"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(num_classes: int = 2) -> nn.Module:
    """Construct the same MobileNetV3 head used in production."""
    m = models.mobilenet_v3_large(weights=None)
    in_features = m.classifier[0].in_features
    m.classifier = nn.Sequential(
        nn.Linear(in_features, 1280),
        nn.Hardswish(),
        nn.Dropout(p=0.2),
        nn.Linear(1280, num_classes),
    )
    return m.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

def test_model_loads():
    """MobileNetV3 with custom binary head initialises without raising."""
    model = _build_model()
    assert isinstance(model, nn.Module)
    # Verify the custom head shape
    assert isinstance(model.classifier[-1], nn.Linear)
    assert model.classifier[-1].out_features == 2


def test_model_loads_from_checkpoint():
    """load_model() processes a mocked checkpoint dict and returns (model, class_names)."""
    real_model = _build_model()
    fake_checkpoint = {
        "class_names": CLASS_NAMES,
        "model_state": real_model.state_dict(),
        "epoch": 5,
        "val_acc": 0.75,
    }
    with patch("torch.load", return_value=fake_checkpoint):
        loaded, names = load_model("dummy.pth", torch.device("cpu"))
    assert names == CLASS_NAMES
    assert isinstance(loaded, nn.Module)


def test_output_shape():
    """Single 224×224 input produces logits of shape (1, 2)."""
    model = _build_model()
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 2)


@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_batch_sizes(batch_size):
    """Model handles batch sizes 1, 4, and 8 with correct output shape."""
    model = _build_model()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (batch_size, 2), (
        f"Expected ({batch_size}, 2), got {tuple(out.shape)}"
    )


def test_model_device():
    """Model runs on CPU; also on CUDA when a GPU is available."""
    model = _build_model()
    for device_str in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
        device = torch.device(device_str)
        model.to(device)
        x = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)
        model.to("cpu")  # reset for next iteration


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def test_preprocess_output_shape(dummy_image_bgr):
    """_preprocess_cv2 converts any BGR image to a (1, 3, 224, 224) float32 tensor."""
    tensor = _preprocess_cv2(dummy_image_bgr)
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype == torch.float32


def test_preprocess_normalised(dummy_image_bgr):
    """Preprocessed values are centred near zero, not in the raw [0, 255] range."""
    tensor = _preprocess_cv2(dummy_image_bgr)
    # After ImageNet normalisation the range is roughly [-2.5, 2.8]
    assert tensor.max().item() < 10.0
    assert tensor.min().item() > -10.0
    # At least some values are negative (mean-centred)
    assert tensor.min().item() < 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def test_model_inference(dummy_image_bgr):
    """predict() returns a (label, confidence, probs_dict) triple with valid types."""
    model = _build_model()
    label, confidence, probs = predict(
        dummy_image_bgr, model, CLASS_NAMES, torch.device("cpu"), threshold=0.5
    )
    assert label in CLASS_NAMES
    assert isinstance(confidence, float)
    assert isinstance(probs, dict)
    assert set(probs.keys()) == set(CLASS_NAMES)


def test_output_range(dummy_image_bgr):
    """Softmax probabilities are in [0, 1] and sum to 1."""
    model = _build_model()
    _, _, probs = predict(
        dummy_image_bgr, model, CLASS_NAMES, torch.device("cpu"), threshold=0.5
    )
    for cls, p in probs.items():
        assert 0.0 <= p <= 1.0, f"p({cls}) = {p} not in [0, 1]"
    assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_inference_deterministic(dummy_image_bgr):
    """Same input in eval mode always produces the same output."""
    model = _build_model()
    device = torch.device("cpu")
    label1, conf1, probs1 = predict(dummy_image_bgr, model, CLASS_NAMES, device, 0.5)
    label2, conf2, probs2 = predict(dummy_image_bgr, model, CLASS_NAMES, device, 0.5)
    assert label1 == label2
    assert abs(conf1 - conf2) < 1e-6
    for cls in probs1:
        assert abs(probs1[cls] - probs2[cls]) < 1e-6


def test_threshold_controls_label():
    """threshold=0.0 always predicts 'trash'; threshold=1.0 always predicts 'no_trash'."""
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    model = _build_model()
    device = torch.device("cpu")

    label_low, _, _ = predict(img, model, CLASS_NAMES, device, threshold=0.0)
    assert label_low == "trash", "threshold=0.0 should always label as trash"

    label_high, _, _ = predict(img, model, CLASS_NAMES, device, threshold=1.0)
    assert label_high == "no_trash", "threshold=1.0 should always label as no_trash"


# ─────────────────────────────────────────────────────────────────────────────
# draw_label utility
# ─────────────────────────────────────────────────────────────────────────────

def test_draw_label_returns_frame(dummy_image_bgr):
    """draw_label returns a numpy array with the same spatial dimensions."""
    annotated = draw_label(dummy_image_bgr.copy(), "trash", 0.85)
    assert annotated.shape == dummy_image_bgr.shape
