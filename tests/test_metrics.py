import pytest
import torch

from ultragat.metrics import classification_margin, expected_calibration_error, masked_accuracy


def test_masked_accuracy() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0], [0.0, 4.0]])
    labels = torch.tensor([0, 1, 0])
    mask = torch.tensor([True, True, False])
    assert masked_accuracy(logits, labels, mask) == 1.0


def test_calibration_error_is_bounded() -> None:
    logits = torch.tensor([[4.0, 0.0], [0.0, 4.0]])
    error = expected_calibration_error(logits, torch.tensor([0, 1]), bins=5)
    assert 0 <= error <= 1


def test_margin_matches_probability_gap() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    probabilities = logits.softmax(dim=-1)
    expected = probabilities[0, 0] - probabilities[0, 1]
    assert classification_margin(logits)[0] == pytest.approx(float(expected))
