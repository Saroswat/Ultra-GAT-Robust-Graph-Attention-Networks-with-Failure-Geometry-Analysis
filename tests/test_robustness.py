import pytest
import torch

from ultragat.robustness import corrupt_features, drop_edges, robustness_auc


def test_zero_corruption_is_identity() -> None:
    edges = torch.tensor([[0, 1], [1, 0]])
    features = torch.ones((2, 3))
    assert torch.equal(drop_edges(edges, 0), edges)
    assert torch.equal(corrupt_features(features, 0), features)


def test_full_corruption_removes_signal() -> None:
    edges = torch.tensor([[0, 1], [1, 0]])
    features = torch.ones((2, 3))
    assert drop_edges(edges, 1).shape == (2, 0)
    assert int(corrupt_features(features, 1).sum()) == 0


def test_robustness_auc() -> None:
    assert robustness_auc([0.0, 0.5, 1.0], [1.0, 0.5, 0.0]) == pytest.approx(0.5)
