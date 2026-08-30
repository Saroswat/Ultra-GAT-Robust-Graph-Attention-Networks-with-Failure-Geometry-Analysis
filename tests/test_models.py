import pytest
import torch

from ultragat.models import build_model


@pytest.mark.parametrize("name", ["gcn", "gat", "gatv2"])
def test_model_forward_and_embedding_shapes(name: str) -> None:
    model = build_model(
        name,
        in_channels=4,
        hidden_channels=8,
        out_channels=3,
        heads=2,
        layers=2,
        dropout=0.0,
    )
    features = torch.randn(5, 4)
    edges = torch.tensor([[0, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 2]])
    assert model(features, edges).shape == (5, 3)
    assert model.embed(features, edges).shape == (5, 8)
