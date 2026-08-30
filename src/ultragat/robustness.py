from __future__ import annotations

import torch
from torch import Tensor


def drop_edges(
    edge_index: Tensor,
    probability: float,
    *,
    training: bool = True,
    generator: torch.Generator | None = None,
) -> Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if not training or probability == 0:
        return edge_index
    keep = (
        torch.rand(edge_index.shape[1], device=edge_index.device, generator=generator)
        >= probability
    )
    return edge_index[:, keep]


def corrupt_features(
    x: Tensor, probability: float, *, generator: torch.Generator | None = None
) -> Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0:
        return x
    keep = torch.rand(x.shape, device=x.device, generator=generator) >= probability
    return x * keep


def robustness_auc(levels: list[float], scores: list[float]) -> float:
    if len(levels) != len(scores) or len(levels) < 2:
        raise ValueError("levels and scores must have the same length of at least two")
    x = torch.tensor(levels, dtype=torch.float)
    y = torch.tensor(scores, dtype=torch.float)
    order = x.argsort()
    width = x[order][-1] - x[order][0]
    return float(torch.trapz(y[order], x[order]) / width.clamp_min(1e-12))
