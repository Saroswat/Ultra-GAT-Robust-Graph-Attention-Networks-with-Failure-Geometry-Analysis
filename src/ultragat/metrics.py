from __future__ import annotations

import torch
from torch import Tensor


def masked_accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean())


def expected_calibration_error(logits: Tensor, labels: Tensor, bins: int = 15) -> float:
    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities = logits.softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction.eq(labels)
    edges = torch.linspace(0, 1, bins + 1, device=logits.device)
    error = torch.zeros((), device=logits.device)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        members = (confidence > lower) & (confidence <= upper)
        if members.any():
            error += (
                members.float().mean()
                * (confidence[members].mean() - correct[members].float().mean()).abs()
            )
    return float(error)


def entropy(logits: Tensor) -> Tensor:
    probabilities = logits.softmax(dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def classification_margin(logits: Tensor) -> Tensor:
    top_two = logits.softmax(dim=-1).topk(k=2, dim=-1).values
    return top_two[:, 0] - top_two[:, 1]
