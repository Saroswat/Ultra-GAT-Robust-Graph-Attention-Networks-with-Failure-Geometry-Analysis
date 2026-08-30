from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import degree

SUPPORTED_DATASETS = ("CiteSeer", "Cora", "PubMed")


def load_dataset(name: str, root: str | Path = "data") -> tuple[Planetoid, Data]:
    canonical = next((item for item in SUPPORTED_DATASETS if item.lower() == name.lower()), None)
    if canonical is None:
        raise ValueError(f"Unsupported dataset {name!r}; choose from {SUPPORTED_DATASETS}")
    dataset = Planetoid(
        root=str(Path(root) / canonical), name=canonical, transform=NormalizeFeatures()
    )
    return dataset, dataset[0]


def add_structural_features(data: Data) -> Data:
    """Append normalized log-degree as a cheap, deterministic structural cue."""
    result = data.clone()
    node_degree = degree(result.edge_index[0], num_nodes=result.num_nodes, dtype=torch.float)
    log_degree = torch.log1p(node_degree)
    log_degree = (log_degree - log_degree.mean()) / log_degree.std().clamp_min(1e-8)
    result.x = torch.cat([result.x, log_degree[:, None]], dim=-1)
    return result


def feature_mask(x: Tensor, probability: float, generator: torch.Generator | None = None) -> Tensor:
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability == 0:
        return x
    keep = torch.rand(x.shape, device=x.device, generator=generator) >= probability
    return x * keep
