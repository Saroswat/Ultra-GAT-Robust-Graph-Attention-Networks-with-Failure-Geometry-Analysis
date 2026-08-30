from __future__ import annotations

import copy
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from .config import ExperimentConfig
from .data import add_structural_features, feature_mask, load_dataset
from .metrics import expected_calibration_error, masked_accuracy
from .models import GraphClassifier, build_model, parameter_count
from .robustness import corrupt_features, drop_edges, robustness_auc


@dataclass(slots=True)
class RunResult:
    seed: int
    model: str
    dataset: str
    best_epoch: int
    validation_accuracy: float
    test_accuracy: float
    ece: float
    temperature: float
    parameters: int


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _label_propagation_targets(data: Data, classes: int, alpha: float, steps: int) -> Tensor:
    source, target = data.edge_index
    adjacency = torch.sparse_coo_tensor(
        torch.stack([target, source]),
        torch.ones(source.numel(), device=data.x.device),
        (data.num_nodes, data.num_nodes),
        check_invariants=False,
    ).coalesce()
    row_degree = torch.sparse.sum(adjacency, dim=1).to_dense().clamp_min(1)
    values = adjacency.values() / row_degree[adjacency.indices()[0]]
    transition = torch.sparse_coo_tensor(
        adjacency.indices(), values, adjacency.shape, check_invariants=False
    ).coalesce()
    initial = torch.zeros((data.num_nodes, classes), device=data.x.device)
    initial[data.train_mask] = nn.functional.one_hot(
        data.y[data.train_mask], num_classes=classes
    ).float()
    scores = initial
    for _ in range(steps):
        scores = alpha * torch.sparse.mm(transition, scores) + (1 - alpha) * initial
    return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def fit_temperature(logits: Tensor, labels: Tensor) -> float:
    log_temperature = nn.Parameter(torch.zeros((), device=logits.device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=50)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_temperature.exp(), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().detach().clamp(0.05, 10.0))


def train_model(
    config: ExperimentConfig,
    *,
    output_dir: str | Path = "artifacts",
    save_artifact: bool = True,
) -> tuple[RunResult, GraphClassifier, Data]:
    seed_everything(config.seed)
    device = choose_device()
    dataset, data = load_dataset(config.dataset)
    if config.structural_features:
        data = add_structural_features(data)
    data = data.to(device)
    model = build_model(
        config.model,
        data.num_features,
        config.hidden_channels,
        dataset.num_classes,
        heads=config.heads,
        layers=config.layers,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    lp_targets = _label_propagation_targets(
        data, dataset.num_classes, config.label_propagation_alpha, config.label_propagation_steps
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -1.0
    best_epoch = 0
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad()
        progress = 1 - (epoch - 1) / max(config.epochs - 1, 1)
        train_edges = drop_edges(data.edge_index, config.dropedge * progress)
        train_x = feature_mask(data.x, config.feature_mask)
        logits = model(train_x, train_edges)
        supervised = nn.functional.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        consistency = nn.functional.kl_div(
            logits.log_softmax(dim=-1), lp_targets, reduction="batchmean"
        )
        loss = supervised + config.auxiliary_weight * consistency
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            clean_logits = model(data.x, data.edge_index)
            validation = masked_accuracy(clean_logits, data.y, data.val_mask)
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    temperature = 1.0
    if config.temperature_calibration:
        temperature = fit_temperature(logits[data.val_mask], data.y[data.val_mask])
    calibrated = logits / temperature
    result = RunResult(
        seed=config.seed,
        model=config.model,
        dataset=config.dataset,
        best_epoch=best_epoch,
        validation_accuracy=masked_accuracy(logits, data.y, data.val_mask),
        test_accuracy=masked_accuracy(logits, data.y, data.test_mask),
        ece=expected_calibration_error(calibrated[data.test_mask], data.y[data.test_mask]),
        temperature=temperature,
        parameters=parameter_count(model),
    )
    if save_artifact:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        stem = f"{config.dataset.lower()}-{config.model}-seed{config.seed}"
        torch.save(
            {
                "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                "config": config.to_dict(),
                "result": asdict(result),
            },
            destination / f"{stem}.pt",
        )
        (destination / f"{stem}.json").write_text(
            json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8"
        )
    return result, model, data


@torch.no_grad()
def robustness_sweep(
    model: GraphClassifier,
    data: Data,
    levels: list[float],
    *,
    mode: str,
    seed: int = 0,
) -> dict:
    model.eval()
    scores: list[float] = []
    for level in levels:
        generator = torch.Generator(device=data.x.device).manual_seed(seed)
        edges = data.edge_index
        features = data.x
        if mode == "edges":
            edges = drop_edges(edges, level, generator=generator)
        elif mode == "features":
            features = corrupt_features(features, level, generator=generator)
        else:
            raise ValueError("mode must be 'edges' or 'features'")
        scores.append(masked_accuracy(model(features, edges), data.y, data.test_mask))
    return {
        "mode": mode,
        "levels": levels,
        "accuracy": scores,
        "auc": robustness_auc(levels, scores),
    }
