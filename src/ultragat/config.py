from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(slots=True)
class ExperimentConfig:
    dataset: str = "CiteSeer"
    model: str = "gatv2"
    seed: int = 42
    hidden_channels: int = 64
    heads: int = 8
    layers: int = 3
    dropout: float = 0.55
    dropedge: float = 0.20
    feature_mask: float = 0.05
    learning_rate: float = 0.005
    weight_decay: float = 0.0005
    epochs: int = 500
    patience: int = 80
    structural_features: bool = True
    label_propagation_alpha: float = 0.90
    label_propagation_steps: int = 50
    auxiliary_weight: float = 0.10
    temperature_calibration: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        return cls(**values)

    def to_dict(self) -> dict:
        return asdict(self)
