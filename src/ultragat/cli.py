from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from .benchmark import run_benchmark
from .config import ExperimentConfig
from .training import robustness_sweep, train_model

app = typer.Typer(help="Train and inspect robust graph attention networks.", no_args_is_help=True)


@app.command()
def train(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/citeseer.yaml"),
    model: Annotated[str | None, typer.Option(help="Override gcn, gat, or gatv2.")] = None,
    seed: Annotated[int | None, typer.Option()] = None,
    robustness: Annotated[bool, typer.Option(help="Run edge and feature sweeps.")] = True,
) -> None:
    settings = ExperimentConfig.from_yaml(config)
    if model is not None:
        settings.model = model
    if seed is not None:
        settings.seed = seed
    result, fitted_model, data = train_model(settings)
    payload: dict = {"result": asdict(result)}
    if robustness:
        levels = [0.0, 0.1, 0.2, 0.3, 0.5]
        payload["robustness"] = [
            robustness_sweep(fitted_model, data, levels, mode="edges"),
            robustness_sweep(fitted_model, data, levels, mode="features"),
        ]
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def benchmark(
    config: Annotated[Path, typer.Option(exists=True)] = Path("configs/citeseer.yaml"),
    models: Annotated[str, typer.Option(help="Comma-separated model names.")] = "gcn,gat,gatv2",
    seeds: Annotated[str, typer.Option(help="Comma-separated random seeds.")] = "11,22,33,44,55",
) -> None:
    report = run_benchmark(
        ExperimentConfig.from_yaml(config),
        models=[item.strip() for item in models.split(",")],
        seeds=[int(item) for item in seeds.split(",")],
    )
    typer.echo(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    app()
