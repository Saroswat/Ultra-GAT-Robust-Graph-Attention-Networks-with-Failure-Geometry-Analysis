from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .training import robustness_sweep, train_model


def run_benchmark(
    config: ExperimentConfig,
    *,
    models: list[str],
    seeds: list[int],
    output_dir: str | Path = "artifacts/benchmark",
) -> dict:
    rows = []
    levels = [0.0, 0.1, 0.2, 0.3, 0.5]
    for model_name in models:
        for seed in seeds:
            run_config = replace(config, model=model_name, seed=seed)
            result, fitted_model, data = train_model(run_config, output_dir=output_dir)
            row = asdict(result)
            row["edge_robustness_auc"] = robustness_sweep(fitted_model, data, levels, mode="edges")[
                "auc"
            ]
            row["feature_robustness_auc"] = robustness_sweep(
                fitted_model, data, levels, mode="features"
            )["auc"]
            rows.append(row)
    summary = []
    for model_name in models:
        selected = [row for row in rows if row["model"] == model_name]
        accuracy = np.array([row["test_accuracy"] for row in selected])
        ece = np.array([row["ece"] for row in selected])
        edge_auc = np.array([row["edge_robustness_auc"] for row in selected])
        feature_auc = np.array([row["feature_robustness_auc"] for row in selected])
        summary.append(
            {
                "model": model_name,
                "runs": len(selected),
                "test_accuracy_mean": float(accuracy.mean()),
                "test_accuracy_std": float(accuracy.std(ddof=1)) if len(accuracy) > 1 else 0.0,
                "ece_mean": float(ece.mean()),
                "ece_std": float(ece.std(ddof=1)) if len(ece) > 1 else 0.0,
                "edge_robustness_auc_mean": float(edge_auc.mean()),
                "edge_robustness_auc_std": (
                    float(edge_auc.std(ddof=1)) if len(edge_auc) > 1 else 0.0
                ),
                "feature_robustness_auc_mean": float(feature_auc.mean()),
                "feature_robustness_auc_std": (
                    float(feature_auc.std(ddof=1)) if len(feature_auc) > 1 else 0.0
                ),
            }
        )
    report = {"config": config.to_dict(), "runs": rows, "summary": summary}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
