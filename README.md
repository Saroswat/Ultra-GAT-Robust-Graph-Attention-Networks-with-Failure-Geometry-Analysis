# Ultra-GAT

**A reproducible research lab for graph attention robustness, uncertainty, and failure geometry.**

[![CI](https://github.com/Saroswat/Ultra-GAT-Robust-Graph-Attention-Networks-with-Failure-Geometry-Analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Saroswat/Ultra-GAT-Robust-Graph-Attention-Networks-with-Failure-Geometry-Analysis/actions)
[![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB.svg)](https://www.python.org/)
[![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-2.5+-EE4C2C.svg)](https://pyg.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPL_v3-blue.svg)](LICENSE)

**[Launch the public Ultra-GAT Lab](https://ultra-gat-lab.saroswat.chatgpt.site)**

Ultra-GAT turns a large exploratory Graph Attention Network notebook into a package that other people
can install, benchmark, inspect, and extend. It compares GCN, GAT, and GATv2 under one protocol, then
goes beyond clean accuracy to reveal how each model fails when graph structure or node features are
damaged.

## Why this project

Most graph-learning demos end with one accuracy number. Ultra-GAT treats that number as the start of
the investigation.

- **Reproducible training:** YAML configurations, fixed seeds, early stopping, and saved checkpoints.
- **Fair baselines:** GCN, original GAT, and GATv2 use the same Planetoid splits and reporting format.
- **Robustness testing:** edge deletion and feature corruption curves with normalized area under curve.
- **Trust signals:** temperature calibration, expected calibration error, uncertainty, and margins.
- **Failure geometry:** interactive embedding maps and node-level inspection of wrong predictions.
- **Research hygiene:** multi-seed summaries, machine-readable artifacts, CI, tests, and citation metadata.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
git clone https://github.com/Saroswat/Ultra-GAT-Robust-Graph-Attention-Networks-with-Failure-Geometry-Analysis.git
cd Ultra-GAT-Robust-Graph-Attention-Networks-with-Failure-Geometry-Analysis
uv sync --extra dev
uv run ultragat train --config configs/citeseer.yaml
```

The first run downloads the selected Planetoid dataset. A checkpoint and JSON result are written to
`artifacts/`.

## Interactive lab

The fastest way to explore the project is the
[public browser lab](https://ultra-gat-lab.saroswat.chatgpt.site). It runs live message passing,
edge deletion, feature masking, node inspection, and result export without an account or local setup.

For the full PyTorch Geometric workspace locally:

```bash
uv run streamlit run app.py
```

Choose CiteSeer, Cora, or PubMed; compare GCN, GAT, and GATv2; then explore:

- latent node clusters and misclassifications;
- accuracy as edges disappear or features are masked;
- confidence, entropy, and decision margin for individual failures;
- configuration, parameter count, calibration, and best checkpoint epoch.

## Credible benchmarks

One lucky seed is not a result. The benchmark command runs the same protocol across models and seeds:

```bash
uv run ultragat benchmark \
  --config configs/citeseer.yaml \
  --models gcn,gat,gatv2 \
  --seeds 11,22,33,44,55
```

It reports mean ± standard deviation and writes every run to
`artifacts/benchmark/benchmark.json`. The verified run is checked in as
[results/citeseer-five-seed.json](results/citeseer-five-seed.json). See
[docs/RESULTS.md](docs/RESULTS.md) for the claims policy and the distinction between reproducible
results and exploratory notebook observations.

Verified on CiteSeer across seeds 11, 22, 33, 44, and 55:

| Model | Test accuracy | ECE ↓ | Edge AUC ↑ | Feature AUC ↑ |
| --- | ---: | ---: | ---: | ---: |
| GCN | **67.74% ± 0.60%** | 0.0727 | **0.6681** | **0.6682** |
| GAT | 67.62% ± 1.31% | **0.0348** | 0.6668 | 0.6607 |
| GATv2 | 67.24% ± 1.19% | 0.0419 | 0.6626 | 0.6556 |

These results do not pretend attention automatically wins. The interesting question is how each
model behaves under damage and where its confidence becomes unreliable.

## Project map

```text
.
├── app.py                     # Streamlit analysis workspace
├── configs/                   # Reproducible experiment settings
├── docs/RESULTS.md            # Protocol, claims, and negative results
├── notebooks/                 # Original exploration, preserved without outputs
├── src/ultragat/
│   ├── benchmark.py           # Multi-model, multi-seed evaluation
│   ├── data.py                # Planetoid loading and structural features
│   ├── metrics.py             # Accuracy, calibration, entropy, margins
│   ├── models.py              # GCN, GAT, and GATv2
│   ├── robustness.py          # Edge and feature stress tests
│   └── training.py            # Training, early stopping, calibration, artifacts
└── tests/                     # Fast unit tests used by CI
```

## Method

The default GATv2 model uses multi-head attention, residual hidden blocks, LayerNorm, feature masking,
and annealed DropEdge. A soft label-propagation target acts as a graph-consistency regularizer during
training. Temperature scaling is fit only on the validation split and applied before test calibration
is measured.

The robustness score is the normalized area under the test-accuracy curve across corruption levels.
It rewards models that preserve useful behavior across the full stress test instead of at one chosen
perturbation.

## Origins and attribution

The exploratory notebook began as a learning exercise influenced by Maxime Labonne's Graph Attention
Network tutorial and was expanded substantially with GATv2, structural augmentation, distillation
experiments, uncertainty, calibration, perturbation sweeps, and failure analysis. That provenance is
retained here because clear attribution is part of credible open research.

## Roadmap

- Add GraphSAGE and GIN benchmark adapters.
- Publish verified five-seed tables for all three Planetoid datasets.
- Add adversarial edge attacks and out-of-distribution graph shifts.
- Release a hosted demo and versioned model cards.
- Archive a stable release on Zenodo for a DOI.

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md), open a focused issue, and cite the
project using [CITATION.cff](CITATION.cff) when it supports published work.

## License

GPL-3.0. See [LICENSE](LICENSE).
