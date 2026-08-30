# Contributing to Ultra-GAT

Thanks for helping make graph-model evaluation more transparent and reproducible.

1. Open an issue describing the change or experiment.
2. Create a focused branch and include tests for behavioral changes.
3. Run `uv run ruff check .` and `uv run pytest`.
4. Report dataset, split, seed, device, package versions, and mean ± standard deviation for results.

Please do not present a single favorable seed as a benchmark result. New models should use the same
data masks, stopping rule, and evaluation protocol as existing baselines.
