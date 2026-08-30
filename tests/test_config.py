from pathlib import Path

import pytest

from ultragat.config import ExperimentConfig


def test_loads_repository_config() -> None:
    config = ExperimentConfig.from_yaml(Path("configs/citeseer.yaml"))
    assert config.dataset == "CiteSeer"
    assert config.model == "gatv2"


def test_rejects_unknown_keys(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("mystery: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration keys"):
        ExperimentConfig.from_yaml(config_file)
