"""
Small helper so every script loads the same config.yaml the same way.
"""
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: str = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)
