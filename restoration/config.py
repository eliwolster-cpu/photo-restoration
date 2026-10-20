"""Load config.yaml and compute its version hash."""
import hashlib
from pathlib import Path

import yaml


class ConfigError(Exception):
    """config.yaml is missing or invalid."""


def load_config(path="config.yaml"):
    """Return (config dict, config_version). The version is a hash of the file bytes."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    raw = path.read_bytes()
    try:
        config = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if not isinstance(config, dict):
        raise ConfigError(f"config must be a mapping: {path}")
    return config, hashlib.sha256(raw).hexdigest()[:12]
