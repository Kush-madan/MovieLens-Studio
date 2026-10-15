"""Project configuration loading and relative path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file and return its mapping."""
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as config_file:
        configuration = yaml.safe_load(config_file)

    if not isinstance(configuration, dict):
        raise ValueError(f"Configuration must contain a YAML mapping: {path}")
    return configuration


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def configured_path(configuration: dict[str, Any], name: str) -> Path:
    """Return a required path from the configuration."""
    paths = configuration.get("paths")
    if not isinstance(paths, dict) or name not in paths:
        raise KeyError(f"Missing paths.{name} in project configuration")
    return resolve_project_path(paths[name])
