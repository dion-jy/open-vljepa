"""YAML config loader for OpenVL-JEPA."""

import yaml
from pathlib import Path


def load_config(path: str) -> dict:
    """Load and return config dict from YAML file."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_encoder_cfg(cfg: dict) -> dict:
    return cfg["encoder"]


def get_y_encoder_cfg(cfg: dict) -> dict:
    return cfg["y_encoder"]


def get_predictor_cfg(cfg: dict) -> dict:
    return cfg["predictor"]


def get_data_cfg(cfg: dict) -> dict:
    return cfg["data"]


def get_training_cfg(cfg: dict) -> dict:
    return cfg["training"]
