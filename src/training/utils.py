"""
utils.py
--------
Shared utilities for training: config loading, reproducibility seeding,
checkpoint management, and Google Drive helpers for Colab.

Member responsible: B
"""

import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    """Load a YAML config file and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """
    Set all random seeds for reproducibility.
    Call this before loading model and data.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def get_device() -> str:
    """Return 'cuda' if a GPU is available, else 'cpu'."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "No GPU detected. Training on CPU will be very slow. "
            "Make sure you have enabled GPU in Colab: Runtime > Change runtime type > T4 GPU."
        )
    return device


def get_checkpoint_dir(base_dir: str, run_name: str) -> Path:
    """
    Construct a checkpoint directory path and create it if it doesn't exist.

    Args:
        base_dir: Base directory from finetune.yaml output_dir
        run_name: Descriptive name, e.g. 'adaptation_r16', 'distillation_filtered'
    """
    path = Path(base_dir) / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_run_config(config: dict, checkpoint_dir: Path) -> None:
    """
    Save the config used for a training run alongside its checkpoint.
    Makes it easy to reproduce or audit any run later.
    """
    config_save_path = checkpoint_dir / "run_config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    logger.info(f"Run config saved to {config_save_path}")


# =============================================================================
# Google Drive helpers (Colab only)
# These functions are no-ops when not running on Colab.
# =============================================================================

def is_colab() -> bool:
    """Check if the current environment is Google Colab."""
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def mount_drive(mount_point: str = "/content/drive") -> str | None:
    """
    Mount Google Drive on Colab. Returns the mount point path or None if not on Colab.

    TODO: call this at the top of training notebooks to ensure checkpoints persist
          across session restarts
    """
    if not is_colab():
        logger.info("Not running on Colab — skipping Drive mount.")
        return None

    from google.colab import drive
    drive.mount(mount_point)
    logger.info(f"Google Drive mounted at {mount_point}")
    return mount_point


def get_drive_path(relative_path: str, drive_root: str = "/content/drive/MyDrive/SC4001") -> str:
    """
    Resolve a path relative to the project folder on Google Drive.

    Usage:
        checkpoint_dir = get_drive_path("results/checkpoints/adaptation_r16")

    TODO: change drive_root to match your actual Drive folder structure
    """
    full_path = Path(drive_root) / relative_path
    full_path.mkdir(parents=True, exist_ok=True)
    return str(full_path)
