"""
setup_cache.py
--------------
Must be called FIRST in every Colab notebook before any HuggingFace imports.

Redirects the HuggingFace cache (datasets + models) to Google Drive so that:
  - Datasets are downloaded once and reused across sessions
  - Model weights are cached and not re-downloaded each time
  - Distillation traces and processed datasets persist

Usage (first cell of every notebook):
    import sys
    sys.path.insert(0, '/content/SC4001-Group133')
    from src.setup_cache import setup

    setup()   # call before ANY import from transformers/datasets/peft

If not running on Colab (local or Docker), this is a no-op.
"""

import os
import logging

logger = logging.getLogger(__name__)

# TODO: change this to match your actual Google Drive folder structure
DRIVE_ROOT = "/content/drive/MyDrive/SC4001"


def setup(drive_root: str = DRIVE_ROOT) -> None:
    """
    Point all HuggingFace caches to Google Drive.
    Must be called before importing transformers, datasets, or peft.
    """
    if not _is_colab():
        logger.info("Not on Colab — using default local HF cache, no action needed.")
        return

    if not os.path.exists("/content/drive"):
        raise RuntimeError(
            "Google Drive is not mounted. "
            "Run this first:\n"
            "  from google.colab import drive\n"
            "  drive.mount('/content/drive')"
        )

    # Paths on Drive that persist across sessions
    hf_cache   = f"{drive_root}/hf_cache"
    data_dir   = f"{drive_root}/data"
    results_dir = f"{drive_root}/results"

    # Create dirs if they don't exist yet
    for path in [hf_cache, data_dir, results_dir]:
        os.makedirs(path, exist_ok=True)

    # Set HuggingFace environment variables BEFORE any hf imports
    os.environ["HF_HOME"]              = hf_cache   # datasets + tokenizers + models
    os.environ["TRANSFORMERS_CACHE"]   = hf_cache   # transformers model weights
    os.environ["HF_DATASETS_CACHE"]    = f"{hf_cache}/datasets"

    # Symlink data/ and results/ inside the repo to their Drive counterparts
    # so all src/ code can use relative paths (data/, results/) without changes
    _symlink_if_needed("/content/SC4001-Group133/data",    data_dir)
    _symlink_if_needed("/content/SC4001-Group133/results", results_dir)

    logger.info(
        f"HF cache → {hf_cache}\n"
        f"data/    → {data_dir}\n"
        f"results/ → {results_dir}"
    )
    print(f"[setup_cache] Cache redirected to Drive: {drive_root}")


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _symlink_if_needed(link_path: str, target_path: str) -> None:
    """
    Create a symlink at link_path pointing to target_path.
    If link_path already exists as a real directory (first run), move its
    contents to Drive first then replace with symlink.
    """
    if os.path.islink(link_path):
        return  # already symlinked, nothing to do

    if os.path.isdir(link_path):
        # Copy any existing local content to Drive before replacing with symlink
        import shutil
        for item in os.listdir(link_path):
            src = os.path.join(link_path, item)
            dst = os.path.join(target_path, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(link_path)

    os.symlink(target_path, link_path)
    logger.info(f"Symlinked {link_path} → {target_path}")
