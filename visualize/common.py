"""Shared utilities for visualization scripts."""

import os
from pathlib import Path

import numpy as np
import torch

CAMERA_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]


def get_dataset_root():
    return Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets")))


def make_dataset(dataset_name, num_steps=4, transform=None):
    """Create a LeRobotDatasetWrapper with standard config."""
    from lerobot_dataset import LeRobotDatasetWrapper

    root = get_dataset_root() / dataset_name
    return LeRobotDatasetWrapper(
        root=root,
        frameskip=5,
        num_steps=num_steps,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )


def get_img_transform(img_size=224):
    """Return the standard image preprocessing transform."""
    from utils import get_img_preprocessor
    return get_img_preprocessor("pixels", "pixels", img_size)


def load_model(ckpt_path, device="cuda"):
    """Load a JEPA model from an _object.ckpt checkpoint."""
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    return model


def extract_epoch(path):
    """Extract epoch number from checkpoint filename like lewm_epoch_10_object.ckpt."""
    parts = Path(path).name.split("_")
    for j, p in enumerate(parts):
        if p == "epoch" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return 0
