"""Temporal Latent Path Straightening (Figure 17).

Computes mean cosine similarity between consecutive latent velocity vectors
across training epochs. An emergent property of LeWM: latent trajectories
become straighter over training without any temporal regularization.

    v_t = z_{t+1} - z_t
    S_straight = mean(cos_sim(v_{t+1}, v_t))

Usage:
    export DATASET_ROOT=/path/to/datasets
    python visualize/temporal_straightening.py \
        --checkpoints \
            ~/.stable_worldmodel/<run>/lewm_epoch_1_object.ckpt \
            ~/.stable_worldmodel/<run>/lewm_epoch_50_object.ckpt \
            ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
        --n_samples 2000 \
        --output temporal_straightening.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lerobot_dataset import LeRobotDatasetWrapper
from utils import get_img_preprocessor

CAMERA_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]


def extract_epoch(path):
    parts = Path(path).name.split("_")
    for j, p in enumerate(parts):
        if p == "epoch" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return 0


def compute_straightness(model, dataset, n_samples, device="cuda"):
    """Mean cosine similarity between consecutive velocity vectors."""
    all_cos = []
    for i in range(min(n_samples, len(dataset))):
        try:
            sample = dataset[i]
        except Exception:
            continue
        pixels = sample["pixels"].unsqueeze(0).to(device)  # (1, T, C, H, W)
        with torch.no_grad():
            info = model.encode({"pixels": pixels})
        emb = info["emb"][0].cpu()  # (T, D)
        vel = emb[1:] - emb[:-1]   # (T-1, D)
        if vel.shape[0] < 2:
            continue
        cos = F.cosine_similarity(vel[1:], vel[:-1], dim=-1)  # (T-2,)
        all_cos.append(cos.numpy())

    if not all_cos:
        return 0.0
    return np.concatenate(all_cos).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--n_samples", type=int, default=2000)
    parser.add_argument("--output", default="temporal_straightening.png")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    # Dataset with image transform
    root = Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))) / args.dataset_name
    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = LeRobotDatasetWrapper(
        root=root, frameskip=5, num_steps=4,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )
    print(f"Dataset: {len(dataset)} samples")

    epochs, scores = [], []
    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Skip {ckpt_path} (not found)")
            continue
        epoch = extract_epoch(ckpt_path)
        print(f"Epoch {epoch}: {ckpt_path.name}", end="", flush=True)
        model = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
        model.eval()
        s = compute_straightness(model, dataset, args.n_samples, args.device)
        epochs.append(epoch)
        scores.append(s)
        print(f"  straightness={s:.4f}")
        del model
        torch.cuda.empty_cache()

    if not epochs:
        print("No valid checkpoints!")
        return

    pairs = sorted(zip(epochs, scores))
    e, s = zip(*pairs)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(e, s, "o-", color="#2196F3", lw=2, markersize=8)
    ax.set_xlabel("Training Epoch", fontsize=12)
    ax.set_ylabel("Mean Cosine Similarity", fontsize=12)
    ax.set_title("Temporal Latent Path Straightening", fontsize=14)
    ax.set_ylim(-1, 1)
    ax.axhline(0, color="gray", ls="--", alpha=0.3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
