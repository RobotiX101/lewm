"""Latent Space t-SNE Visualization (Figure 9).

Encodes dataset frames and projects embeddings to 2D via t-SNE,
colored by a physical quantity derived from joint states. Compares
multiple checkpoints side-by-side to show how the latent space
structure emerges during training.

A well-trained model shows a smooth, structured manifold where
physically similar states cluster together.

Usage:
    export DATASET_ROOT=/path/to/datasets
    python visualize/latent_tsne.py \
        --checkpoints \
            ~/.stable_worldmodel/lewm_epoch_1_object.ckpt \
            ~/.stable_worldmodel/lewm_epoch_99_object.ckpt \
        --dataset_name top_short_merged \
        --max_samples 2000 \
        --output visualize/latent_tsne.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from sklearn.manifold import TSNE

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lerobot_dataset import LeRobotDatasetWrapper
from utils import get_img_preprocessor

CAMERA_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]

# Dual-arm joint names (6 joints per arm)
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]
LEFT_JOINTS = [f"left_{n}" for n in JOINT_NAMES]
RIGHT_JOINTS = [f"right_{n}" for n in JOINT_NAMES]


def extract_epoch(path):
    parts = Path(path).name.split("_")
    for j, p in enumerate(parts):
        if p == "epoch" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return 0


def make_dataset(name, transform):
    root = Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))) / name
    return LeRobotDatasetWrapper(
        root=root, frameskip=5, num_steps=4,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )


def extract_embeddings(model, dataset, max_samples, device):
    """Encode frames and collect corresponding states."""
    all_embs, all_states = [], []
    n_ok = 0
    indices = np.random.RandomState(42).permutation(len(dataset))

    for idx in indices:
        if n_ok >= max_samples:
            break
        try:
            sample = dataset[int(idx)]
        except Exception:
            continue

        pixels = sample["pixels"]   # (T, 3, 224, 896)
        states = sample["state"]    # (T, 12)

        # Encode the last frame of each clip
        pix = pixels[-1:].unsqueeze(0).to(device)  # (1, 1, 3, H, W)
        with torch.no_grad():
            info = model.encode({"pixels": pix})
        emb = info["emb"][0, -1].cpu().numpy()  # (192,)
        all_embs.append(emb)
        all_states.append(states[-1].numpy())  # (12,)
        n_ok += 1

    return (np.stack(all_embs),
            np.stack(all_states) if all_states else None)


def compute_color_values(states, color_mode):
    """Derive a 1-D scalar from 12-D joint state for coloring."""
    if states is None:
        return np.arange(1000), "Sample index"

    if color_mode == "gripper":
        # Gripper opening: average of left+right gripper (dim 5, 11)
        vals = (states[:, 5] + states[:, 11]) / 2
        label = "Gripper opening"

    elif color_mode == "elbow":
        # Elbow flex: average of left+right elbow (dim 2, 8)
        vals = (states[:, 2] + states[:, 8]) / 2
        label = "Elbow flex (avg)"

    elif color_mode == "arm_distance":
        # Euclidean distance between left and right arm joint configurations
        left = states[:, :6]
        right = states[:, 6:]
        vals = np.linalg.norm(left - right, axis=1)
        label = "L-R arm distance"

    elif color_mode == "state_norm":
        # L2 norm of the full state vector — proxy for "how far from rest"
        vals = np.linalg.norm(states, axis=1)
        label = "State L2 norm"

    elif color_mode == "range":
        # Range (max - min) across all joints — proxy for pose diversity
        vals = states.max(axis=1) - states.min(axis=1)
        label = "Joint range"

    else:
        dim = int(color_mode)
        vals = states[:, dim]
        label = f"State dim {dim} ({LEFT_JOINTS[dim] if dim < 6 else RIGHT_JOINTS[dim-6]})"

    return vals, label


def main():
    parser = argparse.ArgumentParser(description="Latent Space t-SNE (Figure 9)")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Model checkpoint paths (side-by-side comparison)")
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--color_mode", default="state_norm",
                        choices=["gripper", "elbow", "arm_distance", "state_norm",
                                 "range", "0", "1", "2", "3", "4", "5",
                                 "6", "7", "8", "9", "10", "11"],
                        help="State-derived scalar for coloring points")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="visualize/latent_tsne.png")
    args = parser.parse_args()

    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = make_dataset(args.dataset_name, transform)
    print(f"Dataset: {len(dataset)} samples")

    # Encode with each checkpoint
    results = {}
    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Skip {ckpt_path}")
            continue
        epoch = extract_epoch(ckpt_path)
        label = f"Epoch {epoch}"
        print(f"\n{label}: {ckpt_path.name}")
        model = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
        model.eval()
        emb, states = extract_embeddings(model, dataset, args.max_samples, args.device)
        results[label] = {"emb": emb, "states": states}
        print(f"  Encoded {emb.shape[0]} frames, dim={emb.shape[1]}")
        del model
        torch.cuda.empty_cache()

    if not results:
        print("No checkpoints loaded!")
        return

    # Compute color values from states (use same states for all panels)
    ref_states = next(iter(results.values()))["states"]
    colors, clabel = compute_color_values(ref_states, args.color_mode)

    # Run t-SNE on each checkpoint's embeddings
    print("\nRunning t-SNE...")
    tsne_results = {}
    for label, data in results.items():
        print(f"  t-SNE for {label}...")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42)
        tsne_results[label] = tsne.fit_transform(data["emb"])

    # Plot side-by-side
    n = len(tsne_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5))
    if n == 1:
        axes = [axes]

    vmin, vmax = np.percentile(colors, [2, 98])

    for ax, (label, emb_2d) in zip(axes, tsne_results.items()):
        sc = ax.scatter(
            emb_2d[:, 0], emb_2d[:, 1],
            c=colors, cmap="viridis", s=4, alpha=0.7,
            vmin=vmin, vmax=vmax,
        )
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    # Shared colorbar
    fig.subplots_adjust(right=0.92)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(clabel, fontsize=11)

    fig.suptitle("LeWM Latent Space (t-SNE)", fontsize=14, y=1.02)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
