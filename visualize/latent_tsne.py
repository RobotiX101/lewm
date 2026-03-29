"""Latent Space t-SNE Visualization (Figure 9).

Encodes dataset frames and projects embeddings to 2D via t-SNE,
colored by physical state. Compares multiple checkpoints side-by-side
to show how the latent space evolves during training.

Usage:
    export DATASET_ROOT=/path/to/datasets
    python visualize/latent_tsne.py \
        --checkpoints \
            ~/.stable_worldmodel/<run>/lewm_epoch_1_object.ckpt \
            ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
        --dataset_name top_short_merged \
        --max_samples 2000 \
        --output latent_tsne.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Add repo root to path so we can import project modules
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
    all_embs, all_states = [], []
    for i in range(min(max_samples, len(dataset))):
        try:
            sample = dataset[i]
        except Exception:
            continue
        pixels = sample["pixels"].unsqueeze(0).to(device)
        with torch.no_grad():
            info = model.encode({"pixels": pixels})
        emb = info["emb"][:, -1].cpu().numpy()  # last timestep
        all_embs.append(emb)
        if "state" in sample:
            all_states.append(sample["state"][-1].numpy())
    return (np.concatenate(all_embs, axis=0),
            np.concatenate(all_states, axis=0) if all_states else None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--state_dim", type=int, default=0,
                        help="State dimension for coloring")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="latent_tsne.png")
    args = parser.parse_args()

    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = make_dataset(args.dataset_name, transform)
    print(f"Dataset: {len(dataset)} samples")

    results = {}
    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Skip {ckpt_path}")
            continue
        epoch = extract_epoch(ckpt_path)
        label = f"Epoch {epoch}"
        print(f"{label}: {ckpt_path.name}")
        model = torch.load(str(ckpt_path), map_location=args.device, weights_only=False)
        model.eval()
        emb, states = extract_embeddings(model, dataset, args.max_samples, args.device)
        results[label] = {"emb": emb, "states": states}
        print(f"  {emb.shape[0]} samples, dim={emb.shape[1]}")
        del model
        torch.cuda.empty_cache()

    if not results:
        print("No checkpoints loaded!")
        return

    print("Running t-SNE...")
    tsne_results = {}
    for label, data in results.items():
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42)
        tsne_results[label] = tsne.fit_transform(data["emb"])

    first = next(iter(results.values()))
    colors = first["states"][:, args.state_dim] if first["states"] is not None \
        else np.arange(first["emb"].shape[0])
    clabel = f"State dim {args.state_dim}" if first["states"] is not None else "Index"

    n = len(tsne_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, (label, emb_2d) in zip(axes, tsne_results.items()):
        sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap="viridis", s=4, alpha=0.7)
        ax.set_title(label, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(sc, ax=ax, label=clabel)

    fig.suptitle("LeWM Latent Space (t-SNE)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
