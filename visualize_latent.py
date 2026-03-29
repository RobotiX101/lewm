"""Visualize LeWM latent space with t-SNE (like Figure 9 in the paper).

Compares embeddings from multiple checkpoints side-by-side.

Usage:
    export DATASET_ROOT=/path/to/datasets
    export STABLEWM_HOME=~/.stable_worldmodel

    python visualize_latent.py \
        --checkpoints ~/.stable_worldmodel/<run_id>/lewm_epoch_1_object.ckpt \
                       ~/.stable_worldmodel/<run_id>/lewm_epoch_50_object.ckpt \
                       ~/.stable_workspace/<run_id>/lewm_epoch_100_object.ckpt \
        --dataset_name top_short_merged \
        --max_samples 2000 \
        --color_by state \
        --state_dims 0 1 \
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


def load_model(ckpt_path: str, device: str = "cuda"):
    """Load a JEPA model from an _object.ckpt checkpoint."""
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    return model


def extract_embeddings(model, dataset, max_samples: int, device: str = "cuda"):
    """Extract embeddings and corresponding states from dataset samples.

    Returns:
        embeddings: (N, D) numpy array
        states: (N, state_dim) numpy array or None
    """
    all_embs = []
    all_states = []
    n_loaded = 0

    for i in range(len(dataset)):
        if n_loaded >= max_samples:
            break
        try:
            sample = dataset[i]
        except Exception:
            continue

        pixels = sample["pixels"].unsqueeze(0).to(device)
        with torch.no_grad():
            info = model.encode({"pixels": pixels})
        emb = info["emb"][:, -1].cpu().numpy()  # (1, D)

        all_embs.append(emb)
        if "state" in sample:
            all_states.append(sample["state"].numpy())
        n_loaded += 1

    embeddings = np.concatenate(all_embs, axis=0)
    states = np.concatenate(all_states, axis=0) if all_states else None
    return embeddings, states


def plot_tsne(ax, emb_2d, colors, title, cmap="viridis", colorbar_label=""):
    """Plot a single t-SNE scatter."""
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, cmap=cmap, s=4, alpha=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(sc, ax=ax, label=colorbar_label)


def main():
    parser = argparse.ArgumentParser(description="Visualize LeWM latent space with t-SNE")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                        help="Paths to _object.ckpt files")
    parser.add_argument("--dataset_name", default="top_short_merged",
                        help="Dataset name under DATASET_ROOT")
    parser.add_argument("--max_samples", type=int, default=2000,
                        help="Max number of samples to embed")
    parser.add_argument("--color_by", choices=["state", "index"], default="state",
                        help="Color scatter points by state value or sample index")
    parser.add_argument("--state_dims", type=int, nargs=2, default=[0, 1],
                        help="Which state dimensions for color (x, y)")
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    parser.add_argument("--device", default="cuda",
                        help="Device for inference")
    parser.add_argument("--output", default="latent_tsne.png",
                        help="Output image path")
    args = parser.parse_args()

    # --- Load dataset (no transform, raw pixels for encoding) ---
    dataset_root = os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))
    dataset_path = Path(dataset_root) / args.dataset_name

    from lerobot_dataset import LeRobotDatasetWrapper
    dataset = LeRobotDatasetWrapper(
        root=dataset_path,
        frameskip=5,
        num_steps=4,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=[
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        ],
    )

    # Apply image preprocessing (resize + normalize)
    from utils import get_img_preprocessor
    from stable_pretraining.data import transforms as dt_transforms
    img_preproc = get_img_preprocessor(source="pixels", target="pixels", img_size=224)
    dataset.transform = img_preproc

    print(f"Dataset: {len(dataset)} samples")

    # --- Extract embeddings for each checkpoint ---
    results = {}
    for ckpt_path in args.checkpoints:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_path} (not found)")
            continue

        label = ckpt_path.stem
        # Extract epoch number from filename like "lewm_epoch_10_object"
        parts = ckpt_path.name.split("_")
        for i, p in enumerate(parts):
            if p == "epoch" and i + 1 < len(parts):
                label = f"Epoch {parts[i+1]}"
                break

        print(f"Loading {label} from {ckpt_path}")
        model = load_model(str(ckpt_path), device=args.device)
        embeddings, states = extract_embeddings(
            model, dataset, args.max_samples, device=args.device
        )
        results[label] = {"embeddings": embeddings, "states": states}
        print(f"  Extracted {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")

        del model
        torch.cuda.empty_cache()

    if not results:
        print("No valid checkpoints found!")
        sys.exit(1)

    # --- Run t-SNE ---
    print("\nRunning t-SNE...")
    tsne_results = {}
    for label, data in results.items():
        print(f"  t-SNE for {label} ({data['embeddings'].shape[0]} points, dim={data['embeddings'].shape[1]})")
        tsne = TSNE(n_components=2, perplexity=args.perplexity, random_state=42)
        emb_2d = tsne.fit_transform(data["embeddings"])
        tsne_results[label] = emb_2d

    # --- Determine colors ---
    first_data = next(iter(results.values()))
    if args.color_by == "state" and first_data["states"] is not None:
        d1, d2 = args.state_dims
        colors = first_data["states"][:, d1]
        colorbar_label = f"State dim {d1}"
    else:
        colors = np.arange(first_data["embeddings"].shape[0])
        colorbar_label = "Sample index"

    # --- Plot ---
    n_checkpoints = len(tsne_results)
    fig, axes = plt.subplots(1, n_checkpoints, figsize=(6 * n_checkpoints, 5))
    if n_checkpoints == 1:
        axes = [axes]

    for ax, (label, emb_2d) in zip(axes, tsne_results.items()):
        plot_tsne(ax, emb_2d, colors, title=label, colorbar_label=colorbar_label)

    fig.suptitle("LeWM Latent Space (t-SNE)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
