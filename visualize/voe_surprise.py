"""Violation-of-Expectation Surprise Analysis (Figure 10).

Computes per-timestep surprise (MSE between predicted and actual next embedding)
under three conditions:
  1. Unperturbed  — normal trajectory
  2. Visual       — color channel rotation at perturbation point
  3. Physical     — frame swapped with a random frame (teleportation)

A well-trained model shows a large spike for physical perturbation but only a
small increase for visual, indicating it learned physical structure not pixels.

Usage:
    export DATASET_ROOT=/path/to/datasets
    python visualize/voe_surprise.py \
        --checkpoint ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
        --dataset_name top_short_merged \
        --n_trajectories 50 \
        --output voe_surprise.png
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lerobot_dataset import LeRobotDatasetWrapper
from utils import get_img_preprocessor

CAMERA_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]


def make_dataset(name, num_steps, transform=None):
    root = Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))) / name
    return LeRobotDatasetWrapper(
        root=root, frameskip=5, num_steps=num_steps,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )


def compute_surprise_curve(model, pixels, actions, history_size, device="cuda"):
    """Per-timestep surprise = MSE(predicted_emb, actual_emb).

    Args:
        pixels: (T, C, H, W) preprocessed
        actions: (T, frameskip * act_dim)
    Returns:
        list[float] of length T - history_size
    """
    T = pixels.shape[0]
    pix = pixels.unsqueeze(0).to(device)
    act = actions.unsqueeze(0).to(device)

    with torch.no_grad():
        info = model.encode({"pixels": pix, "action": act})
        emb = info["emb"]          # (1, T, D)
        act_emb = info["act_emb"]  # (1, T, D_act)

    surprises = []
    for t in range(history_size, T):
        ctx_emb = emb[:, t - history_size : t]
        ctx_act = act_emb[:, t - history_size : t]
        with torch.no_grad():
            pred = model.predict(ctx_emb, ctx_act)[:, -1]  # (1, D)
        actual = emb[:, t]  # (1, D)
        mse = (pred - actual).pow(2).sum(dim=-1).item()
        surprises.append(mse)
    return surprises


def color_shift_frame(frame):
    """Rotate color channels: R->G, G->B, B->R."""
    p = frame.clone()
    p[0], p[1], p[2] = frame[1], frame[2], frame[0]
    return p


def preprocess_frames(raw_pixels):
    """Apply standard image preprocessing to a (T, C, H, W) tensor."""
    from utils import get_img_preprocessor
    transform = get_img_preprocessor("pixels", "pixels", 224)
    return transform({"pixels": raw_pixels})["pixels"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--n_trajectories", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--perturb_frac", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="voe_surprise.png")
    args = parser.parse_args()

    print(f"Loading model: {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.eval()

    # Dataset with transform for normal trajectories
    dataset = make_dataset(args.dataset_name, num_steps=args.num_steps,
                           transform=get_img_preprocessor("pixels", "pixels", 224))
    # Raw dataset (no transform) for perturbation control
    dataset_raw = make_dataset(args.dataset_name, num_steps=args.num_steps)

    perturb_step = int(args.num_steps * args.perturb_frac)
    print(f"Dataset: {len(dataset)} samples, perturbation at step {perturb_step}")

    unperturbed_all, visual_all, physical_all = [], [], []
    n_ok = 0
    order = np.random.permutation(len(dataset))

    for idx in order:
        if n_ok >= args.n_trajectories:
            break
        try:
            sample = dataset[int(idx)]
            sample_raw = dataset_raw[int(idx)]
        except Exception:
            continue

        pixels = sample["pixels"]    # (T, C, H, W) preprocessed
        actions = sample["action"]   # (T, frameskip * act_dim)
        raw_px = sample_raw["pixels"]  # (T, C, H, W) raw
        T = pixels.shape[0]
        if T < args.history_size + 3 or perturb_step >= T:
            continue

        # 1. Unperturbed
        s_normal = compute_surprise_curve(
            model, pixels, actions, args.history_size, args.device)

        # 2. Visual: shift colors at perturb_step
        vis_raw = raw_px.clone()
        vis_raw[perturb_step] = color_shift_frame(vis_raw[perturb_step])
        vis_px = preprocess_frames(vis_raw)
        s_visual = compute_surprise_curve(
            model, vis_px, actions, args.history_size, args.device)

        # 3. Physical: swap frame with random one
        phys_raw = raw_px.clone()
        rand_idx = np.random.randint(0, len(dataset_raw))
        try:
            rand_sample = dataset_raw[rand_idx]
            rand_frames = rand_sample["pixels"]
            phys_raw[perturb_step] = rand_frames[min(perturb_step, rand_frames.shape[0] - 1)]
        except Exception:
            phys_raw[perturb_step] = raw_px[perturb_step].flip(-1)
        phys_px = preprocess_frames(phys_raw)
        s_physical = compute_surprise_curve(
            model, phys_px, actions, args.history_size, args.device)

        min_len = min(len(s_normal), len(s_visual), len(s_physical))
        if min_len < 2:
            continue

        unperturbed_all.append(s_normal[:min_len])
        visual_all.append(s_visual[:min_len])
        physical_all.append(s_physical[:min_len])
        n_ok += 1
        if n_ok % 10 == 0:
            print(f"  {n_ok}/{args.n_trajectories}")

    if not unperturbed_all:
        print("No valid trajectories!")
        return

    # Aggregate
    max_len = max(len(c) for c in unperturbed_all)

    def pad_mean(curves):
        arr = np.full((len(curves), max_len), np.nan)
        for i, c in enumerate(curves):
            arr[i, :len(c)] = c
        m = np.nanmean(arr, axis=0)
        n = np.count_nonzero(~np.isnan(arr), axis=0).clip(1)
        se = np.nanstd(arr, axis=0) / np.sqrt(n)
        return m, se

    m_unp, se_unp = pad_mean(unperturbed_all)
    m_vis, se_vis = pad_mean(visual_all)
    m_phy, se_phy = pad_mean(physical_all)
    x = np.arange(max_len) + args.history_size

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, m_unp, label="Unperturbed", color="#4CAF50", lw=2)
    ax.fill_between(x, m_unp - se_unp, m_unp + se_unp, color="#4CAF50", alpha=0.15)
    ax.plot(x, m_vis, label="Visual perturbation", color="#FF9800", lw=2)
    ax.fill_between(x, m_vis - se_vis, m_vis + se_vis, color="#FF9800", alpha=0.15)
    ax.plot(x, m_phy, label="Physical perturbation", color="#F44336", lw=2)
    ax.fill_between(x, m_phy - se_phy, m_phy + se_phy, color="#F44336", alpha=0.15)

    if args.history_size <= perturb_step < args.history_size + max_len:
        ax.axvline(perturb_step, color="gray", ls="--", alpha=0.5,
                   label=f"Perturbation (step {perturb_step})")

    ax.set_xlabel("Timestep", fontsize=12)
    ax.set_ylabel("Surprise (MSE)", fontsize=12)
    ax.set_title("Violation-of-Expectation Surprise", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")

    pi = perturb_step - args.history_size
    if 0 <= pi < max_len:
        print(f"\nSurprise at perturbation (t={perturb_step}):")
        print(f"  Unperturbed:  {m_unp[pi]:.4f}")
        print(f"  Visual:       {m_vis[pi]:.4f}")
        print(f"  Physical:     {m_phy[pi]:.4f}")


if __name__ == "__main__":
    main()
