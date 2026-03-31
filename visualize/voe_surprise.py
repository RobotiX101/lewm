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
        --checkpoints \
            ~/.stable_worldmodel/lewm_epoch_1_object.ckpt \
            ~/.stable_worldmodel/lewm_epoch_100_object.ckpt \
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


def extract_epoch(path):
    parts = Path(path).name.split("_")
    for j, p in enumerate(parts):
        if p == "epoch" and j + 1 < len(parts):
            try:
                return int(parts[j + 1])
            except ValueError:
                pass
    return 0


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
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--n_trajectories", type=int, default=50)
    parser.add_argument("--num_steps", type=int, default=16)
    parser.add_argument("--history_size", type=int, default=3)
    parser.add_argument("--perturb_frac", type=float, default=0.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="voe_surprise.png")
    args = parser.parse_args()

    # Dataset with transform for normal trajectories
    dataset = make_dataset(args.dataset_name, num_steps=args.num_steps,
                           transform=get_img_preprocessor("pixels", "pixels", 224))
    # Raw dataset (no transform) for perturbation control
    dataset_raw = make_dataset(args.dataset_name, num_steps=args.num_steps)

    perturb_step = int(args.num_steps * args.perturb_frac)
    print(f"Dataset: {len(dataset)} samples, perturbation at step {perturb_step}")

    all_results = {}
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

            pixels = sample["pixels"]
            actions = sample["action"]
            raw_px = sample_raw["pixels"]
            T = pixels.shape[0]
            if T < args.history_size + 3 or perturb_step >= T:
                continue

            s_normal = compute_surprise_curve(
                model, pixels, actions, args.history_size, args.device)

            vis_raw = raw_px.clone()
            vis_raw[perturb_step] = color_shift_frame(vis_raw[perturb_step])
            vis_px = preprocess_frames(vis_raw)
            s_visual = compute_surprise_curve(
                model, vis_px, actions, args.history_size, args.device)

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
            print(f"  No valid trajectories for {label}")
            del model
            torch.cuda.empty_cache()
            continue

        all_results[label] = {
            "unperturbed": unperturbed_all,
            "visual": visual_all,
            "physical": physical_all,
        }
        print(f"  {n_ok} trajectories processed")
        del model
        torch.cuda.empty_cache()

    if not all_results:
        print("No results!")
        return

    # Plot
    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (label, results) in zip(axes, all_results.items()):
        max_len = max(len(c) for c in results["unperturbed"])

        def pad_mean(curves):
            arr = np.full((len(curves), max_len), np.nan)
            for i, c in enumerate(curves):
                arr[i, :len(c)] = c
            m = np.nanmean(arr, axis=0)
            n_valid = np.count_nonzero(~np.isnan(arr), axis=0).clip(1)
            se = np.nanstd(arr, axis=0) / np.sqrt(n_valid)
            return m, se

        m_unp, se_unp = pad_mean(results["unperturbed"])
        m_vis, se_vis = pad_mean(results["visual"])
        m_phy, se_phy = pad_mean(results["physical"])
        x = np.arange(max_len) + args.history_size

        ax.plot(x, m_unp, label="Unperturbed", color="#4CAF50", lw=2)
        ax.fill_between(x, m_unp - se_unp, m_unp + se_unp, color="#4CAF50", alpha=0.15)
        ax.plot(x, m_vis, label="Visual", color="#FF9800", lw=2)
        ax.fill_between(x, m_vis - se_vis, m_vis + se_vis, color="#FF9800", alpha=0.15)
        ax.plot(x, m_phy, label="Physical", color="#F44336", lw=2)
        ax.fill_between(x, m_phy - se_phy, m_phy + se_phy, color="#F44336", alpha=0.15)

        if args.history_size <= perturb_step < args.history_size + max_len:
            ax.axvline(perturb_step, color="gray", ls="--", alpha=0.5)

        ax.set_title(label, fontsize=12)
        ax.set_xlabel("Timestep")
        ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Surprise (MSE)", fontsize=12)

    axes[-1].legend(fontsize=9)
    fig.suptitle("Violation-of-Expectation Surprise", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
