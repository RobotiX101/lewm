"""Behavioral Cloning with frozen JEPA encoder backbone + action chunking.

Predicts a chunk of K future actions from a single observation, providing
temporal structure without needing recurrent state.

Three-phase approach:
  1. Encode all frames through frozen JEPA (num_steps=2, fast)
  2. Build action chunks using temporal indices into cached action data
  3. Train MLP head on (emb, state, action_chunk) tuples

Usage:
    python train_bc.py \
        --checkpoint ~/.stable_worldmodel/lewm_epoch_100_object.ckpt \
        --dataset_name top_short_merged \
        --chunk_size 16 \
        --epochs 50 --batch_size 256 --lr 1e-3
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import DataLoader, TensorDataset

from lerobot_dataset import LeRobotDatasetWrapper


# Same preprocessing as JEPA training
_imagenet_mean = [0.485, 0.456, 0.406]
_imagenet_std = [0.229, 0.224, 0.225]
_img_transform = v2.Compose([
    v2.Normalize(mean=_imagenet_mean, std=_imagenet_std),
    v2.Resize(224, antialias=True),
])


class ActionHead(nn.Module):
    """MLP head that predicts a chunk of K future actions."""

    def __init__(self, embed_dim=192, state_dim=12, action_dim=12,
                 hidden_dim=256, chunk_size=16):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim + state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, emb, state):
        x = torch.cat([emb, state], dim=-1)
        out = self.net(x)
        return out.view(-1, self.chunk_size, self.action_dim)


def compute_col_stats(dataset, col, dim):
    """Compute mean/std for a column from cached data."""
    data = dataset.get_col_data(col)
    data = data[~np.isnan(data).any(axis=1)]
    mean = data.mean(axis=0)[:dim].astype(np.float32)
    std = data.std(axis=0)[:dim].astype(np.float32)
    return mean, std


def precompute_embeddings(dataset, jepa_model, device, batch_size=32):
    """Encode all frames through frozen JEPA encoder.

    Uses num_steps=2 (fast video loading). Returns per-frame data with
    global indices for action chunk construction.

    Returns:
        all_emb: (N, 192) float32 — CLS embeddings
        all_state: (N, 12) float32 — raw joint states
        all_action: (N, 12) float32 — raw 12-dim actions
        all_global_idx: (N,) int64 — global frame indices
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0,
    )

    all_emb = []
    all_state = []
    all_action = []
    all_global_idx = []

    n_batches = len(loader)
    offsets = dataset.offsets
    frameskip = dataset.frameskip

    for i, batch in enumerate(loader):
        pixels = batch["pixels"]    # (B, 2, 3, 480, 1920)
        actions = batch["action"]   # (B, 2, 60)
        states = batch["state"]     # (B, 2, 12)

        B, T = pixels.shape[:2]

        # Preprocess pixels
        px = pixels.reshape(B * T, *pixels.shape[2:])
        px = _img_transform(px)
        px_input = px.unsqueeze(1).to(device)

        # Encode through JEPA
        with torch.no_grad():
            info = jepa_model.encode({"pixels": px_input})
            emb = info["emb"][:, -1].cpu()  # (B*T, 192)

        # Raw state and action
        state = states.reshape(B * T, -1)               # (B*T, 12)
        action = actions[:, :, :12].reshape(B * T, -1)   # (B*T, 12)

        # Compute global frame indices for temporal chunk lookup
        for j in range(B):
            ep_idx, start = dataset.clip_indices[i * batch_size + j]
            base_global = offsets[ep_idx] + start
            for t in range(T):
                all_global_idx.append(base_global + t * frameskip)

        all_emb.append(emb)
        all_state.append(state)
        all_action.append(action)

        if (i + 1) % 200 == 0 or i == n_batches - 1:
            print(f"  Encoded {i+1}/{n_batches} batches "
                  f"({(i+1)*batch_size*2:,} frames)", flush=True)

    all_emb = torch.cat(all_emb, dim=0)
    all_state = torch.cat(all_state, dim=0)
    all_action = torch.cat(all_action, dim=0)
    all_global_idx = np.array(all_global_idx, dtype=np.int64)

    return all_emb, all_state, all_action, all_global_idx


def build_action_chunks(action_cache, global_indices, chunk_size,
                        episode_lengths, episode_offsets):
    """Build action chunks from cached action data using global indices.

    For each frame at global index g, the action chunk is:
        [action_cache[g], action_cache[g+1], ..., action_cache[g+K-1]]

    Frames near episode boundaries are filtered out.

    Args:
        action_cache: (N_total, action_dim) — raw actions for all frames
        global_indices: (N,) — global frame index for each training sample
        chunk_size: number of future actions per chunk
        episode_lengths: array of episode lengths
        episode_offsets: array of episode start offsets

    Returns:
        chunks: (M, K, action_dim) where M <= N (filtered)
        valid_mask: (N,) boolean — which samples have valid chunks
    """
    action_dim = action_cache.shape[1]
    N = len(global_indices)

    # Build max valid index per episode
    max_idx_per_ep = {}
    for ep in range(len(episode_lengths)):
        end = episode_offsets[ep] + episode_lengths[ep]
        max_idx_per_ep[ep] = end

    # For each global index, find its episode and check boundary
    valid_mask = np.ones(N, dtype=bool)
    for i in range(N):
        g = global_indices[i]
        # Find which episode this frame belongs to
        ep = np.searchsorted(episode_offsets, g, side='right') - 1
        if ep < 0:
            valid_mask[i] = False
            continue
        ep_end = episode_offsets[ep] + episode_lengths[ep]
        # Check if there are enough frames remaining for the chunk
        if g + chunk_size > ep_end:
            valid_mask[i] = False

    # Build chunks for valid samples
    valid_indices = global_indices[valid_mask]
    M = len(valid_indices)
    chunks = np.zeros((M, chunk_size, action_dim), dtype=np.float32)

    for k in range(chunk_size):
        chunks[:, k, :] = action_cache[valid_indices + k]

    return chunks, valid_mask


def main():
    parser = argparse.ArgumentParser(
        description="BC training with JEPA backbone + action chunking")
    parser.add_argument("--checkpoint",
                        default="~/.stable_worldmodel/lewm_epoch_100_object.ckpt")
    parser.add_argument("--dataset_root", default="~/Survey/Datasets")
    parser.add_argument("--dataset_name", default="top_short_merged")
    parser.add_argument("--output_dir", default="~/.stable_worldmodel")
    parser.add_argument("--chunk_size", type=int, default=16,
                        help="Number of future actions to predict per step")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--train_split", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_every", type=int, default=10)
    args = parser.parse_args()

    args.checkpoint = str(Path(args.checkpoint).expanduser())
    args.dataset_root = str(Path(args.dataset_root).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())
    device = args.device
    chunk_size = args.chunk_size

    # SwanLab setup
    import swanlab
    api_key = os.environ.get("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key=api_key)
    swanlab_run = swanlab.init(
        project="lewm",
        experiment_name=f"bc_chunk{chunk_size}_{args.dataset_name}",
        config={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "chunk_size": chunk_size,
            "dataset": args.dataset_name,
        },
    )

    # Load dataset with num_steps=2 (fast encoding, still loads action cache)
    dataset = LeRobotDatasetWrapper(
        root=Path(args.dataset_root) / args.dataset_name,
        frameskip=5,
        num_steps=2,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=[
            "observation.images.top_rgb",
            "observation.images.left_rgb",
            "observation.images.right_rgb",
        ],
        key_map={"state": "observation.state"},
        transform=None,
    )

    # Compute normalization stats
    action_mean, action_std = compute_col_stats(dataset, "action", 12)
    state_mean, state_std = compute_col_stats(dataset, "state", 12)
    print(f"Action mean: {action_mean[:4]}...")
    print(f"Action std:  {action_std[:4]}...")
    print(f"State  mean: {state_mean[:4]}...")
    print(f"State  std:  {state_std[:4]}...")
    print(f"Dataset: {len(dataset)} clips -> ~{len(dataset) * 2} frames",
          flush=True)

    # Load JEPA model
    print(f"\nLoading JEPA model from {args.checkpoint}", flush=True)
    jepa_model = torch.load(args.checkpoint, map_location=device,
                            weights_only=False)
    jepa_model.eval()

    # Phase 1: Pre-compute all embeddings (fast, same as single-step BC)
    print("\n=== Phase 1: Pre-computing embeddings ===", flush=True)
    t0 = time.time()
    all_emb, all_state, all_action, all_global_idx = precompute_embeddings(
        dataset, jepa_model, device, batch_size=32,
    )
    encode_time = time.time() - t0
    print(f"Encoded {len(all_emb):,} frames in {encode_time:.0f}s")
    print(f"  emb shape: {all_emb.shape}")
    print(f"  state shape: {all_state.shape}")
    print(f"  action shape: {all_action.shape}", flush=True)

    # Free JEPA model from GPU
    del jepa_model
    torch.cuda.empty_cache()

    # Phase 2: Build action chunks from cached action data
    print(f"\n=== Phase 2: Building action chunks (K={chunk_size}) ===",
          flush=True)
    t0 = time.time()

    # Get raw action data from cache (12-dim actions for all frames)
    raw_actions = dataset.get_col_data("action")[:, :12]  # (N_total, 12)

    # Filter out NaN rows
    nan_mask = np.isnan(raw_actions).any(axis=1)
    raw_actions[nan_mask] = 0.0

    action_chunks, valid_mask = build_action_chunks(
        raw_actions, all_global_idx, chunk_size,
        dataset.lengths, dataset.offsets,
    )

    # Filter embeddings/states to match valid chunks
    all_emb = all_emb[valid_mask]
    all_state = all_state[valid_mask]
    all_action = all_action[valid_mask]

    chunk_time = time.time() - t0
    print(f"Built {len(action_chunks):,} action chunks in {chunk_time:.1f}s")
    print(f"  Filtered {len(valid_mask) - valid_mask.sum()} frames "
          f"near episode boundaries")
    print(f"  action_chunks shape: {action_chunks.shape}", flush=True)

    # Convert action chunks to tensor
    action_chunks_t = torch.from_numpy(action_chunks)  # (N, K, 12)

    # Normalize state and actions (same stats for every step in chunk)
    state_mean_t = torch.from_numpy(state_mean)
    state_std_t = torch.from_numpy(state_std + 1e-8)
    action_mean_t = torch.from_numpy(action_mean)
    action_std_t = torch.from_numpy(action_std + 1e-8)

    all_state_norm = (all_state - state_mean_t) / state_std_t
    # Normalize each action in the chunk with same per-dim mean/std
    action_chunks_norm = (
        action_chunks_t - action_mean_t.unsqueeze(0).unsqueeze(0)
    ) / (action_std_t.unsqueeze(0).unsqueeze(0) + 1e-8)

    # Flatten action chunks for TensorDataset: (N, K*12)
    action_chunks_flat = action_chunks_norm.reshape(len(all_emb), -1)

    # Split into train/val
    n = len(all_emb)
    n_train = int(n * args.train_split)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_dataset = TensorDataset(
        all_emb[train_idx], all_state_norm[train_idx],
        action_chunks_flat[train_idx],
    )
    val_dataset = TensorDataset(
        all_emb[val_idx], all_state_norm[val_idx],
        action_chunks_flat[val_idx],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
    )

    # Phase 3: Train MLP head
    print(f"\n=== Phase 3: Training MLP head (chunk_size={chunk_size}) ===",
          flush=True)
    head = ActionHead(
        hidden_dim=args.hidden_dim, chunk_size=chunk_size,
    ).to(device)
    head_params = sum(p.numel() for p in head.parameters())
    print(f"MLP head params: {head_params:,}")
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}",
          flush=True)

    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        head.train()
        train_losses = []

        for emb_b, state_b, action_b in train_loader:
            emb_b = emb_b.to(device)
            state_b = state_b.to(device)
            action_b = action_b.to(device)

            action_target = action_b.view(-1, chunk_size, 12)
            pred = head(emb_b, state_b)  # (B, K, 12)
            loss = criterion(pred, action_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        epoch_time = time.time() - t0

        # Validation
        head.eval()
        val_losses = []
        with torch.no_grad():
            for emb_b, state_b, action_b in val_loader:
                emb_b = emb_b.to(device)
                state_b = state_b.to(device)
                action_b = action_b.to(device)

                action_target = action_b.view(-1, chunk_size, 12)
                pred = head(emb_b, state_b)
                loss = criterion(pred, action_target)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        lr = scheduler.get_last_lr()[0]

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train={train_loss:.6f}  val={val_loss:.6f}  "
              f"lr={lr:.6f}  time={epoch_time:.1f}s", flush=True)

        # Log to SwanLab
        swanlab.log({
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/lr": lr,
            "train/epoch_time_s": epoch_time,
        }, step=epoch)

        # Save best (use .tolist() for cross-venv numpy compatibility)
        ckpt_data = {
            "head_state_dict": head.state_dict(),
            "hidden_dim": args.hidden_dim,
            "embed_dim": 192,
            "state_dim": 12,
            "action_dim": 12,
            "chunk_size": chunk_size,
            "action_mean": action_mean.tolist(),
            "action_std": action_std.tolist(),
            "state_mean": state_mean.tolist(),
            "state_std": state_std.tolist(),
            "epoch": epoch,
            "val_loss": val_loss,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = Path(args.output_dir) / f"bc_chunk{chunk_size}_best.pt"
            torch.save(ckpt_data, save_path, pickle_protocol=2)
            print(f"  -> Best model saved (val={val_loss:.6f})", flush=True)

        # Periodic checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            save_path = (Path(args.output_dir)
                         / f"bc_chunk{chunk_size}_epoch_{epoch}.pt")
            torch.save(ckpt_data, save_path, pickle_protocol=2)

    swanlab_run.finish()
    print(f"\nDone. Best val loss: {best_val_loss:.6f}")
    print(f"Saved to {args.output_dir}/bc_chunk{chunk_size}_best.pt")


if __name__ == "__main__":
    main()
