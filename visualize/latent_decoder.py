"""Lightweight transformer decoder: 192-dim CLS embedding → reconstructed image.

Takes the frozen LeWM encoder's CLS-token embedding (192-dim) and decodes it
back to pixel space via cross-attention with learnable query tokens (one per
spatial patch), following Appendix D of the LeWorldModel paper.

Architecture:
  CLS (192) → linear project to hidden_dim (384)
  Learnable queries: (num_patches, hidden_dim) — one per spatial patch
  N transformer layers with cross-attention (queries attend to projected CLS)
  Output MLP per query → (patch_size² × channels) pixel values
  Unpatchify → (3, 224, W) image

Usage:
    python visualize/latent_decoder.py train \
        --checkpoint ~/.stable_worldmodel/lewm_epoch_99_object.ckpt \
        --dataset_name top_short_merged \
        --output_dir visualize/decoder_checkpoints

    python visualize/latent_decoder.py visualize \
        --checkpoint ~/.stable_worldmodel/lewm_epoch_99_object.ckpt \
        --decoder_path visualize/decoder_checkpoints/best_decoder.pt \
        --dataset_name top_short_merged \
        --output visualize/rollout_decoded.png
"""

import argparse
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lerobot_dataset import LeRobotDatasetWrapper
from utils import get_img_preprocessor

CAMERA_KEYS = [
    "observation.images.top_rgb",
    "observation.images.left_rgb",
    "observation.images.right_rgb",
]


# ---------------------------------------------------------------------------
# Decoder Model
# ---------------------------------------------------------------------------

class PatchDecoder(nn.Module):
    """Cross-attention decoder: CLS embedding → pixel image.

    Args:
        cls_dim: dimension of the CLS embedding (192)
        hidden_dim: internal transformer dimension (384)
        num_heads: number of attention heads
        num_layers: number of transformer decoder layers
        patch_size: spatial patch size (14)
        img_h: image height (224)
        img_w: image width (896)
        channels: number of image channels (3)
    """

    def __init__(
        self,
        cls_dim: int = 192,
        hidden_dim: int = 384,
        num_heads: int = 6,
        num_layers: int = 4,
        patch_size: int = 14,
        img_h: int = 224,
        img_w: int = 896,
        channels: int = 3,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.img_h = img_h
        self.img_w = img_w
        self.num_patches_h = img_h // patch_size
        self.num_patches_w = img_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        self.patch_dim = patch_size * patch_size * channels

        # Project CLS embedding to hidden_dim
        self.cls_proj = nn.Linear(cls_dim, hidden_dim)

        # Learnable query tokens — one per spatial patch
        self.query_tokens = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Positional embedding for spatial layout
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, hidden_dim) * 0.02
        )

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

        # Per-patch output head: hidden_dim → patch_dim
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.patch_dim),
        )

    def forward(self, cls_emb: torch.Tensor) -> torch.Tensor:
        """Decode CLS embedding to pixel image.

        Args:
            cls_emb: (B, D) CLS embeddings

        Returns:
            (B, C, H, W) reconstructed image
        """
        B = cls_emb.shape[0]

        # Project CLS → (B, 1, hidden_dim) as memory for cross-attention
        memory = self.cls_proj(cls_emb).unsqueeze(1)  # (B, 1, hidden_dim)

        # Query tokens + positional embedding
        queries = self.query_tokens.expand(B, -1, -1)  # (B, N, hidden_dim)
        queries = queries + self.pos_embed  # add spatial position info

        # Transformer decoder: queries attend to memory
        for layer in self.layers:
            queries = layer(tgt=queries, memory=memory)
        queries = self.norm(queries)  # (B, N, hidden_dim)

        # Project to patches
        patches = self.output_head(queries)  # (B, N, patch_dim)

        # Unpatchify
        patches = patches.reshape(
            B, self.num_patches_h, self.num_patches_w,
            self.patch_size, self.patch_size, self.channels,
        )
        images = patches.permute(0, 5, 1, 3, 2, 4)  # (B, C, H_ph, P, W_pw, P)
        images = images.reshape(B, self.channels, self.img_h, self.img_w)

        return images


# ---------------------------------------------------------------------------
# Dataset for training the decoder
# ---------------------------------------------------------------------------

class EmbeddingImageDataset(Dataset):
    """Pre-computed (CLS_embedding, target_image) pairs from LeWM encoder."""

    def __init__(self, embeddings: torch.Tensor, images: torch.Tensor):
        self.embeddings = embeddings  # (N, 192)
        self.images = images          # (N, 3, 224, 896)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.images[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_decoder(args):
    """Train the patch decoder on frozen LeWM embeddings."""
    from tqdm import tqdm

    device = args.device

    # 1. Load frozen model
    print(f"Loading LeWM model from {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2. Create dataset
    print(f"Loading dataset: {args.dataset_name}")
    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = LeRobotDatasetWrapper(
        root=Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))) / args.dataset_name,
        frameskip=5, num_steps=4,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )
    print(f"Dataset: {len(dataset)} samples")

    # 3. Pre-compute embeddings (or load from cache)
    cache_path = Path(args.output_dir) / "embedding_cache.pt"
    if cache_path.exists() and not args.no_cache:
        print(f"Loading cached embeddings from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        all_emb = cache["embeddings"]
        all_img = cache["images"]
    else:
        print("Computing embeddings from frozen encoder...")
        all_emb, all_img = [], []
        n_max = min(len(dataset), args.max_samples)

        for i in tqdm(range(n_max), desc="Encoding"):
            try:
                sample = dataset[i]
            except Exception:
                continue
            pixels = sample["pixels"]  # (T, 3, 224, 896)
            actions = sample["action"]  # (T, 60)

            with torch.no_grad():
                pix = pixels.unsqueeze(0).to(device)
                act = actions.unsqueeze(0).to(device)
                info = model.encode({"pixels": pix, "action": act})
                emb = info["emb"]  # (1, T, 192)

            # Collect per-timestep embeddings and images
            for t in range(pixels.shape[0]):
                all_emb.append(emb[0, t].cpu())
                all_img.append(pixels[t])

        all_emb = torch.stack(all_emb)
        all_img = torch.stack(all_img)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": all_emb, "images": all_img}, cache_path)
        print(f"Cached {len(all_emb)} embeddings to {cache_path}")

    # Free encoder + dataset from GPU memory
    del model
    del dataset
    torch.cuda.empty_cache()
    import gc; gc.collect()

    print(f"Total samples: {len(all_emb)}, image shape: {all_img.shape[1:]}")

    # 4. Split train/val
    n = len(all_emb)
    perm = torch.randperm(n)
    n_val = max(1, n // 10)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = EmbeddingImageDataset(all_emb[train_idx], all_img[train_idx])
    val_ds = EmbeddingImageDataset(all_emb[val_idx], all_img[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=2)

    # 5. Create decoder
    decoder = PatchDecoder(
        cls_dim=192,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patch_size=14,
        img_h=224,
        img_w=all_img.shape[-1],
        channels=3,
    ).to(device)

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"Decoder params: {n_params:,}")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 6. Training loop
    best_val_loss = float("inf")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Train
        decoder.train()
        train_loss = 0
        for emb_batch, img_batch in train_loader:
            emb_batch = emb_batch.to(device)
            img_batch = img_batch.to(device)

            pred = decoder(emb_batch)
            loss = F.mse_loss(pred, img_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(emb_batch)

        train_loss /= len(train_ds)
        scheduler.step()

        # Validate
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for emb_batch, img_batch in val_loader:
                pred = decoder(emb_batch.to(device))
                val_loss += F.mse_loss(pred, img_batch.to(device)).item() * len(emb_batch)
        val_loss /= len(val_ds)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(decoder.state_dict(), output_dir / "best_decoder.pt")
            print(f"  → Saved best decoder (val_loss={val_loss:.6f})")

        # Periodic reconstruction check
        if epoch % args.sample_every == 0 or epoch == args.epochs:
            _save_reconstruction_sample(decoder, val_ds, device, output_dir / f"recon_epoch{epoch}.png")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Decoder saved to {output_dir / 'best_decoder.pt'}")


def _save_reconstruction_sample(decoder, dataset, device, path):
    """Save a side-by-side reconstruction sample."""
    decoder.eval()
    emb, img = dataset[0]
    with torch.no_grad():
        pred = decoder(emb.unsqueeze(0).to(device))[0].cpu()

    img_np = img.numpy().transpose(1, 2, 0)
    pred_np = pred.numpy().transpose(1, 2, 0)

    # Denormalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean).clip(0, 1)
    pred_np = (pred_np * std + mean).clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    axes[1].imshow(pred_np)
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Visualization: Figure 7/8 rollout
# ---------------------------------------------------------------------------

def visualize_rollout(args):
    """Generate Figure 7/8 style rollout visualization.

    1. Encode a trajectory through the frozen LeWM encoder
    2. Autoregressively predict future embeddings (rollout)
    3. Decode both actual and predicted embeddings to images
    4. Display as a grid: rows = ground truth / predicted / difference
    """
    device = args.device

    # Load frozen model
    print(f"Loading LeWM model from {args.checkpoint}")
    model = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.eval()

    # Load decoder
    print(f"Loading decoder from {args.decoder_path}")
    # Get image width from a sample to initialize decoder
    transform = get_img_preprocessor("pixels", "pixels", 224)
    dataset = LeRobotDatasetWrapper(
        root=Path(os.environ.get("DATASET_ROOT", str(Path.home() / "Datasets"))) / args.dataset_name,
        frameskip=5, num_steps=args.num_steps,
        keys_to_load=["pixels", "action", "state"],
        keys_to_cache=["action", "state"],
        camera_keys=CAMERA_KEYS,
        key_map={"state": "observation.state"},
        transform=transform,
    )
    sample = dataset[0]
    img_w = sample["pixels"].shape[-1]

    decoder = PatchDecoder(
        cls_dim=192,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patch_size=14,
        img_h=224,
        img_w=img_w,
        channels=3,
    ).to(device)
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device, weights_only=False))
    decoder.eval()

    # Pick a trajectory
    idx = args.trajectory_idx
    sample = dataset[idx]
    pixels = sample["pixels"]  # (T, 3, 224, 896)
    actions = sample["action"]  # (T, 60)
    T = pixels.shape[0]
    history_size = args.history_size
    print(f"Trajectory: T={T}, history_size={history_size}")

    # Encode full trajectory
    with torch.no_grad():
        pix = pixels.unsqueeze(0).to(device)
        act = actions.unsqueeze(0).to(device)
        info = model.encode({"pixels": pix, "action": act})
        emb = info["emb"]          # (1, T, 192)
        act_emb = info["act_emb"]  # (1, T, D_act)

    # Autoregressive rollout starting from history
    pred_embs = []
    current_emb = emb[:, :history_size].clone()
    current_act = act_emb[:, :history_size].clone()

    for t in range(history_size, T):
        with torch.no_grad():
            pred = model.predict(current_emb[:, -history_size:],
                                 current_act[:, -history_size:])[:, -1:]  # (1, 1, 192)
        pred_embs.append(pred[0, 0].cpu())

        # For next step, use actual action embedding but predicted obs
        current_emb = torch.cat([current_emb, pred], dim=1)
        if t < T:
            current_act = torch.cat([current_act, act_emb[:, t:t+1]], dim=1)

    pred_embs = torch.stack(pred_embs)  # (T - history_size, 192)
    actual_embs = emb[0, history_size:].cpu()  # (T - history_size, 192)

    # Decode all embeddings to images
    print("Decoding actual embeddings...")
    with torch.no_grad():
        actual_imgs = decoder(actual_embs.to(device)).cpu()  # (N, 3, 224, 896)

    print("Decoding predicted embeddings...")
    with torch.no_grad():
        pred_imgs = decoder(pred_embs.to(device)).cpu()  # (N, 3, 224, 896)

    # Also decode context frames
    context_embs = emb[0, :history_size].cpu()
    with torch.no_grad():
        context_imgs = decoder(context_embs.to(device)).cpu()

    # Denormalize helper
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def denorm(t):
        return (t * std + mean).clip(0, 1)

    # Build grid: 3 rows × N columns
    N = len(actual_imgs)
    n_total = history_size + N
    fig_cols = min(n_total, args.max_cols)
    fig, axes = plt.subplots(3, fig_cols, figsize=(4 * fig_cols, 12))

    # ImageNet denormalize
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np = np.array([0.229, 0.224, 0.225])

    def to_numpy(t):
        return (t.numpy().transpose(1, 2, 0) * std_np + mean_np).clip(0, 1)

    for col in range(fig_cols):
        if col < history_size:
            # Context frames
            img_gt = to_numpy(pixels[col])
            img_ctx = to_numpy(context_imgs[col])
            img_pred = img_ctx  # context is the same for both
            title = f"t={col}\n(context)"
        else:
            i = col - history_size
            if i >= N:
                break
            img_gt = to_numpy(pixels[history_size + i])
            img_pred = to_numpy(pred_imgs[i])
            title = f"t={history_size + i}"

        # Row 0: Ground truth
        axes[0, col].imshow(img_gt)
        axes[0, col].set_title(title, fontsize=8)
        axes[0, col].axis("off")

        # Row 1: Predicted
        axes[1, col].imshow(img_pred)
        axes[1, col].axis("off")

        # Row 2: Absolute difference
        diff = np.abs(img_gt - img_pred).mean(axis=-1)
        axes[2, col].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, col].axis("off")

    axes[0, 0].set_ylabel("Ground Truth", fontsize=11)
    axes[1, 0].set_ylabel("Predicted (decoded)", fontsize=11)
    axes[2, 0].set_ylabel("|Error|", fontsize=11)

    fig.suptitle("Latent Rollout Visualization (Figure 7/8)", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {args.output}")

    # Also save a per-camera crop comparison for clarity
    _save_camera_crops(pixels, pred_imgs, history_size, args)


def _save_camera_crops(pixels, pred_imgs, history_size, args):
    """Save a version showing individual camera crops (top, left, right)."""
    mean_np = np.array([0.485, 0.456, 0.406])
    std_np = np.array([0.229, 0.224, 0.225])

    def to_numpy(t):
        return (t.numpy().transpose(1, 2, 0) * std_np + mean_np).clip(0, 1)

    N = len(pred_imgs)
    # Pick 4 evenly-spaced timesteps to display
    indices = np.linspace(0, N - 1, min(4, N), dtype=int)

    camera_names = ["top_rgb", "left_rgb", "right_rgb"]
    W = pixels.shape[-1]
    cam_w = W // 3

    fig, axes = plt.subplots(len(camera_names), len(indices), figsize=(3 * len(indices), 3 * len(camera_names)))
    if len(indices) == 1:
        axes = axes[:, None]

    for cam_idx, cam_name in enumerate(camera_names):
        for col, pred_i in enumerate(indices):
            t = history_size + pred_i
            img_gt = to_numpy(pixels[t])
            img_pred = to_numpy(pred_imgs[pred_i])

            # Crop camera region
            x0 = cam_idx * cam_w
            x1 = (cam_idx + 1) * cam_w

            # Side by side: GT (left) / Pred (right)
            combined = np.concatenate([img_gt[:, x0:x1], img_pred[:, x0:x1]], axis=0)
            axes[cam_idx, col].imshow(combined)
            axes[cam_idx, col].set_title(f"{cam_name} t={t}", fontsize=9)
            axes[cam_idx, col].axis("off")

    fig.suptitle("Per-Camera: Top=GT, Bottom=Predicted", fontsize=12)
    fig.tight_layout()
    cam_path = Path(args.output).with_stem(Path(args.output).stem + "_cameras")
    fig.savefig(cam_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Camera crops saved to {cam_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LeWM Latent Decoder")
    sub = parser.add_subparsers(dest="command")

    # Train subcommand
    train_p = sub.add_parser("train", help="Train the decoder")
    train_p.add_argument("--checkpoint", required=True, help="Path to LeWM _object.ckpt")
    train_p.add_argument("--dataset_name", default="top_short_merged")
    train_p.add_argument("--output_dir", default="visualize/decoder_checkpoints")
    train_p.add_argument("--max_samples", type=int, default=5000)
    train_p.add_argument("--epochs", type=int, default=50)
    train_p.add_argument("--batch_size", type=int, default=16)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--hidden_dim", type=int, default=384)
    train_p.add_argument("--num_heads", type=int, default=6)
    train_p.add_argument("--num_layers", type=int, default=4)
    train_p.add_argument("--sample_every", type=int, default=5)
    train_p.add_argument("--device", default="cuda")
    train_p.add_argument("--no_cache", action="store_true")

    # Visualize subcommand
    vis_p = sub.add_parser("visualize", help="Generate Figure 7/8 rollout viz")
    vis_p.add_argument("--checkpoint", required=True)
    vis_p.add_argument("--decoder_path", required=True)
    vis_p.add_argument("--dataset_name", default="top_short_merged")
    vis_p.add_argument("--num_steps", type=int, default=16)
    vis_p.add_argument("--history_size", type=int, default=3)
    vis_p.add_argument("--trajectory_idx", type=int, default=0)
    vis_p.add_argument("--max_cols", type=int, default=10)
    vis_p.add_argument("--hidden_dim", type=int, default=384)
    vis_p.add_argument("--num_heads", type=int, default=6)
    vis_p.add_argument("--num_layers", type=int, default=4)
    vis_p.add_argument("--device", default="cuda")
    vis_p.add_argument("--output", default="visualize/rollout_decoded.png")

    args = parser.parse_args()

    if args.command == "train":
        train_decoder(args)
    elif args.command == "visualize":
        visualize_rollout(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
