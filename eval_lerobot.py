"""Evaluation for LeWM on LeRobot datasets.

Supports two modes:
  --mode predict   Prediction quality: MSE/cosine-sim between predicted and
                   actual future embeddings (offline, fast).
  --mode cem       CEM planning with trajectory-matching cost: CEM searches for
                   actions whose predicted future embeddings match the actual
                   encoded future frames.

Usage:
    # Prediction quality (fast)
    python eval_lerobot.py --mode predict --num_eval 50

    # CEM planning (slow)
    python eval_lerobot.py --mode cem --num_eval 10 --num_samples 200 --cem_iterations 20
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2

from lerobot_dataset import LeRobotDatasetWrapper


# ---------------------------------------------------------------------------
# Model & data loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: str) -> torch.nn.Module:
    model = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.eval()
    model.requires_grad_(False)
    return model


def load_dataset(args) -> LeRobotDatasetWrapper:
    root = Path(args.dataset_root) / args.dataset_name
    return LeRobotDatasetWrapper(
        root=root,
        frameskip=args.frameskip,
        num_steps=args.history_size + args.num_preds,
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


def compute_action_stats(dataset):
    data = dataset.get_col_data("action")
    data = data[~np.isnan(data).any(axis=1)]
    mean = torch.from_numpy(data.mean(axis=0)).float()
    std = torch.from_numpy(data.std(axis=0)).float()
    return mean, std


# ---------------------------------------------------------------------------
# Preprocessing (must match training exactly)
# ---------------------------------------------------------------------------

_imagenet_mean = [0.485, 0.456, 0.406]
_imagenet_std = [0.229, 0.224, 0.225]
_img_transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=_imagenet_mean, std=_imagenet_std),
    v2.Resize(224, antialias=True),
])


def preprocess_images(pixel_tensor):
    """(T, 3, H, 3*W) → (T, 3, 224, 672) float32 normalized."""
    T = pixel_tensor.shape[0]
    return torch.stack([_img_transform(pixel_tensor[i]) for i in range(T)])


def normalize_actions(actions, mean, std):
    return (actions - mean) / std


# ---------------------------------------------------------------------------
# Episode sampling
# ---------------------------------------------------------------------------

def sample_eval_episodes(dataset, args):
    rng = np.random.default_rng(args.seed)
    episodes = []
    num_episodes = len(dataset.episodes)
    for _ in range(args.num_eval * 5):
        ep_idx = int(rng.integers(0, num_episodes))
        ep_len = int(dataset.lengths[ep_idx])
        min_start = args.history_size * args.frameskip
        max_start = ep_len - args.goal_offset - 1
        if max_start <= min_start:
            continue
        start_idx = int(rng.integers(min_start, max_start))
        goal_idx = start_idx + args.goal_offset
        episodes.append((ep_idx, start_idx, goal_idx))
        if len(episodes) >= args.num_eval:
            break
    return episodes[: args.num_eval]


def sample_predict_sequences(dataset, args, num_sequences):
    HS = args.history_size
    NP = args.num_preds
    FS = args.frameskip
    span = (HS + NP) * FS

    rng = np.random.default_rng(args.seed)
    seqs = []
    num_episodes = len(dataset.episodes)
    for _ in range(num_sequences * 5):
        ep_idx = int(rng.integers(0, num_episodes))
        ep_len = int(dataset.lengths[ep_idx])
        if ep_len < span + 1:
            continue
        start = int(rng.integers(0, ep_len - span))
        seqs.append((ep_idx, start))
        if len(seqs) >= num_sequences:
            break
    return seqs[:num_sequences]


# ---------------------------------------------------------------------------
# Prediction quality evaluation
# ---------------------------------------------------------------------------

def evaluate_prediction_quality(model, dataset, args, action_mean, action_std):
    HS = args.history_size
    NP = args.num_preds
    FS = args.frameskip
    T = HS + NP
    device = args.device

    sequences = sample_predict_sequences(dataset, args, args.num_eval)
    print(f"Prediction quality eval: {len(sequences)} sequences, "
          f"history_size={HS}, num_preds={NP}, frameskip={FS}")

    all_mse = []
    all_cos = []

    for batch_start in range(0, len(sequences), args.batch_size):
        batch_seqs = sequences[batch_start:batch_start + args.batch_size]
        B = len(batch_seqs)

        all_pixels = []
        all_actions = []
        for ep_idx, start in batch_seqs:
            sl = dataset._load_slice(ep_idx, start, start + T * FS)
            px = preprocess_images(sl["pixels"])
            act = normalize_actions(sl["action"], action_mean, action_std)
            act = act.reshape(T, -1)
            all_pixels.append(px)
            all_actions.append(act)

        pixels = torch.stack(all_pixels).to(device)
        actions = torch.stack(all_actions).to(device)

        with torch.no_grad():
            output = model.encode({"pixels": pixels, "action": actions})
            emb = output["emb"]
            act_emb = output["act_emb"]

            ctx_emb = emb[:, :HS]
            ctx_act = act_emb[:, :HS]
            tgt_emb = emb[:, NP:]
            pred_emb = model.predict(ctx_emb, ctx_act)

            mse = (pred_emb - tgt_emb).pow(2).mean(dim=-1)
            all_mse.append(mse.cpu())

            cos = F.cosine_similarity(pred_emb, tgt_emb, dim=-1)
            all_cos.append(cos.cpu())

    mse = torch.cat(all_mse, dim=0).numpy()
    cos = torch.cat(all_cos, dim=0).numpy()

    print(f"\n{'='*60}")
    print(f"PREDICTION QUALITY  ({len(sequences)} sequences)")
    print(f"{'='*60}")
    print(f"Overall MSE:            {mse.mean():.6f}")
    print(f"Overall Cosine Sim:     {cos.mean():.4f}")
    print()
    for t in range(HS):
        label = f"  t+{NP+t}" if NP > 0 else f"  t+{t+1}"
        print(f"  {label}: MSE={mse[:, t].mean():.6f}  "
              f"Cos={cos[:, t].mean():.4f}")

    return {
        "config": vars(args),
        "summary": {
            "num_sequences": len(sequences),
            "overall_mse": float(mse.mean()),
            "overall_cosine_sim": float(cos.mean()),
        },
    }


# ---------------------------------------------------------------------------
# CEM Solver (searches in 12-dim action space)
# ---------------------------------------------------------------------------

class SimpleCEMSolver:
    """CEM solver that searches in 12-dim action space and expands to 60-dim
    internally before calling model.get_cost().

    Key design choices:
      - Searches in raw 12-dim space (not 60-dim), reducing dimensionality
        from horizon*60 to horizon*12.
      - Supports warm-starting from an initial action sequence.
      - Applies variance decay for convergence.
    """

    def __init__(self, model, horizon, action_dim, frameskip,
                 num_samples=300, n_iterations=30, topk=30,
                 var_scale=1.0, var_decay=0.95, min_var=0.01,
                 device="cuda", seed=42):
        self.model = model
        self.horizon = horizon
        self.action_dim = action_dim
        self.frameskip = frameskip
        self.num_samples = num_samples
        self.n_iterations = n_iterations
        self.topk = min(topk, num_samples)
        self.var_scale = var_scale
        self.var_decay = var_decay
        self.min_var = min_var
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def _expand(self, c12):
        """(B, S, H, 12) -> (B, S, H, 60) repeat each action frameskip times."""
        return c12.repeat_interleave(self.frameskip, dim=-1)

    def solve(self, info_dict, init_actions=None):
        """Standard CEM using model.get_cost()."""
        dev = self.device
        H, S, D = self.horizon, self.num_samples, self.action_dim

        mean = init_actions.to(dev) if init_actions is not None else torch.zeros(1, H, D, device=dev)
        var = self.var_scale * torch.ones(1, H, D, device=dev)

        cost_history = []
        for it in range(self.n_iterations):
            noise = torch.randn(1, S, H, D, generator=self.gen, device=dev)
            c12 = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            c12[:, 0] = mean
            c60 = self._expand(c12)

            cur = {k: v.expand(1, S, *v.shape[2:]) if torch.is_tensor(v) else v
                   for k, v in info_dict.items()}
            costs = self.model.get_cost(cur, c60)

            _, topk_idx = torch.topk(costs, self.topk, dim=1, largest=False)
            bidx = torch.arange(1, device=dev).unsqueeze(1).expand(-1, self.topk)
            elites = c12[bidx, topk_idx]

            mean = elites.mean(dim=1)
            var = elites.std(dim=1).clamp(min=self.min_var)
            if it > 5:
                var = var * self.var_decay

            cost_history.append(costs[0, topk_idx[0, 0]].item())

        return {"actions": mean.detach().cpu(), "costs": cost_history}

    def solve_trajectory(self, info_dict, target_emb, init_actions=None):
        """CEM with trajectory-matching cost.

        Encodes history once, then for each candidate action sequence:
          1. Encode candidate actions via action_encoder
          2. Run predict() to get predicted embeddings
          3. Cost = MSE(predicted_emb, target_emb from actual future pixels)

        This measures whether CEM finds actions the model "agrees" produce
        embeddings matching real future frames.
        """
        dev = self.device
        H, S, D = self.horizon, self.num_samples, self.action_dim
        HS = info_dict["pixels"].shape[2]

        mean = init_actions.to(dev) if init_actions is not None else torch.zeros(1, H, D, device=dev)
        var = self.var_scale * torch.ones(1, H, D, device=dev)

        # Encode history once
        with torch.no_grad():
            hist_info = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
            hist_info = self.model.encode(hist_info)
            hist_emb = hist_info["emb"]       # (1, HS, D_emb)
            hist_act_emb = hist_info["act_emb"]  # (1, HS, D_act_emb)

        target = target_emb.to(dev)

        cost_history = []
        for it in range(self.n_iterations):
            noise = torch.randn(1, S, H, D, generator=self.gen, device=dev)
            c12 = noise * var.unsqueeze(1) + mean.unsqueeze(1)
            c12[:, 0] = mean
            c60 = self._expand(c12)

            costs = []
            for s in range(S):
                act_60 = c60[0, s]  # (H, 60)
                with torch.no_grad():
                    act_emb_s = self.model.action_encoder(act_60.unsqueeze(0))  # (1, H, D_act_emb)
                # Concatenate history + candidate for prediction context
                all_act_emb = torch.cat([hist_act_emb, act_emb_s], dim=1)  # (1, HS+H, D_act_emb)
                # For prediction, take the last HS entries
                pred = self.model.predict(hist_emb, all_act_emb[:, -HS:])  # (1, HS, D_emb)
                # We want the prediction to match the actual future embeddings
                cost = F.mse_loss(pred[:, -1, :], target[:, -1, :]).item()
                costs.append(cost)

            costs_t = torch.tensor(costs, device=dev).unsqueeze(0)

            _, topk_idx = torch.topk(costs_t, self.topk, dim=1, largest=False)
            bidx = torch.arange(1, device=dev).unsqueeze(1).expand(-1, self.topk)
            elites = c12[bidx, topk_idx]

            mean = elites.mean(dim=1)
            var = elites.std(dim=1).clamp(min=self.min_var)
            if it > 5:
                var = var * self.var_decay

            cost_history.append(costs_t[0, topk_idx[0, 0]].item())

        return {"actions": mean.detach().cpu(), "costs": cost_history}


# ---------------------------------------------------------------------------
# Episode evaluation (CEM mode)
# ---------------------------------------------------------------------------

def evaluate_episode(model, solver, dataset, ep_idx, start_idx, goal_idx,
                     args, action_mean, action_std):
    """CEM eval with trajectory-matching cost.

    Measures:
      - oracle_cost: predict with GT actions, MSE vs actual future emb
      - cem_warm_cost: CEM with GT warm start
      - cem_zero_cost: CEM from zero init
      - zero_cost: predict with zero actions, MSE vs actual future emb
    """
    device = args.device
    HS = args.history_size
    FS = args.frameskip
    H = solver.horizon
    AD = args.action_dim

    hist_end = start_idx + 1
    hist_start = hist_end - HS * FS

    # Load history + future frames
    total_frames = HS + H  # at frameskip intervals
    total_raw = total_frames * FS
    sl = dataset._load_slice(ep_idx, hist_start, hist_start + total_raw)
    all_px = preprocess_images(sl["pixels"])[:total_frames]  # (HS+H, 3, 224, 672)
    all_act = normalize_actions(sl["action"], action_mean, action_std)
    all_act = all_act[:total_frames * FS].reshape(total_frames, -1)  # (HS+H, 60)

    hist_px = all_px[:HS]
    hist_act = all_act[:HS]
    future_px = all_px[HS:]       # (H, 3, 224, 672)
    gt_act_12d = all_act[HS:, :AD]  # (H, 12) — raw 12-dim actions for warm start
    gt_act_60d = all_act[HS:]       # (H, 60)

    # Encode future frames to get target embeddings
    with torch.no_grad():
        future_emb = model.encode({
            "pixels": future_px.unsqueeze(0).to(device)
        })["emb"]  # (1, H, D)

    # Encode history
    with torch.no_grad():
        hist_info = model.encode({
            "pixels": hist_px.unsqueeze(0).to(device),
            "action": hist_act.unsqueeze(0).to(device),
        })
        hist_emb = hist_info["emb"]       # (1, HS, D)
        hist_act_emb = hist_info["act_emb"]  # (1, HS, D_a)

    # --- Oracle: predict with GT actions ---
    with torch.no_grad():
        gt_act_emb = model.action_encoder(gt_act_60d.unsqueeze(0).to(device))  # (1, H, D_a)
        all_act_emb = torch.cat([hist_act_emb, gt_act_emb], dim=1)  # (1, HS+H, D_a)
        pred_oracle = model.predict(hist_emb, all_act_emb[:, -HS:])  # (1, HS, D)
        oracle_cost = F.mse_loss(pred_oracle[:, -1, :], future_emb[:, -1, :]).item()

    # --- Zero-action baseline ---
    with torch.no_grad():
        zero_act_60d = torch.zeros(H, 60, device=device)
        zero_act_emb = model.action_encoder(zero_act_60d.unsqueeze(0))
        all_act_emb_z = torch.cat([hist_act_emb, zero_act_emb], dim=1)
        pred_zero = model.predict(hist_emb, all_act_emb_z[:, -HS:])
        zero_cost = F.mse_loss(pred_zero[:, -1, :], future_emb[:, -1, :]).item()

    # --- CEM with warm start ---
    info_cem = {
        "pixels": hist_px.unsqueeze(0).unsqueeze(0).to(device),
        "action": hist_act.unsqueeze(0).unsqueeze(0).to(device),
    }
    with torch.no_grad():
        res_warm = solver.solve_trajectory(info_cem, future_emb,
                                           init_actions=gt_act_12d.unsqueeze(0).to(device))
    cem_warm_cost = res_warm["costs"][-1]

    # --- CEM from zero ---
    with torch.no_grad():
        res_zero = solver.solve_trajectory(info_cem, future_emb, init_actions=None)
    cem_zero_cost = res_zero["costs"][-1]

    return {
        "episode_idx": ep_idx,
        "start_idx": start_idx,
        "goal_idx": goal_idx,
        "zero_cost": zero_cost,
        "oracle_cost": oracle_cost,
        "cem_warm_cost": cem_warm_cost,
        "cem_zero_cost": cem_zero_cost,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="LeWM eval on LeRobot")
    p.add_argument("--mode", choices=["predict", "cem"], default="predict")
    p.add_argument("--checkpoint", default="~/.stable_worldmodel/lewm_epoch_100_object.ckpt")
    p.add_argument("--dataset_root", default="~/Survey/Datasets")
    p.add_argument("--dataset_name", default="top_short_merged")
    p.add_argument("--history_size", type=int, default=3)
    p.add_argument("--num_preds", type=int, default=1)
    p.add_argument("--frameskip", type=int, default=5)
    p.add_argument("--action_dim", type=int, default=12)
    p.add_argument("--horizon", type=int, default=5)
    p.add_argument("--num_samples", type=int, default=200)
    p.add_argument("--cem_iterations", type=int, default=20)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--num_eval", type=int, default=20)
    p.add_argument("--goal_offset", type=int, default=None,
                   help="Auto = (horizon - history_size + 1) * frameskip")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="eval_results.json")
    return p.parse_args()


def run_cem_eval(model, dataset, args, action_mean, action_std):
    solver = SimpleCEMSolver(
        model=model,
        horizon=args.horizon,
        action_dim=args.action_dim,
        frameskip=args.frameskip,
        num_samples=args.num_samples,
        n_iterations=args.cem_iterations,
        topk=args.topk,
        device=args.device,
        seed=args.seed,
    )

    if args.goal_offset is None:
        args.goal_offset = (args.horizon - args.history_size + 1) * args.frameskip

    episodes = sample_eval_episodes(dataset, args)
    print(f"goal_offset={args.goal_offset}  "
          f"(horizon reach = {(args.horizon - args.history_size + 1) * args.frameskip} frames)")
    print(f"Evaluating {len(episodes)} episodes ...\n")

    results = []
    for i, (ep_idx, start, goal) in enumerate(episodes):
        r = evaluate_episode(model, solver, dataset, ep_idx, start, goal,
                             args, action_mean, action_std)
        results.append(r)
        print(f"[{i+1}/{len(episodes)}] ep={ep_idx}  "
              f"zero={r['zero_cost']:.4f}  oracle={r['oracle_cost']:.4f}  "
              f"cem_warm={r['cem_warm_cost']:.4f}  cem_zero={r['cem_zero_cost']:.4f}")

    # aggregate
    zeros = [r["zero_cost"] for r in results]
    oracles = [r["oracle_cost"] for r in results]
    cem_warm = [r["cem_warm_cost"] for r in results]
    cem_zero = [r["cem_zero_cost"] for r in results]

    beats_oracle = sum(1 for c in cem_warm if c < np.mean(oracles))
    beats_zero = sum(1 for c in cem_zero if c < np.mean(zeros))

    print(f"\n{'='*60}")
    print(f"CEM RESULTS  ({len(results)} episodes, goal_offset={args.goal_offset})")
    print(f"{'='*60}")
    print(f"Zero-action cost:          {np.mean(zeros):.4f} +/- {np.std(zeros):.4f}")
    print(f"Oracle (GT actions):       {np.mean(oracles):.4f} +/- {np.std(oracles):.4f}")
    print(f"CEM (warm-start from GT):  {np.mean(cem_warm):.4f} +/- {np.std(cem_warm):.4f}")
    print(f"CEM (zero-init):           {np.mean(cem_zero):.4f} +/- {np.std(cem_zero):.4f}")
    print(f"CEM_warm < mean oracle:    {beats_oracle}/{len(results)}")
    print(f"CEM_zero < mean zero-cost: {beats_zero}/{len(results)}")

    return {
        "config": vars(args),
        "summary": {
            "num_episodes": len(results),
            "mean_zero_cost": float(np.mean(zeros)),
            "mean_oracle_cost": float(np.mean(oracles)),
            "mean_cem_warm_cost": float(np.mean(cem_warm)),
            "mean_cem_zero_cost": float(np.mean(cem_zero)),
        },
        "episodes": results,
    }


def main():
    args = parse_args()
    args.checkpoint = str(Path(args.checkpoint).expanduser())
    args.dataset_root = str(Path(args.dataset_root).expanduser())

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {args.dataset_root}/{args.dataset_name}")
    print(f"Mode:       {args.mode}\n")

    model = load_model(args.checkpoint, args.device)
    dataset = load_dataset(args)
    action_mean, action_std = compute_action_stats(dataset)

    if args.mode == "predict":
        results = evaluate_prediction_quality(
            model, dataset, args, action_mean, action_std)
    else:
        results = run_cem_eval(model, dataset, args, action_mean, action_std)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
