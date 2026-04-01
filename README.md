# LeWorldModel — LeRobot Training Fork

This repo is a fork of [LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm) adapted for training on [LeRobot](https://github.com/huggingface/lerobot)-format datasets. It adds:

- **LeRobot dataset wrapper** (`lerobot_dataset.py`) — reads LeRobot v3.0 datasets (Parquet + MP4) directly, no HDF5 conversion needed
- **Multi-camera support** — concatenates multiple camera views as input
- **SwanLab logging** — experiment tracking via [SwanLab](https://swanlab.cn) instead of WandB
- **Convenience launch script** (`train.sh`) with environment-variable-based configuration

## Citation

This work is based on the following paper:

> Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels.** arXiv preprint, 2026.

```bibtex
@article{maes_lelidec2026lewm,
  title={LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint},
  year={2026}
}
```

Paper: [arxiv.org/abs/2603.19312](https://arxiv.org/abs/2603.19312) | Original repo: [github.com/lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)

## Setup

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Training

```bash
export SWANLAB_API_KEY=your_key_here
export DATASET_ROOT=/path/to/datasets   # parent dir containing dataset folders

# Train on a specific dataset
bash train.sh dataset.dataset_name=top_short_merged

# Full dataset
bash train.sh dataset.dataset_name=four_types_merged
```

Override any Hydra config via CLI:

```bash
bash train.sh dataset.dataset_name=top_short_merged loader.batch_size=32 trainer.max_epochs=50
```

## Dataset Format

Expects LeRobot v3.0 datasets with the following structure:

```
$DATASET_ROOT/<dataset_name>/
├── meta/
│   └── info.json
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.<camera_key>/
        └── chunk-000/
            └── file-000.mp4
```

Supported features: `observation.state`, `action`, and multiple `observation.images.*` cameras.

## Visualization

Scripts under `visualize/` reproduce key analyses from the paper. Each reads checkpoints from `$STABLEWM_HOME` (default `~/.stable_worldmodel/`).

**Latent Space t-SNE** (Figure 9) — 2D projection of embeddings colored by state, comparing checkpoints:
```bash
python visualize/latent_tsne.py \
    --checkpoints ~/.stable_worldmodel/<run>/lewm_epoch_1_object.ckpt \
                   ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
    --dataset_name top_short_merged \
    --output latent_tsne.png
```

**Temporal Straightening** (Figure 17) — cosine similarity of consecutive velocity vectors over training:
```bash
python visualize/temporal_straightening.py \
    --checkpoints ~/.stable_worldmodel/<run>/lewm_epoch_1_object.ckpt \
                   ~/.stable_worldmodel/<run>/lewm_epoch_50_object.ckpt \
                   ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
    --output temporal_straightening.png
```

**Violation-of-Expectation** (Figure 10) — surprise under unperturbed / visual / physical perturbation:
```bash
python visualize/voe_surprise.py \
    --checkpoint ~/.stable_worldmodel/<run>/lewm_epoch_100_object.ckpt \
    --dataset_name top_short_merged \
    --n_trajectories 50 \
    --output voe_surprise.png
```

**Latent Decoder** (Appendix D) — train a cross-attention decoder to reconstruct images from CLS embeddings, then visualize rollout predictions:
```bash
# Train decoder (requires ~20GB RAM for embedding cache)
python visualize/latent_decoder.py train \
    --checkpoint ~/.stable_worldmodel/lewm_epoch_100_object.ckpt \
    --dataset_name top_short_merged \
    --output_dir visualize/decoder_checkpoints

# Visualize decoded rollout (all cameras concatenated)
python visualize/latent_decoder.py visualize \
    --checkpoint ~/.stable_worldmodel/lewm_epoch_100_object.ckpt \
    --decoder_path visualize/decoder_checkpoints/best_decoder.pt \
    --dataset_name top_short_merged \
    --output visualize/rollout_decoded.png

# Per-camera breakdown
python visualize/latent_decoder.py visualize \
    --checkpoint ~/.stable_worldmodel/lewm_epoch_100_object.ckpt \
    --decoder_path visualize/decoder_checkpoints/best_decoder.pt \
    --dataset_name top_short_merged \
    --per_camera \
    --output visualize/rollout_decoded_cameras.png
```

## Evaluation

`eval_lerobot.py` evaluates a trained model on LeRobot datasets in two modes:

### Prediction Quality (`--mode predict`)

Replicates the training forward pass on held-out data and measures how accurately the model predicts future embeddings:

```bash
python eval_lerobot.py --mode predict --num_eval 50
```

Reports MSE and cosine similarity between predicted and actual future embeddings. A good model should achieve cosine similarity > 0.99.

### CEM Planning (`--mode cem`)

Uses Cross-Entropy Method (CEM) to find action sequences whose predicted future embeddings match actual encoded future frames. Measures whether the learned dynamics are controllable:

```bash
python eval_lerobot.py --mode cem --num_eval 10 --num_samples 200 --cem_iterations 20
```

Reports costs for four methods:
- **Zero-action**: predict with zero actions (baseline)
- **Oracle**: predict with ground-truth actions from the dataset
- **CEM (warm)**: CEM initialized from ground-truth actions
- **CEM (zero)**: CEM initialized from scratch

Lower cost = predicted embeddings closer to actual future frame embeddings. Key parameters:
- `--horizon` — planning horizon in model steps (default 5)
- `--goal_offset` — distance between start and goal frames (auto-computed from horizon)
- `--num_samples` — CEM population size (default 200)
- `--cem_iterations` — CEM optimization steps (default 20)
