# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LeWorldModel (LeWM) is a stable end-to-end Joint-Embedding Predictive Architecture (JEPA) that learns world models from raw pixels. The codebase is built on top of `stable-worldmodel` (environment management, planning, evaluation) and `stable-pretraining` (training infrastructure).

## Installation

```bash
uv venv --python=3.10
source .venv/bin/activate
uv pip install stable-worldmodel[train,env]
```

## Common Commands

### Training

Train a model with Hydra configuration:
```bash
python train.py data=pusht
```

The training script uses configs under `config/train/`:
- `config/train/lewm.yaml` - main training config
- `config/train/data/*.yaml` - dataset-specific configs (pusht, dmc, tworoom, ogb)

Set WandB entity/project in `config/train/lewm.yaml` before training.

### Evaluation

Evaluate a trained model:
```bash
python eval.py --config-name=pusht.yaml policy=pusht/lewm
```

Important: `policy` must be the checkpoint path **relative to `$STABLEWM_HOME`**, without the `_object.ckpt` suffix.

Available configs under `config/eval/`:
- Environment configs: `pusht.yaml`, `cube.yaml`, `reacher.yaml`, `tworoom.yaml`
- Solver configs: `solver/cem.yaml`, `solver/adam.yaml`
- Launcher config: `launcher/local.yaml`

## Data Storage

Datasets are HDF5 files located at `$STABLEWM_HOME` (defaults to `~/.stable-wm/`). Override with:
```bash
export STABLEWM_HOME=/path/to/your/storage
```

Download datasets from [HuggingFace](https://huggingface.co/collections/quentinll/lewm) and decompress:
```bash
tar --zstd -xvf archive.tar.zst
```

Place extracted `.h5` files under `$STABLEWM_HOME`.

## Model Architecture

### Core Components (jepa.py)
- **JEPA**: Main model class that orchestrates encoding, prediction, and inference
  - `encode()`: Encodes observations and actions into embeddings
  - `predict()`: Predicts next state embedding from history
  - `rollout()`: Autoregressive rollout for planning/MPC
  - `get_cost()`: Computes cost for action candidates (used by MPC solvers)

### Neural Modules (module.py)
- **Transformer**: Standard transformer with optional AdaLN-zero conditioning
- **ConditionalBlock**: Transformer block with AdaLN-zero modulation (used by predictor)
- **ARPredictor**: Autoregressive predictor using conditioned transformer blocks
- **SIGReg**: Sketch Isotropic Gaussian Regularizer - enforces Gaussian-distributed latent embeddings
- **Embedder**: Action encoder with Conv1D + MLP
- **MLP**: Simple MLP with optional normalization

### Training Pipeline (train.py)

1. `lejepa_forward()`: Main forward pass
   - Encodes observations → embeddings
   - Predicts next embedding from history
   - Computes two losses:
     - `pred_loss`: MSE between predicted and target embeddings
     - `sigreg_loss`: Gaussian regularization via SIGReg

2. Model composition:
   - Encoder: ViT from `stable_pretraining.backbone.utils.vit_hf()`
   - Predictor: ARPredictor with AdaLN-zero conditioning
   - Action encoder: Embedder
   - Projectors: MLPs (encoder → embed_dim, predictor → embed_dim)

3. Checkpoints saved to `$STABLEWM_HOME/{run_id}/` as:
   - `{name}_object.ckpt`: Pickled model object (used by eval)
   - `{name}_weights.ckpt`: Weights-only state_dict

### Evaluation Pipeline (eval.py)

Evaluation uses the `stable_worldmodel` API:
1. Creates world environment (Gymnasium-based)
2. Loads model via `swm.policy.AutoCostModel(policy_path)`
3. Creates solver (CEM/Adam) and `WorldModelPolicy`
4. Evaluates on dataset trajectories with goal-conditioned planning

## Configuration System

Uses Hydra with YAML configs under `config/`:

**Training structure:**
```
config/train/
├── lewm.yaml          # Main config (model, optimizer, trainer, wandb)
├── data/              # Dataset configs (pusht.yaml, dmc.yaml, etc.)
└── launcher/          # Training launcher configs
```

**Evaluation structure:**
```
config/eval/
├── pusht.yaml         # Environment + eval parameters
├── solver/            # MPC solver configs (cem.yaml, adam.yaml)
└── launcher/          # Eval launcher configs
```

Key config parameters:
- `wm.history_size`: Number of context frames
- `wm.num_preds`: Number of prediction steps
- `loss.sigreg.weight`: Weight for Gaussian regularization
- `plan_config.horizon`: Planning horizon for MPC

## Loading a Checkpoint

Via `stable_worldmodel` API:
```python
import stable_worldmodel as swm

# Load model for planning/MPC
cost = swm.policy.AutoCostModel('pusht/lewm')
```

The function accepts:
- `run_name`: Path relative to `$STABLEWM_HOME`, without `_object.ckpt` suffix
- `cache_dir`: Optional override for checkpoint root

## Key Dependencies

- `stable-worldmodel`: Environments, planning, evaluation
- `stable-pretraining`: Training infrastructure, datasets
- `einops`: Tensor reshaping operations
- `hydra`: Configuration management
- `lightning`: PyTorch Lightning trainer
- `torchvision`: Image transforms

## Device Handling

The code uses `proj.device` instead of hardcoded CUDA devices. Checkpoints saved as `.ckpt` files are loaded via `stable_worldmodel` API.
