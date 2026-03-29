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
