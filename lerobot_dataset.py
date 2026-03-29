"""LeRobot dataset wrapper for the LeWM training pipeline.

Wraps LeRobot datasets (Parquet + MP4) as a stable_worldmodel Dataset,
compatible with the existing training pipeline in train.py.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from lerobot.datasets.utils import load_episodes, load_info, load_nested_dataset
from lerobot.datasets.video_utils import decode_video_frames
from stable_worldmodel.data.dataset import Dataset


class LeRobotDatasetWrapper(Dataset):
    """Dataset wrapper that reads LeRobot format (Parquet + MP4) directly.

    Inherits from stable_worldmodel's Dataset base class, providing the same
    interface as HDF5Dataset. Supports multi-camera views concatenated
    horizontally into a single pixel tensor.

    Args:
        root: Path to the LeRobot dataset root directory.
        frameskip: Number of frames to skip between samples.
        num_steps: Number of steps per sample sequence.
        transform: Optional data transform callable.
        keys_to_load: List of output keys (e.g. ['pixels', 'action', 'state']).
        keys_to_cache: Output keys to load entirely into memory.
        camera_keys: LeRobot video keys for camera views.
        key_map: Mapping from output key name to LeRobot parquet column name.
            Useful when the LeRobot column name contains dots (e.g. 'observation.state')
            which would break OmegaConf's setattr in train.py.
            Default: {'state': 'observation.state'}
    """

    # Default mapping: LeWM key -> LeRobot parquet column
    DEFAULT_KEY_MAP = {
        "state": "observation.state",
    }

    def __init__(
        self,
        root: str | Path,
        frameskip: int = 1,
        num_steps: int = 1,
        transform: Callable[[dict], dict] | None = None,
        keys_to_load: list[str] | None = None,
        keys_to_cache: list[str] | None = None,
        camera_keys: list[str] | None = None,
        key_map: dict[str, str] | None = None,
    ) -> None:
        self.root = Path(root)
        self._cache: dict[str, np.ndarray] = {}
        self.camera_keys = camera_keys or []
        self.key_map = {**self.DEFAULT_KEY_MAP, **(key_map or {})}

        # Load metadata
        self.info = load_info(self.root)
        self.episodes = load_episodes(self.root)
        self.fps = self.info["fps"]

        # Build episode lengths and offsets
        num_episodes = len(self.episodes)
        self.lengths = np.array(
            [self.episodes[i]["length"] for i in range(num_episodes)], dtype=np.int64
        )
        self.offsets = np.array(
            [self.episodes[i]["dataset_from_index"] for i in range(num_episodes)],
            dtype=np.int64,
        )

        # Load non-video data from parquet
        self.hf_data = load_nested_dataset(self.root / "data")

        # Determine keys
        self._keys = keys_to_load or ["pixels", "action", "state"]

        # Cache requested columns (resolve to parquet column names)
        for key in keys_to_cache or []:
            if key == "pixels":
                continue
            parquet_key = self.key_map.get(key, key)
            raw = np.array(self.hf_data[parquet_key][:], dtype=np.float32)
            self._cache[key] = raw
            logging.info(f"Cached '{key}' ({raw.shape}) from LeRobot dataset")

        super().__init__(self.lengths, self.offsets, frameskip, num_steps, transform)

    @property
    def column_names(self) -> list[str]:
        return self._keys

    def _get_video_path(self, ep_idx: int, vid_key: str) -> Path:
        ep = self.episodes[ep_idx]
        video_path_tmpl = self.info["video_path"]
        return self.root / video_path_tmpl.format(
            video_key=vid_key,
            chunk_index=ep[f"videos/{vid_key}/chunk_index"],
            file_index=ep[f"videos/{vid_key}/file_index"],
        )

    def _decode_camera_frames(
        self, ep_idx: int, vid_key: str, frame_indices: list[int]
    ) -> torch.Tensor:
        ep = self.episodes[ep_idx]
        video_path = self._get_video_path(ep_idx, vid_key)
        from_ts = ep[f"videos/{vid_key}/from_timestamp"]
        timestamps = [from_ts + idx / self.fps for idx in frame_indices]
        return decode_video_frames(video_path, timestamps, tolerance_s=1e-4)

    def _load_slice(self, ep_idx: int, start: int, end: int) -> dict:
        ep = self.episodes[ep_idx]
        g_start = ep["dataset_from_index"] + start
        g_end = ep["dataset_from_index"] + end

        steps = {}
        for col in self._keys:
            if col == "pixels":
                frame_indices = list(range(start, end, self.frameskip))

                if len(self.camera_keys) == 1:
                    steps["pixels"] = self._decode_camera_frames(
                        ep_idx, self.camera_keys[0], frame_indices
                    )
                elif len(self.camera_keys) > 1:
                    cam_frames = [
                        self._decode_camera_frames(ep_idx, cam_key, frame_indices)
                        for cam_key in self.camera_keys
                    ]
                    steps["pixels"] = torch.cat(cam_frames, dim=3)
                else:
                    raise ValueError("No camera_keys specified for pixel loading")

            else:
                # Resolve to parquet column name
                parquet_key = self.key_map.get(col, col)
                if col in self._cache:
                    data = self._cache[col][g_start:g_end]
                else:
                    data = np.array(
                        self.hf_data[parquet_key][g_start:g_end], dtype=np.float32
                    )

                if col != "action":
                    data = data[:: self.frameskip]

                steps[col] = torch.from_numpy(data.copy()) if isinstance(data, np.ndarray) else data

        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col: str) -> np.ndarray:
        if col in self._cache:
            return self._cache[col]
        parquet_key = self.key_map.get(col, col)
        return np.array(self.hf_data[parquet_key][:], dtype=np.float32)

    def get_dim(self, col: str) -> int:
        data = self.get_col_data(col)
        return int(np.prod(data.shape[1:])) if data.ndim > 1 else 1
