"""LIBERO observation/action windows, adapted from ../DP/src/dp/data.py.

Modified to return natural-language instructions and preserve modality tokens.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class LiberoDatasetConfig:
    dataset_dir: str | Path
    horizon: int = 6
    camera_keys: tuple[str, ...] = ("obs/agentview_rgb", "obs/eye_in_hand_rgb")
    action_key: str = "actions"
    proprio_keys: tuple[str, ...] = ("obs/joint_states", "obs/gripper_states")
    feature_cache_dir: str | Path | None = None
    expected_tasks: int | None = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "camera_keys", tuple(self.camera_keys))
        object.__setattr__(self, "proprio_keys", tuple(self.proprio_keys))
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not self.camera_keys:
            raise ValueError("camera_keys cannot be empty")


@dataclass(frozen=True)
class Task:
    name: str
    instruction: str
    path: Path


@dataclass(frozen=True)
class Window:
    task_index: int
    demo_key: str
    step: int


@dataclass(frozen=True)
class DatasetStatistics:
    action_minimum: torch.Tensor
    action_maximum: torch.Tensor
    proprio_mean: torch.Tensor
    proprio_std: torch.Tensor


def _demo_keys(group: h5py.Group) -> list[str]:
    return sorted(
        (name for name, value in group.items() if isinstance(value, h5py.Group) and name.startswith("demo_")),
        key=lambda name: int(name.split("_")[-1]),
    )


def _instruction(group: h5py.Group, path: Path) -> str:
    raw = group.attrs.get("problem_info", "")
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return str(json.loads(raw)["language_instruction"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Missing language_instruction in {path}") from error


class LiberoDataset(Dataset[dict[str, Any]]):
    """Episode-safe, current-observation action chunks.

    Uses current RGB, joint/gripper state, and actual language instructions.
    Task IDs are metadata only. Targets never cross episode boundaries.
    """

    def __init__(self, config: LiberoDatasetConfig) -> None:
        self.config = config
        root = Path(config.dataset_dir).expanduser().resolve()
        paths = sorted(root.glob("*_demo.hdf5"))
        if not paths or (config.expected_tasks is not None and len(paths) != config.expected_tasks):
            raise ValueError(f"Expected {config.expected_tasks or 'at least one'} task files under {root}, found {len(paths)}")
        self.tasks: list[Task] = []
        self.windows: list[Window] = []
        self._handles: dict[Path, h5py.File] = {}
        self._feature_handles: dict[Path, h5py.File] = {}
        self.feature_cache_path: Path | None = None
        self.cache_manifest: dict[str, Any] | None = None
        if config.feature_cache_dir is not None:
            self.feature_cache_path = Path(config.feature_cache_dir).expanduser().resolve() / "features.hdf5"
            if not self.feature_cache_path.is_file():
                raise FileNotFoundError(f"Token feature cache does not exist: {self.feature_cache_path}")
            manifest_path = self.feature_cache_path.parent / "manifest.json"
            self.cache_manifest = json.loads(manifest_path.read_text())
            if self.cache_manifest.get("schema") != "representation_fm_tokens_v1":
                raise ValueError("Incompatible cache: regenerate camera/spatial/language token features")
        self.image_transform = "none"
        for task_index, path in enumerate(paths):
            with h5py.File(path, "r") as handle:
                data = handle["data"]
                transform = data.attrs.get("clad_image_transform", "none")
                if isinstance(transform, bytes):
                    transform = transform.decode()
                if str(transform) not in {"none", "rotate_180"}:
                    raise ValueError(f"Unsupported stored image transform: {transform}")
                if task_index == 0:
                    self.image_transform = str(transform)
                elif str(transform) != self.image_transform:
                    raise ValueError("Dataset files disagree on stored image transform")
                self.tasks.append(Task(path.stem.removesuffix("_demo"), _instruction(data, path), path))
                for demo_key in _demo_keys(data):
                    demo = data[demo_key]
                    required = [config.action_key, *config.camera_keys, *config.proprio_keys]
                    missing = [key for key in required if key not in demo]
                    if missing:
                        raise ValueError(f"{path}:{demo_key} is missing {missing}")
                    length = int(demo[config.action_key].shape[0])
                    if demo[config.action_key].shape[1:] != (7,):
                        raise ValueError(f"{path}:{demo_key} actions must be [T,7]")
                    for key in required:
                        if demo[key].shape[0] != length:
                            raise ValueError(f"Temporal mismatch for {path}:{demo_key}/{key}")
                    for step in range(0, length - config.horizon + 1):
                        self.windows.append(Window(task_index, demo_key, step))
        if not self.windows:
            raise ValueError("No valid action windows found")
        if self.cache_manifest is not None:
            expected = self.cache_data_signature()
            if any(self.cache_manifest.get(key) != value for key, value in expected.items()):
                raise ValueError("Token cache does not match the dataset sources, cameras, or transforms")
            with h5py.File(self.feature_cache_path, "r") as cached:
                if cached.attrs.get("cache_id") != self.cache_manifest.get("cache_id"):
                    raise ValueError("Token cache data and manifest belong to different builds")

    def cache_data_signature(self) -> dict[str, Any]:
        return {
            "camera_keys": list(self.config.camera_keys),
            "task_names": [task.name for task in self.tasks],
            "instructions": [task.instruction for task in self.tasks],
            "image_transform": self.image_transform,
            "source_files": [{"path": str(task.path.resolve()),
                              "size": task.path.stat().st_size,
                              "mtime_ns": task.path.stat().st_mtime_ns}
                             for task in self.tasks],
        }

    def validate_feature_cache(self, encoder_config: Any) -> None:
        if len(self.config.camera_keys) != encoder_config.num_views:
            raise ValueError("Dataset cameras and encoder num_views differ")
        if self.cache_manifest is not None:
            if self.cache_manifest.get("encoder") != encoder_config.cache_signature():
                raise ValueError("Token cache was created with a different frozen encoder configuration")

    def __len__(self) -> int:
        return len(self.windows)

    def _handle(self, path: Path) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def _feature_handle(self, path: Path) -> h5py.File:
        if path not in self._feature_handles:
            self._feature_handles[path] = h5py.File(path, "r")
        return self._feature_handles[path]

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.windows[index]
        task = self.tasks[window.task_index]
        demo = self._handle(task.path)["data"][window.demo_key]
        step, horizon = window.step, self.config.horizon
        proprio = np.concatenate([np.asarray(demo[key][step], dtype=np.float32) for key in self.config.proprio_keys])
        actions = np.asarray(demo[self.config.action_key][step : step + horizon], dtype=np.float32)
        sample = {
            "proprio": torch.from_numpy(proprio),
            "task_id": torch.tensor(window.task_index, dtype=torch.long),
            "instructions": task.instruction,
            "actions": torch.from_numpy(actions),
        }
        if self.feature_cache_path is None:
            images = np.stack([np.asarray(demo[key][step], dtype=np.uint8) for key in self.config.camera_keys])
            # [V,H,W,C] -> [V,C,H,W]; transforms were already applied while rerendering.
            sample["images"] = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 3, 1, 2)))
        else:
            features = self._feature_handle(self.feature_cache_path)["features"][task.name][window.demo_key]
            sample["vision_features"] = torch.from_numpy(np.asarray(features[step], dtype=np.float32))
            language = self._feature_handle(self.feature_cache_path)["language"][task.name]
            sample["text_features"] = torch.from_numpy(np.asarray(language["features"], dtype=np.float32))
            sample["text_padding_mask"] = torch.from_numpy(np.asarray(language["padding_mask"], dtype=bool))
        return sample

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        for handle in self._feature_handles.values():
            handle.close()
        self._feature_handles.clear()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_feature_handles"] = {}
        return state


def compute_statistics(dataset: LiberoDataset) -> DatasetStatistics:
    """Scan each stored state/action once, avoiding window duplication."""

    action_minimum: np.ndarray | None = None
    action_maximum: np.ndarray | None = None
    values: list[np.ndarray] = []
    for task in dataset.tasks:
        with h5py.File(task.path, "r") as handle:
            for demo_key in _demo_keys(handle["data"]):
                demo = handle["data"][demo_key]
                actions = np.asarray(demo[dataset.config.action_key], dtype=np.float32)
                proprio = np.concatenate(
                    [np.asarray(demo[key], dtype=np.float32) for key in dataset.config.proprio_keys], axis=1
                )
                action_minimum = actions.min(0) if action_minimum is None else np.minimum(action_minimum, actions.min(0))
                action_maximum = actions.max(0) if action_maximum is None else np.maximum(action_maximum, actions.max(0))
                values.append(proprio)
    if action_minimum is None or action_maximum is None:
        raise ValueError("Dataset contains no actions")
    proprio_values = np.concatenate(values, axis=0)
    return DatasetStatistics(
        torch.from_numpy(action_minimum),
        torch.from_numpy(action_maximum),
        torch.from_numpy(proprio_values.mean(0).astype(np.float32)),
        torch.from_numpy(proprio_values.std(0).clip(1e-6).astype(np.float32)),
    )
