"""Synthetic LIBERO/cache integration and deterministic training-resume tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch
import yaml

from repr_fm.checkpoint import read_checkpoint
from repr_fm.data import LiberoDataset, LiberoDatasetConfig
from repr_fm.encoders import EncoderConfig
from test_policy import tiny_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def synthetic_data(tmp_path):
    """Two distinct episodes expose target windows that accidentally cross demos."""
    data_dir, cache_dir = tmp_path / "data", tmp_path / "cache"
    data_dir.mkdir()
    cache_dir.mkdir()
    task_name = "synthetic_pick_task"
    instruction = "Pick up the red block and place it beside the blue bowl."
    random = np.random.default_rng(57)
    episodes = {}
    with h5py.File(data_dir / f"{task_name}_demo.hdf5", "w") as handle:
        data = handle.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
        data.attrs["clad_image_transform"] = "none"
        for episode_index, length in enumerate((8, 9)):
            name = f"demo_{episode_index}"
            demo = data.create_group(name)
            low, high = ((-0.8, -0.2) if episode_index == 0 else (0.2, 0.8))
            values = {
                "actions": random.uniform(low, high, size=(length, 7)).astype(np.float32),
                "obs/joint_states": random.normal(size=(length, 7)).astype(np.float32),
                "obs/gripper_states": random.normal(size=(length, 2)).astype(np.float32),
                "obs/agentview_rgb": random.integers(0, 256, (length, 4, 4, 3), dtype=np.uint8),
                "obs/eye_in_hand_rgb": random.integers(0, 256, (length, 4, 4, 3), dtype=np.uint8),
            }
            for key, value in values.items():
                demo.create_dataset(key, data=value)
            values["features"] = random.normal(size=(length, 2, 4, 8)).astype(np.float32)
            episodes[name] = values

    model_config = tiny_config()
    encoder_config = EncoderConfig(**model_config["encoder"])
    raw_config = LiberoDatasetConfig(dataset_dir=str(data_dir), horizon=6, expected_tasks=1)
    dataset = LiberoDataset(raw_config)
    try:
        signature = dataset.cache_data_signature()
    finally:
        dataset.close()
    cache_id = "synthetic-spatial-and-language-cache"
    language = random.normal(size=(8, 12)).astype(np.float32)
    padding = np.array([False, False, False, False, False, True, True, True])
    with h5py.File(cache_dir / "features.hdf5", "w") as handle:
        handle.attrs["cache_id"] = cache_id
        for name, values in episodes.items():
            handle.create_dataset(f"features/{task_name}/{name}", data=values["features"])
        handle.create_dataset(f"language/{task_name}/features", data=language)
        handle.create_dataset(f"language/{task_name}/padding_mask", data=padding)
    manifest = {
        "schema": "representation_fm_tokens_v1", "cache_id": cache_id,
        "encoder": encoder_config.cache_signature(), **signature,
    }
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return {
        "data_dir": data_dir, "cache_dir": cache_dir, "manifest_path": manifest_path,
        "instruction": instruction, "task_name": task_name, "episodes": episodes,
        "language": language, "padding": padding, "encoder_config": encoder_config,
        "model_config": model_config,
    }


def dataset_config(fixture, cached=True):
    return LiberoDatasetConfig(
        dataset_dir=str(fixture["data_dir"]), horizon=6, expected_tasks=1,
        feature_cache_dir=str(fixture["cache_dir"]) if cached else None,
    )


@pytest.mark.parametrize("cached", [False, True])
def test_current_observation_and_targets_stay_aligned_within_each_episode(synthetic_data, cached):
    fixture = synthetic_data
    dataset = LiberoDataset(dataset_config(fixture, cached))
    try:
        dataset.validate_feature_cache(fixture["encoder_config"])
        assert len(dataset) == (8 - 6 + 1) + (9 - 6 + 1)
        assert dataset.tasks[0].instruction == fixture["instruction"]
        for index, window in enumerate(dataset.windows):
            sample = dataset[index]
            episode = fixture["episodes"][window.demo_key]
            step = window.step
            assert sample["instructions"] == fixture["instruction"]
            assert sample["proprio"].shape == (9,)
            expected_proprio = np.concatenate((
                episode["obs/joint_states"][step], episode["obs/gripper_states"][step],
            ))
            torch.testing.assert_close(sample["proprio"], torch.from_numpy(expected_proprio))
            torch.testing.assert_close(
                sample["actions"], torch.from_numpy(episode["actions"][step : step + 6]),
                rtol=0, atol=0,
            )
            assert sample["actions"].shape == (6, 7)
            if cached:
                assert "images" not in sample
                torch.testing.assert_close(
                    sample["vision_features"], torch.from_numpy(episode["features"][step]),
                    rtol=0, atol=0,
                )
                torch.testing.assert_close(sample["text_features"], torch.from_numpy(fixture["language"]))
                assert torch.equal(sample["text_padding_mask"], torch.from_numpy(fixture["padding"]))
            else:
                assert sample["images"].shape == (2, 3, 4, 4)
                assert sample["images"].dtype == torch.uint8
                expected_images = np.stack([
                    episode[key][step] for key in dataset.config.camera_keys
                ]).transpose(0, 3, 1, 2)
                assert torch.equal(sample["images"], torch.from_numpy(expected_images.copy()))
    finally:
        dataset.close()


@pytest.mark.parametrize("corruption", ["legacy_cls", "source", "camera_order", "cache_build", "encoder"])
def test_incompatible_feature_caches_are_rejected(synthetic_data, corruption):
    fixture = synthetic_data
    path = fixture["manifest_path"]
    manifest = json.loads(path.read_text())
    if corruption == "legacy_cls":
        manifest["schema"] = "legacy_dinov2_mean_cls"
        with h5py.File(fixture["cache_dir"] / "features.hdf5", "r+") as handle:
            key = f"features/{fixture['task_name']}/demo_0"
            del handle[key]
            handle.create_dataset(key, data=np.zeros((8, 768), dtype=np.float32))
    elif corruption == "source":
        manifest["source_files"][0]["size"] += 1
    elif corruption == "camera_order":
        manifest["camera_keys"].reverse()
    elif corruption == "cache_build":
        manifest["cache_id"] = "a-different-build"
    else:
        manifest["encoder"]["vision_grid_size"] = 1
    path.write_text(json.dumps(manifest))

    if corruption == "encoder":
        dataset = LiberoDataset(dataset_config(fixture))
        try:
            with pytest.raises(ValueError, match="encoder"):
                dataset.validate_feature_cache(fixture["encoder_config"])
        finally:
            dataset.close()
    else:
        with pytest.raises(ValueError):
            LiberoDataset(dataset_config(fixture))


def test_training_cli_resume_matches_uninterrupted_parameters_and_ema(synthetic_data, tmp_path):
    fixture = synthetic_data
    config = {
        "data": {
            "dataset_dir": str(fixture["data_dir"]),
            "feature_cache_dir": str(fixture["cache_dir"]),
            "horizon": 6, "expected_tasks": 1,
        },
        "model": fixture["model_config"],
        "train": {
            "seed": 42, "max_steps": 4, "batch_size": 2, "num_workers": 0,
            "learning_rate": 0.001, "warmup_steps": 0, "min_lr_ratio": 0.2,
            "amp": False, "log_interval": 1, "save_interval": 2,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    environment = {
        **os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    }

    def train(output, resume=None):
        command = [
            sys.executable, str(PROJECT_ROOT / "scripts/train.py"),
            "--config", str(config_path), "--output-dir", str(output), "--device", "cpu",
        ]
        if resume is not None:
            command.extend(("--resume", str(resume)))
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, env=environment, capture_output=True,
            text=True, timeout=45, check=False,
        )
        assert result.returncode == 0, result.stdout + "\n" + result.stderr

    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    train(full_dir)
    # Both runs retain the same four-step LR schedule and training endpoint.
    train(resumed_dir, resume=full_dir / "checkpoint_000002.pt")
    full = read_checkpoint(full_dir / "checkpoint_000004.pt")
    resumed = read_checkpoint(resumed_dir / "checkpoint_000004.pt")
    assert full["step"] == resumed["step"] == 4
    for section in ("model_trainable", "model_buffers", "ema"):
        assert full[section].keys() == resumed[section].keys()
        for name, expected in full[section].items():
            torch.testing.assert_close(resumed[section][name], expected, rtol=0, atol=0)
    assert full["sampler"] == resumed["sampler"]
    assert full["scheduler"] == resumed["scheduler"]
    resumed_metrics = [json.loads(line) for line in (resumed_dir / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in resumed_metrics] == [3, 4]
