#!/usr/bin/env python3
"""Cache frozen camera/spatial and language tokens; never trainable modules."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from repr_fm.data import LiberoDataset, LiberoDatasetConfig, _demo_keys
from repr_fm.encoders import EncoderConfig, FrozenTextTokens, FrozenVisionTokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/libero10_fm.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    config = yaml.safe_load(args.config.read_text())
    data_config = replace(LiberoDatasetConfig(**config["data"]), feature_cache_dir=None)
    encoder_config = EncoderConfig(**config["model"]["encoder"])
    dataset = LiberoDataset(data_config)
    dataset.validate_feature_cache(encoder_config)
    device = torch.device(args.device)
    vision = FrozenVisionTokens(encoder_config).to(device).eval()
    text = FrozenTextTokens(encoder_config).to(device).eval()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "features.hdf5"
    manifest_path = output / "manifest.json"
    if destination.exists() or manifest_path.exists():
        raise FileExistsError("Choose a new cache directory; existing feature caches are not overwritten")
    cache_id = str(uuid.uuid4())
    temporary = output / f"features.{cache_id}.tmp"
    frames = 0
    try:
        with h5py.File(temporary, "w") as cached:
            cached.attrs["cache_id"] = cache_id
            features = cached.create_group("features")
            language = cached.create_group("language")
            for task in tqdm(dataset.tasks, desc="Caching tasks"):
                text_features, text_mask = text([task.instruction], device)
                sentence = language.create_group(task.name)
                sentence.create_dataset("features", data=text_features[0].cpu().numpy())
                sentence.create_dataset("padding_mask", data=text_mask[0].cpu().numpy())
                task_features = features.create_group(task.name)
                with h5py.File(task.path, "r") as source:
                    for demo_key in _demo_keys(source["data"]):
                        demo = source["data"][demo_key]
                        length = len(demo[data_config.action_key])
                        shape = (length, encoder_config.num_views,
                                 encoder_config.vision_grid_size ** 2,
                                 encoder_config.vision_feature_dim)
                        target = task_features.create_dataset(
                            demo_key, shape=shape, dtype=np.float32,
                            chunks=(min(length, args.batch_size), *shape[1:]),
                        )
                        for start in range(0, length, args.batch_size):
                            stop = min(start + args.batch_size, length)
                            images = np.stack([np.asarray(demo[key][start:stop], dtype=np.uint8)
                                               for key in data_config.camera_keys], axis=1)
                            images = torch.from_numpy(np.ascontiguousarray(images.transpose(0, 1, 4, 2, 3))).to(device)
                            target[start:stop] = vision(images).cpu().numpy()
                            frames += stop - start
        manifest = {
            "schema": "representation_fm_tokens_v1", "cache_id": cache_id,
            "encoder": encoder_config.cache_signature(),
            **dataset.cache_data_signature(), "frames": frames,
            "feature_dtype": "float32",
        }
        temporary.replace(destination)
        manifest_temporary = manifest_path.with_suffix(".json.tmp")
        manifest_temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        manifest_temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
        dataset.close()
    print(f"Cached {frames} frames in {output}")


if __name__ == "__main__":
    main()
