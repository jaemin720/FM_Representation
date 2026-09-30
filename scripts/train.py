#!/usr/bin/env python3
"""Train a modular language-conditioned FM policy on LIBERO HDF5 data.

Optimizer/AMP/EMA orchestration is adapted from practice/DP/scripts/train.py;
the old project and its checkpoint schema are not modified.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Sampler
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from repr_fm import build_policy
from repr_fm.checkpoint import (
    EMA, capture_rng_state, load_model_state, make_model_payload,
    read_checkpoint, restore_rng_state, save_checkpoint, update_latest,
)
from repr_fm.data import LiberoDataset, LiberoDatasetConfig, compute_statistics


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/libero10_fm.yaml")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/libero10_fm")
    parser.add_argument("--device", default="cuda", help="cuda, cpu, or auto")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--feature-cache-dir", type=Path, help="New spatial-token/text cache; legacy CLS caches are incompatible")
    return parser.parse_args()


class ResumableBatchSampler(Sampler[list[int]]):
    """Deterministic epoch permutations with a trainer-owned consumed cursor.

    DataLoader prefetch cannot advance the saved cursor. Restoring preserves
    sample order; worker-side random augmentations would need additional state.
    """

    def __init__(self, size: int, batch_size: int, seed: int) -> None:
        if size < 1 or batch_size < 1:
            raise ValueError("Dataset size and batch size must be positive")
        self.size, self.batch_size, self.seed = size, batch_size, seed
        self.epoch, self.next_batch = 0, 0

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(self.size / self.batch_size)

    def __len__(self) -> int:
        return self.batches_per_epoch - self.next_batch

    def __iter__(self) -> Iterator[list[int]]:
        indices = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for batch_index in range(self.next_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            yield indices[start : start + self.batch_size]

    def mark_consumed(self) -> None:
        self.next_batch += 1
        if self.next_batch == self.batches_per_epoch:
            self.epoch += 1
            self.next_batch = 0

    def state_dict(self) -> dict[str, int]:
        return {key: getattr(self, key) for key in ("size", "batch_size", "seed", "epoch", "next_batch")}

    def load_state_dict(self, state: dict[str, int]) -> None:
        for key in ("size", "batch_size", "seed"):
            if state[key] != getattr(self, key):
                raise ValueError(f"Resume sampler {key} changed")
        if state["epoch"] < 0 or not 0 <= state["next_batch"] < self.batches_per_epoch:
            raise ValueError("Invalid sampler cursor in checkpoint")
        self.epoch, self.next_batch = state["epoch"], state["next_batch"]


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def canonical_data_config(config: LiberoDatasetConfig) -> dict[str, Any]:
    values = asdict(config)
    for key in ("dataset_dir", "feature_cache_dir"):
        if values.get(key) is not None:
            values[key] = str(Path(values[key]).expanduser().resolve())
    for key in ("camera_keys", "proprio_keys"):
        if key in values:
            values[key] = list(values[key])
    return values


def build_optimizer(model: torch.nn.Module, train_values: dict[str, Any]) -> AdamW:
    """Include pretrained trainable weights, with their own fine-tuning rate."""
    model.encoder.initialize_trainable_backbones(next(model.parameters()).device)
    base_lr = float(train_values["learning_rate"])
    vision_lr = float(train_values.get("vision_learning_rate", base_lr))
    if not all(math.isfinite(rate) and rate > 0 for rate in (base_lr, vision_lr)):
        raise ValueError("Learning rates must be finite and positive")
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    groups = [{"params": [p for n, p in named if not n.startswith("encoder.vision.backbone.")],
               "lr": base_lr, "name": "policy"}]
    vision = [p for n, p in named if n.startswith("encoder.vision.backbone.")]
    if vision:
        groups.append({"params": vision, "lr": vision_lr, "name": "vision"})
    return AdamW(groups, lr=base_lr, weight_decay=float(train_values.get("weight_decay", 0.01)),
                 betas=tuple(train_values.get("betas", (0.9, 0.999))))


def main() -> None:
    args = arguments()
    values = yaml.safe_load(args.config.read_text())
    if not isinstance(values, dict) or not {"data", "model", "train"} <= values.keys():
        raise ValueError("Config requires data, model, and train mappings")
    data_values, train_values = dict(values["data"]), dict(values["train"])
    for name in ("max_steps", "batch_size", "num_workers"):
        if getattr(args, name) is not None:
            train_values[name] = getattr(args, name)
    if args.feature_cache_dir is not None:
        data_values["feature_cache_dir"] = str(args.feature_cache_dir)
    values["data"], values["train"] = data_values, train_values
    target = int(train_values["max_steps"])
    batch_size = int(train_values["batch_size"])
    workers = int(train_values.get("num_workers", 0))
    log_interval = int(train_values.get("log_interval", 100))
    save_interval = int(train_values.get("save_interval", 20000))
    if min(target, batch_size, log_interval, save_interval) < 1 or workers < 0:
        raise ValueError("Step/batch/interval values must be positive; num_workers must be nonnegative")
    warmup_steps = int(train_values.get("warmup_steps", 0))
    min_lr_ratio = float(train_values.get("min_lr_ratio", 1.0))
    if not 0 <= warmup_steps < target or not 0 < min_lr_ratio <= 1:
        raise ValueError("Require 0 <= warmup_steps < max_steps and 0 < min_lr_ratio <= 1")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    seed = int(train_values.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    dataset_config = LiberoDatasetConfig(**data_values)
    dataset = LiberoDataset(dataset_config)
    try:
        run_training(args, values, train_values, dataset, dataset_config, device, seed, target, batch_size,
                     workers, log_interval, save_interval, warmup_steps, min_lr_ratio)
    finally:
        dataset.close()


def run_training(args: argparse.Namespace, values: dict[str, Any], train_values: dict[str, Any],
                 dataset: LiberoDataset, dataset_config: LiberoDatasetConfig, device: torch.device,
                 seed: int, target: int, batch_size: int, workers: int, log_interval: int,
                 save_interval: int, warmup_steps: int, min_lr_ratio: float) -> None:
    model = build_policy(values["model"]).to(device)
    if dataset_config.horizon != model.action_head.config.horizon:
        raise ValueError("Dataset and action head horizons must match")
    dataset.validate_feature_cache(model.encoder.config)
    sampler = ResumableBatchSampler(len(dataset), batch_size, seed)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=workers, pin_memory=device.type == "cuda",
                        persistent_workers=workers > 0, generator=loader_generator)
    optimizer = build_optimizer(model, train_values)
    trainable = [value for value in model.parameters() if value.requires_grad]

    def lr_multiplier(scheduler_step: int) -> float:
        if warmup_steps and scheduler_step < warmup_steps:
            return (scheduler_step + 1) / warmup_steps
        progress = min(1.0, max(0.0, (scheduler_step - warmup_steps) / max(1, target - warmup_steps)))
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_multiplier)
    amp_name = str(train_values.get("amp_dtype", "float16"))
    if amp_name not in ("float16", "bfloat16"):
        raise ValueError("amp_dtype must be float16 or bfloat16")
    amp_dtype = getattr(torch, amp_name)
    amp_enabled = bool(train_values.get("amp", True)) and device.type == "cuda"
    if amp_enabled and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested bfloat16 is unsupported on this CUDA device")
    # torch.amp.GradScaler was introduced after the project's torch>=2.1 floor.
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    ema = EMA(model, float(train_values.get("ema_decay", 0.995)))
    step = 0
    stats_payload = None
    resume = read_checkpoint(args.resume) if args.resume else None
    if resume is not None:
        if resume.get("data_config") != canonical_data_config(dataset_config):
            raise ValueError("Checkpoint data configuration differs")
        if resume.get("task_names") != [task.name for task in dataset.tasks]:
            raise ValueError("Checkpoint dataset task list differs")
        if resume.get("data_signature") != dataset.cache_data_signature():
            raise ValueError("Dataset files, instructions, or image transforms changed since the checkpoint")
        # Schedule/data-order changes are explicit new experiments, not resumes.
        mutable = {"num_workers", "log_interval", "save_interval", "max_consecutive_optimizer_skips"}
        before = {key: value for key, value in resume["train_config"].items() if key not in mutable}
        after = {key: value for key, value in train_values.items() if key not in mutable}
        if before != after:
            raise ValueError("Training configuration changed on resume (including max_steps/schedule/batch size)")
        load_model_state(model, resume)
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        scaler.load_state_dict(resume["scaler"])
        if resume.get("ema_decay") != ema.decay:
            raise ValueError("EMA decay changed on resume")
        # Reuse strict parameter validation for EMA before assigning its state.
        load_model_state(model, resume, weights="ema")
        ema.shadow = {key: value.to(device) for key, value in resume["ema"].items()}
        load_model_state(model, resume)
        sampler.load_state_dict(resume["sampler"])
        loader_generator.set_state(resume["loader_generator"])
        restore_rng_state(resume["rng"])
        step = int(resume["step"])
        stats_payload = resume.get("statistics")
    else:
        stats = compute_statistics(dataset)
        model.action_normalizer.fit(stats.action_minimum.to(device), stats.action_maximum.to(device))
        model.encoder.set_proprio_statistics(stats.proprio_mean.to(device), stats.proprio_std.to(device))
        stats_payload = {name: value.tolist() for name, value in asdict(stats).items()}
    if not 0 <= step <= target:
        raise ValueError("Checkpoint step is outside the requested training range")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if not args.resume and (output / "latest.pt").exists():
        raise FileExistsError(f"{output} already contains a run; use --resume or a new --output-dir")
    (output / "config.yaml").write_text(yaml.safe_dump(values, sort_keys=False))
    model.train()  # Encoder.train() keeps its frozen backbones in eval mode.
    print(f"Training setup | device={device} | windows={len(dataset):,} | tasks={len(dataset.tasks)} | "
          f"trainable_parameters={sum(value.numel() for value in trainable):,} | "
          f"vision_trainable={model.encoder.config.vision_trainable} | text_trainable=False | "
          f"cached_inputs={dataset_config.feature_cache_dir is not None} | amp={amp_name if amp_enabled else 'off'}", flush=True)
    print("Optimizer groups | " + " | ".join(
        f"{group.get('name', 'policy')}: parameters={sum(p.numel() for p in group['params']):,}, base_lr={base_lr:g}"
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs)
    ), flush=True)
    if resume is not None:
        print("Resumed optimizer, EMA, RNG and consumed sample cursor. Worker-side random augmentation state is not saved.", flush=True)
    progress = tqdm(total=target, initial=step, desc="Training", unit="step", dynamic_ncols=True, mininterval=1)
    started = time.perf_counter()
    previous_elapsed = float(resume.get("elapsed_training_seconds", 0)) if resume else 0.0

    def save() -> None:
        payload = make_model_payload(model, ema)
        payload.update({
            "step": step, "data_config": canonical_data_config(dataset_config),
            "data_signature": dataset.cache_data_signature(),
            "train_config": train_values, "task_names": [task.name for task in dataset.tasks],
            "instructions": [task.instruction for task in dataset.tasks],
            "image_transform": dataset.image_transform, "statistics": stats_payload,
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
            "rng": capture_rng_state(), "sampler": sampler.state_dict(),
            "loader_generator": loader_generator.get_state(),
            "elapsed_training_seconds": previous_elapsed + time.perf_counter() - started,
        })
        destination = output / f"checkpoint_{step:06d}.pt"
        save_checkpoint(destination, payload)
        update_latest(destination)
        progress.write(f"Checkpoint saved | step={step:,} | path={destination}")

    iterator = iter(loader)
    consecutive_skips = 0
    max_skips = int(train_values.get("max_consecutive_optimizer_skips", 16))
    interrupted = False
    try:
        with (output / "metrics.jsonl").open("a") as metrics:
            while step < target:
                try:
                    batch = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    batch = next(iterator)
                sampler.mark_consumed()
                batch = move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                    losses = model.loss(batch)
                    loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite loss at completed step {step}")
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, float(train_values.get("max_grad_norm", 1.0)))
                if not torch.isfinite(gradient_norm) and not scaler.is_enabled():
                    raise FloatingPointError(f"Non-finite gradient norm at completed step {step}")
                if scaler.is_enabled():
                    previous_scale = float(scaler.get_scale())
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_ran = float(scaler.get_scale()) >= previous_scale
                else:
                    optimizer.step()
                    optimizer_ran = True
                if not optimizer_ran:
                    consecutive_skips += 1
                    if consecutive_skips >= max_skips:
                        raise FloatingPointError("Too many consecutive FP16 optimizer skips")
                    progress.set_postfix(amp_skip=consecutive_skips, scale=f"{scaler.get_scale():.0f}")
                    continue
                consecutive_skips = 0
                if step == 0 and model.encoder.config.vision_trainable:
                    vision_gradients = [p.grad for p in model.encoder.vision.parameters()
                                        if p.requires_grad and p.grad is not None]
                    if not vision_gradients:
                        raise RuntimeError("No gradient reached the trainable vision encoder")
                    vision_norm = torch.stack([g.detach().float().norm() for g in vision_gradients]).norm()
                    if not torch.isfinite(vision_norm) or vision_norm <= 0:
                        raise FloatingPointError("Vision encoder gradient must be finite and nonzero")
                    if any(p.requires_grad or p.grad is not None for p in model.encoder.text.parameters()):
                        raise RuntimeError("BERT must remain frozen")
                    progress.write(f"Backbone gradient check | vision_norm={float(vision_norm):.6g} | BERT=frozen")
                    del vision_gradients
                scheduler.step()
                ema.update(model)
                step += 1
                progress.update(1)
                if step % log_interval == 0 or step == 1:
                    record = {"step": step, **{key: float(value.detach()) for key, value in losses.items()},
                              "lr": optimizer.param_groups[0]["lr"], "gradient_norm": float(gradient_norm),
                              "learning_rates": {group.get("name", "policy"): group["lr"] for group in optimizer.param_groups},
                              "elapsed_seconds": time.perf_counter() - started}
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    progress.set_postfix(loss=f"{record['loss']:.4f}", flow=f"{record['flow_loss']:.4f}",
                                         rep=f"{record['representation_loss']:.4f}", lr=f"{record['lr']:.2e}")
                if step % save_interval == 0 or step == target:
                    save()
    except KeyboardInterrupt:
        interrupted = True
        # An interrupt can land inside optimizer.step(); do not label a
        # partially updated model as a completed-step checkpoint.
        progress.write("Interrupted; resume from the most recent completed checkpoint.")
    finally:
        progress.close()
    elapsed = time.perf_counter() - started
    print(f"Training {'interrupted' if interrupted else 'finished'} | step={step:,} | elapsed={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
