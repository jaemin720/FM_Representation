#!/usr/bin/env python3
"""Evaluate a language-conditioned FM policy on fixed LIBERO initial states.

Simulator rollout handling is adapted from practice/DP/scripts/evaluate.py.
Task IDs select benchmark tasks only; the policy always receives real text.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from repr_fm import build_policy
from repr_fm.checkpoint import load_model_state, read_checkpoint


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "outputs/libero10_fm/latest.pt")
    parser.add_argument("--libero-config-dir", type=Path, required=True, help="Directory containing LIBERO config.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--instruction", help="Override the actual language input; requires exactly one task ID")
    parser.add_argument("--rollouts-per-task", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of simulator environments whose policy inference is batched together.")
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--execution-steps", type=int, default=None, help="Actions executed before replanning; default min(6, horizon)")
    parser.add_argument("--inference-steps", type=int, help="Override FM integration steps")
    parser.add_argument("--camera-height", type=int, default=256)
    parser.add_argument("--camera-width", type=int, default=256)
    parser.add_argument("--render-gpu-device-id", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/libero10_fm/evaluation.json")
    return parser.parse_args()


def live_proprio(observation: dict[str, object]) -> np.ndarray:
    return np.concatenate((np.asarray(observation["robot0_joint_pos"], dtype=np.float32), np.asarray(observation["robot0_gripper_qpos"], dtype=np.float32)))


def rollout_summary(args: argparse.Namespace, results: list[dict[str, object]], checkpoint: Path, complete: bool) -> dict[str, object]:
    tasks = {}
    for task_id in args.task_ids:
        task_results = [value for value in results if value["task_id"] == task_id]
        if not task_results: continue
        successes = sum(bool(value["success"]) for value in task_results)
        tasks[str(task_id)] = {
            "task_name": task_results[0]["task_name"],
            "instruction": task_results[0]["instruction"],
            "completed_rollouts": len(task_results),
            "successes": successes,
            "success_rate": successes / len(task_results),
            "mean_steps": float(np.mean([value["steps"] for value in task_results])),
        }
    macro_success = float(np.mean([task["success_rate"] for task in tasks.values()])) if tasks else None
    return {
        "checkpoint": str(checkpoint),
        "weights": args.weights,
        "suite": args.suite,
        "seed": args.seed,
        "simulator_seed": 0,
        "policy_seed_rule": "seed + task_id * 10000 + first_rollout_in_batch",
        "initial_state_rule": "rollout_index modulo number of official task initial states",
        "num_envs": args.num_envs,
        "rollouts_per_task": args.rollouts_per_task,
        "max_steps": args.max_steps,
        "execution_steps": args.execution_steps,
        "inference_steps": args.inference_steps,
        "instruction_override": args.instruction,
        "warmup_steps": args.warmup_steps,
        "camera_height": args.camera_height,
        "camera_width": args.camera_width,
        "complete": complete,
        "completed_rollouts": len(results),
        "macro_success": macro_success,
        "tasks": tasks,
        "rollouts": results,
    }


def save_results(args: argparse.Namespace, results: list[dict[str, object]], checkpoint: Path, complete: bool) -> None:
    summary = rollout_summary(args, results, checkpoint, complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(args.output)


def split_observations(observations: object, expected: int) -> list[Mapping[str, object]]:
    # LIBERO's vector wrappers return either a single mapping or an object
    # array whose items are mappings; ``set_init_state`` uses the latter.
    split = [observations] if isinstance(observations, Mapping) else list(observations)
    if len(split) != expected: raise ValueError("LIBERO vector observation batch size mismatch")
    if any(not isinstance(observation, Mapping) for observation in split):
        raise TypeError("LIBERO vector observations must contain mappings")
    return split


def main() -> None:
    args = arguments()
    if args.num_envs < 1: raise ValueError("num_envs must be at least 1")
    if args.rollouts_per_task < 1: raise ValueError("rollouts_per_task must be at least 1")
    if args.max_steps < 1 or args.warmup_steps < 0: raise ValueError("Invalid rollout step counts")
    if args.camera_height < 1 or args.camera_width < 1: raise ValueError("Camera dimensions must be positive")
    if len(set(args.task_ids)) != len(args.task_ids): raise ValueError("task-ids must be distinct")
    if args.instruction is not None and (len(args.task_ids) != 1 or not args.instruction.strip()):
        raise ValueError("--instruction requires exactly one task ID and nonempty text")
    if args.inference_steps is not None and args.inference_steps < 1:
        raise ValueError("inference-steps must be positive")
    # Forking after CUDA has initialized can copy a broken EGL/CUDA state into
    # the simulator workers. LIBERO's parallel evaluator must use fresh spawn
    # processes, before initializing policy backbones on the GPU.
    if args.num_envs > 1:
        start_method = multiprocessing.get_start_method(allow_none=True)
        if start_method is None:
            multiprocessing.set_start_method("spawn")
        elif start_method != "spawn":
            raise RuntimeError(
                "Parallel LIBERO evaluation requires multiprocessing start method "
                f"'spawn', but {start_method!r} is already active"
            )
    payload = read_checkpoint(args.checkpoint)
    model = build_policy(payload["model_config"])
    if model.action_head.config.action_dim != 7:
        raise ValueError("LIBERO evaluation requires a 7-dimensional action head")
    load_model_state(model, payload, weights=args.weights)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
    model.to(device).eval().requires_grad_(False)
    if args.execution_steps is None: args.execution_steps = min(6, model.action_head.config.horizon)
    if not 1 <= args.execution_steps <= model.action_head.config.horizon:
        raise ValueError("execution_steps must be within action horizon")
    if args.inference_steps is None: args.inference_steps = model.action_head.config.inference_steps
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config_dir.expanduser().resolve())
    # Importing LIBERO can initialize configuration and simulator dependencies.
    # Keep it below argparse so --help works without LIBERO installed.
    from libero.libero import benchmark
    from libero.libero.envs import DummyVectorEnv, OffScreenRenderEnv, SubprocVectorEnv
    suites = benchmark.get_benchmark_dict()
    if args.suite not in suites: raise ValueError(f"Unknown LIBERO suite: {args.suite}")
    suite = suites[args.suite](0)
    if any(task_id < 0 or task_id >= suite.n_tasks for task_id in args.task_ids):
        raise ValueError("task-ids are outside the selected suite")
    camera_keys = payload["data_config"]["camera_keys"]
    observation_keys = {"obs/agentview_rgb": "agentview_image", "obs/eye_in_hand_rgb": "robot0_eye_in_hand_image"}
    if any(key not in observation_keys for key in camera_keys): raise ValueError("Unsupported camera key in checkpoint")
    if list(payload["data_config"]["proprio_keys"]) != ["obs/joint_states", "obs/gripper_states"]:
        raise ValueError("Evaluator currently supports joint_states + gripper_states proprioception")
    transform = payload.get("image_transform", "none")
    if transform not in ("none", "rotate_180", "flip_vertical"):
        raise ValueError(f"Unsupported stored image transform: {transform}")
    results: list[dict[str, object]] = []
    total_rollouts = len(args.task_ids) * args.rollouts_per_task
    progress = tqdm(total=total_rollouts, desc="Evaluation", unit="rollout", dynamic_ncols=True)
    started = time.perf_counter()
    try:
        for task_id in args.task_ids:
            task = suite.get_task(task_id)
            instruction = args.instruction if args.instruction is not None else task.language
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError(f"Task {task_id} has no valid language instruction")
            env_count = min(args.num_envs, args.rollouts_per_task)
            environment_factory = partial(
                OffScreenRenderEnv,
                bddl_file_name=suite.get_task_bddl_file_path(task_id),
                camera_heights=args.camera_height,
                camera_widths=args.camera_width,
                render_gpu_device_id=args.render_gpu_device_id,
                horizon=args.max_steps + args.warmup_steps + 1,
            )
            vector_class = DummyVectorEnv if env_count == 1 else SubprocVectorEnv
            environment = vector_class([environment_factory for _ in range(env_count)])
            initial_states = suite.get_task_init_states(task_id)
            if not len(initial_states): raise ValueError(f"Task {task_id} has no official initial states")
            try:
                for first_rollout in range(0, args.rollouts_per_task, env_count):
                    rollout_ids = list(range(first_rollout, min(first_rollout + env_count, args.rollouts_per_task)))
                    slots = list(range(len(rollout_ids)))
                    warmup = np.zeros(7, np.float32); warmup[-1] = -1
                    environment.seed([0] * env_count)
                    environment.reset(id=slots)
                    raw_observations = environment.set_init_state(
                        np.stack([np.asarray(initial_states[rollout % len(initial_states)]) for rollout in rollout_ids]),
                        id=slots,
                    )
                    observations = split_observations(raw_observations, len(slots))
                    success_flags = np.asarray(environment.check_success(), dtype=bool)
                    successes = [bool(success_flags[slot]) for slot in slots]
                    steps = [0 for _ in slots]
                    active = [slot for slot in slots if not successes[slot]]
                    for _ in range(args.warmup_steps):
                        if not active: break
                        actions = np.repeat(warmup[None, :], len(active), axis=0)
                        raw_observations, rewards, dones, _ = environment.step(actions, id=active)
                        step_observations = split_observations(raw_observations, len(active))
                        success_flags = np.asarray(environment.check_success(), dtype=bool)
                        for index, slot in enumerate(active):
                            observations[slot] = step_observations[index]
                            successes[slot] = bool(rewards[index] > 0 or success_flags[slot])
                        active = [slot for index, slot in enumerate(active) if not successes[slot] and not bool(dones[index])]
                    generator = torch.Generator(device=device).manual_seed(args.seed + task_id * 10000 + first_rollout)
                    while active:
                        image_batch = []
                        for slot in active:
                            images = []
                            for key in camera_keys:
                                image = np.asarray(observations[slot][observation_keys[key]], dtype=np.uint8)
                                if transform == "rotate_180": image = np.ascontiguousarray(image[::-1, ::-1])
                                elif transform == "flip_vertical": image = np.ascontiguousarray(image[::-1])
                                images.append(torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))))
                            image_batch.append(torch.stack(images))
                        images = torch.stack(image_batch).to(device)
                        proprio = torch.from_numpy(np.stack([live_proprio(observations[slot]) for slot in active])).to(device)
                        batch = {"images": images, "proprio": proprio, "instructions": [instruction] * len(active)}
                        action_batches = model.sample(batch, generator=generator, inference_steps=args.inference_steps)[:, :args.execution_steps].cpu().numpy()
                        if not np.isfinite(action_batches).all():
                            raise FloatingPointError("Policy produced non-finite actions")
                        planned_actions = {slot: action_batches[index] for index, slot in enumerate(active)}
                        for chunk_index in range(args.execution_steps):
                            step_slots = [slot for slot in active if steps[slot] < args.max_steps]
                            if not step_slots:
                                active = []
                                break
                            actions = np.stack([np.clip(planned_actions[slot][chunk_index], -1, 1) for slot in step_slots])
                            raw_observations, rewards, dones, _ = environment.step(actions, id=step_slots)
                            step_observations = split_observations(raw_observations, len(step_slots))
                            success_flags = np.asarray(environment.check_success(), dtype=bool)
                            for index, slot in enumerate(step_slots):
                                observations[slot] = step_observations[index]; steps[slot] += 1
                                successes[slot] = bool(rewards[index] > 0 or success_flags[slot])
                            active = [slot for index, slot in enumerate(step_slots) if not successes[slot] and not bool(dones[index]) and steps[slot] < args.max_steps]
                            if not active: break
                    for slot, rollout in enumerate(rollout_ids):
                        result = {"task_id": task_id, "task_name": task.name, "instruction": instruction,
                                  "rollout": rollout, "initial_state_index": rollout % len(initial_states),
                                  "success": successes[slot], "steps": steps[slot]}
                        results.append(result); progress.update(1); progress.write(json.dumps(result))
                    save_results(args, results, args.checkpoint, complete=False)
            finally:
                environment.close()
    except KeyboardInterrupt:
        progress.write("Evaluation interrupted; partial results were saved.")
    finally:
        progress.close()
    complete = len(results) == total_rollouts
    save_results(args, results, args.checkpoint, complete=complete)
    summary = rollout_summary(args, results, args.checkpoint, complete)
    print(json.dumps({"macro_success": summary["macro_success"], "complete": complete, "output": str(args.output),
                      "elapsed_seconds": time.perf_counter() - started}), flush=True)


if __name__ == "__main__": main()
