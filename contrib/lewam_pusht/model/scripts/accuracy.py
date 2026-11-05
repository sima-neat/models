"""Score actual native closed-loop actions on explicitly frozen PushT episodes.

This adapter is staged, not yet a qualified replacement for the source evaluator.
The dataset must bind the exact reset seeds, recorded initial and goal images,
original time/horizon features and package versions; none is guessed. Physics
observations after step zero come only from actions returned by the native policy.
The original 150-row corpus is not held out and is not generalization evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path

import numpy as np
from performance import Policy, finite, parse_mpks

SELECTION_SHA256 = "07f15a12ac7c649758ac26783e3632ed7aba91c6e15f6ebe3b46bb0dc33e305a"


def success(goal, state) -> bool:
    goal, state = np.asarray(goal), np.asarray(state)
    if goal.shape != (7,) or state.shape != (7,) or not np.isfinite([goal, state]).all():
        raise ValueError("Finite seven-element states required")
    angle = abs(goal[4] - state[4])
    return bool(np.linalg.norm(goal[:4] - state[:4]) < 20 and min(angle, 2 * np.pi - angle) < np.pi / 9)


def load_episodes(dataset: Path):
    manifest = json.loads((dataset / "episodes.json").read_text())
    if (
        manifest.get("schema"),
        manifest.get("held_out"),
        manifest.get("goal_offset_steps"),
        manifest.get("eval_budget"),
        manifest.get("exec_actions"),
    ) != ("lewam.episodes.v1", False, 25, 50, 5):
        raise ValueError("Expected the explicit non-held-out direct-policy reproduction protocol")
    rows = manifest["episodes"]
    selection_path = dataset / "selection.json"
    if selection_path.is_symlink() or hashlib.sha256(selection_path.read_bytes()).hexdigest() != SELECTION_SHA256:
        raise ValueError("Frozen evaluation selection bytes changed")
    original = json.loads(selection_path.read_text())["episodes"]
    if [{key: row[key] for key in ("seed", "episode", "start", "goal_step")} for row in rows] != original:
        raise ValueError("Runtime episodes differ from the original selection")
    if len(rows) != 150 or [x["seed"] for x in rows] != [42] * 50 + [0] * 50 + [1] * 50:
        raise ValueError("Frozen seed-group order/count changed")
    identities = [(x["episode"], x["start"], x["goal_step"]) for x in rows]
    if len(set(identities)) != 150 or len({x[0] for x in identities}) != 148:
        raise ValueError("Frozen episode identity cardinality changed")
    if any(x[2] - x[1] != 25 for x in identities):
        raise ValueError("Goal offset changed")
    if any(type(x.get("reset_seed")) is not int for x in rows):
        raise ValueError("Source-recorded environment reset seeds required")
    path = dataset / "episodes.npz"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != manifest["arrays_sha256"]:
        raise ValueError("Episode arrays changed")
    with np.load(path, allow_pickle=False) as saved:
        arrays = dict(saved)
    expected = {"initial_states", "goal_states", "initial_images", "goal_images", "time_features", "horizon_features"}
    if set(arrays) != expected:
        raise ValueError("Unexpected episode arrays")
    for key in ("initial_states", "goal_states"):
        if arrays[key].shape != (150, 7) or arrays[key].dtype != np.float64 or not np.isfinite(arrays[key]).all():
            raise ValueError(f"Invalid {key}")
    for key in ("initial_images", "goal_images"):
        if arrays[key].shape != (150, 224, 224, 3) or arrays[key].dtype != np.uint8:
            raise ValueError(f"Source-recorded RGB224 uint8 {key} required")
    finite(arrays["time_features"], (8, 192), "source time features")
    finite(arrays["horizon_features"], (10, 192), "source horizon features for all ten replans")
    required = {"stable-worldmodel", "pymunk", "pygame", "torch"}
    if set(manifest.get("environment", {})) != required:
        raise ValueError("Complete physics/rendering/RNG package version binding required")
    for package, expected_version in manifest["environment"].items():
        if importlib.metadata.version(package) != expected_version:
            raise ValueError(f"Source/candidate environment mismatch: {package}")
    return manifest, arrays


def evaluate(policy, rows, arrays, environment_factory, noise_factory):
    """Lockstep cohorts preserve the source's alive-batch noise draw order."""
    results = []
    for group in range(3):
        offset = group * 50
        environments, frames, queues = [], [{} for _ in range(50)], [[] for _ in range(50)]
        done = np.zeros(50, np.bool_)
        steps = np.zeros(50, np.int64)
        noise = noise_factory(rows[offset]["seed"])
        try:
            for i in range(50):
                env = environment_factory()
                environments.append(env)
                env.reset(seed=rows[offset + i]["reset_seed"])
                # Preserve upstream callable order; _set_state advances one physics tick.
                env._set_state(arrays["initial_states"][offset + i])
                env._set_goal_state(arrays["goal_states"][offset + i])
            for step in range(50):
                alive = np.flatnonzero(~done).tolist()
                if not alive:
                    break
                for i in alive:
                    # Upstream supplies recorded pixels at step zero, not a post-reset render.
                    image = arrays["initial_images"][offset + i] if step == 0 else environments[i].render()
                    frames[i][step] = image.copy()
                    frames[i] = {k: v for k, v in frames[i].items() if k >= step - 5}
                replanning = [i for i in alive if not queues[i]]
                if replanning:
                    initial_noise = noise(len(replanning))
                    for row, i in enumerate(replanning):
                        images = ([frames[i][step - 5]] if step >= 5 else []) + [frames[i][step]]
                        images += [arrays["goal_images"][offset + i]]
                        request = {
                            "images": np.stack(images),
                            "history_pad": np.array([step < 5, False]),
                            "noise": initial_noise[row],
                            "time_features": arrays["time_features"],
                            "horizon_features": arrays["horizon_features"][step // 5],
                        }
                        actions = finite(policy.replan(request), (10, 2), "native actions")
                        queues[i] = list(actions[:5].copy())
                for i in alive:
                    observation, _, terminated, truncated, _ = environments[i].step(queues[i].pop(0))
                    steps[i] += 1
                    reached = success(arrays["goal_states"][offset + i], observation["state"])
                    if bool(terminated) != reached or truncated:
                        raise RuntimeError("Physics termination differs from the pinned success predicate")
                    done[i] = reached
            results.extend(
                {
                    "episode": rows[offset + i]["episode"],
                    "start": rows[offset + i]["start"],
                    "seed": rows[offset + i]["seed"],
                    "success": bool(done[i]),
                    "steps": int(steps[i]),
                }
                for i in range(50)
            )
        finally:
            close_errors = []
            for env in environments:
                try:
                    env.close()
                except Exception as error:
                    close_errors.append(error)
            if close_errors:
                raise RuntimeError(f"{len(close_errors)} PushT environment(s) failed to close") from close_errors[0]
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mpk", action="append", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_id, directory = args.dataset.split("=", 1)
    manifest, arrays = load_episodes(Path(directory))
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import model_report
    import torch
    from stable_worldmodel.envs.pusht import PushT

    def noise_factory(seed):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        return lambda count: torch.randn(count, 10, 2, generator=generator, dtype=torch.float32).numpy()

    policy = Policy(parse_mpks(args.mpk))
    try:
        results = evaluate(
            policy,
            manifest["episodes"],
            arrays,
            lambda: PushT(resolution=224, relative=True, render_mode="rgb_array"),
            noise_factory,
        )
    finally:
        policy.close()
    report = model_report.from_accuracy(
        {
            dataset_id: {
                "metrics": {"pusht_success_rate": sum(x["success"] for x in results) / len(results)},
                "split": "reproduction-non-held-out",
                "evaluated_samples": len(results),
                "available_samples": 150,
                "details": {
                    "held_out": False,
                    "mode": "direct_policy",
                    "episodes": results,
                    "goal_offset_steps": 25,
                    "eval_budget": 50,
                    "executed_prefix": 5,
                    "feedback": "native_actions_to_physics",
                    "adapter_qualification": "pending",
                },
            }
        }
    )
    report.save(args.output)


if __name__ == "__main__":
    main()
