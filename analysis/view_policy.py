#!/usr/bin/env python3
"""
Open an Isaac Gym viewer and run a trained (condition, seed) policy continuously
so it can be watched over VNC. Reuses the loading + config logic from
eval_under_conditions.py (so the ablation's training-time bio-knob config is
respected — unlike SATA's play.py which forces all bio knobs True).

Usage (on the lab box, with the VNC display):
    source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh
    cd ~/workspace/bio-inspired-adaptive-locomotion
    DISPLAY=:0 python analysis/view_policy.py --condition reference --seed 1 --gpu 2

Runs until the viewer window is closed or the process is killed.
"""

from __future__ import annotations

import argparse

# Import eval_under_conditions FIRST: it performs the isaacgym -> legged_gym.envs
# -> legged_gym.utils import sequence in the one order that avoids legged_gym's
# circular import. We then reuse its already-initialised task_registry.
import eval_under_conditions as ev
from eval_under_conditions import (
    CONDITION_INFO, SCENARIOS, find_run_dir, make_fake_args, apply_eval_overrides,
)

task_registry = ev.task_registry
from isaacgym import gymapi  # noqa: E402  (isaacgym already imported by ev)
import torch  # noqa: E402  (safe now — isaacgym already imported by ev)


def view(condition: str, seed: int, scenario: str, gpu: int, num_envs: int):
    prefix, task = CONDITION_INFO[condition]
    sc = SCENARIOS[scenario]
    run_dir = find_run_dir(prefix, seed)
    print(f"viewing {condition} s{seed} ({scenario}) from {run_dir.name} on cuda:{gpu}")

    args = make_fake_args(task=task, load_run=run_dir.name, num_envs=num_envs, seed=seed, gpu=gpu)
    args.headless = False  # <- the whole point: create a viewer

    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    apply_eval_overrides(env_cfg, sc, num_envs=num_envs)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_dir.name
    train_cfg.runner.checkpoint = 3000

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    print("viewer running — press 'v' in the window to pause, close window to quit")
    follow = True
    step = 0
    while True:
        # stop if the viewer window was closed
        if env.viewer is not None and env.gym.query_viewer_has_closed(env.viewer):
            print("viewer closed; exiting")
            break
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        # Follow camera: keep the (first) robot centred so it never walks out of
        # frame. Offset is behind-and-above, looking at the robot base.
        if follow and env.viewer is not None:
            step += 1
            if step % 2 == 0:  # update every other step is smooth enough, cheaper
                base = env.root_states[0, :3].cpu().numpy()
                cam_pos = gymapi.Vec3(base[0] - 2.0, base[1] - 2.0, base[2] + 1.2)
                cam_tgt = gymapi.Vec3(base[0], base[1], base[2] + 0.2)
                env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="reference", choices=list(CONDITION_INFO.keys()))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenario", default="nominal", choices=list(SCENARIOS.keys()))
    ap.add_argument("--gpu", type=int, default=2)
    ap.add_argument("--num-envs", type=int, default=1)
    args = ap.parse_args()
    view(args.condition, args.seed, args.scenario, args.gpu, args.num_envs)


if __name__ == "__main__":
    main()
