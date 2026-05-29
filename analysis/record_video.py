#!/usr/bin/env python3
"""
Record an mp4 of a trained (condition, seed) policy under a scenario, using
Isaac Gym off-screen camera sensors (no viewer / VNC needed — renders straight
to image frames on the GPU). A follow-camera keeps the robot centred.

Reuses eval_under_conditions.py for policy loading + scenario config, so the
recorded policy runs in the SAME env it was evaluated in (respects each
ablation's bio-knob config; respects the residual for ref_residual).

Usage (sata conda env; graphics device = the --gpu):
    python analysis/record_video.py --condition reference --scenario payload_10kg \
        --seed 1 --gpu 0 --seconds 12 --out results/.../videos/ref_payload10.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import eval_under_conditions as ev
from eval_under_conditions import (
    CONDITION_INFO, SCENARIOS, find_run_dir, make_fake_args, apply_eval_overrides,
)
task_registry = ev.task_registry
from isaacgym import gymapi  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402


def record(condition, seed, scenario, gpu, seconds, out_path, stride, fps,
           width, height, residual_overrides=None):
    prefix, task = CONDITION_INFO[condition]
    sc = SCENARIOS[scenario]
    run_dir = find_run_dir(prefix, seed)
    print(f"recording {condition} s{seed} ({scenario}) from {run_dir.name}")

    args = make_fake_args(task=task, load_run=run_dir.name, num_envs=1, seed=seed, gpu=gpu)
    # NOTE: legged_gym's base_task sets graphics_device_id = -1 when headless=True,
    # which disables ALL rendering including off-screen camera sensors. So we must
    # run with headless=False to keep a valid graphics device. This opens an
    # interactive viewer window too (needs a display, e.g. DISPLAY=:0 over VNC),
    # but we ignore it and capture frames from our own camera sensor.
    args.headless = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    apply_eval_overrides(env_cfg, sc, num_envs=1)
    if residual_overrides and hasattr(env_cfg, "residual"):
        for k, v in residual_overrides.items():
            setattr(env_cfg.residual, k, v)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_dir.name
    train_cfg.runner.checkpoint = 3000

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    # Off-screen camera sensor attached to env 0.
    cam_props = gymapi.CameraProperties()
    cam_props.width = width
    cam_props.height = height
    cam_props.enable_tensors = False
    cam = env.gym.create_camera_sensor(env.envs[0], cam_props)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=8)

    n_steps = int(seconds / env.dt)
    print(f"  {n_steps} sim steps, capturing every {stride} → ~{n_steps//stride} frames")
    for i in range(n_steps):
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())
        if i % stride != 0:
            continue
        # follow camera: behind-and-above the base, looking at it
        base = env.root_states[0, :3].cpu().numpy()
        env.gym.set_camera_location(
            cam, env.envs[0],
            gymapi.Vec3(base[0] - 1.8, base[1] - 1.8, base[2] + 1.0),
            gymapi.Vec3(base[0], base[1], base[2] + 0.15),
        )
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        img = env.gym.get_camera_image(env.sim, env.envs[0], cam, gymapi.IMAGE_COLOR)
        img = img.reshape(height, width, 4)[:, :, :3]  # RGBA → RGB
        writer.append_data(np.ascontiguousarray(img))
    writer.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="reference", choices=list(CONDITION_INFO.keys()))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--scenario", default="nominal", choices=list(SCENARIOS.keys()))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--stride", type=int, default=4, help="capture every Nth sim step")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--out", required=True)
    # residual overrides (for ref_residual videos)
    ap.add_argument("--residual-sign", type=float, default=None)
    ap.add_argument("--residual-kp", type=float, default=None)
    ap.add_argument("--residual-tau-cap", type=float, default=None)
    args = ap.parse_args()

    ro = {}
    if args.residual_sign is not None: ro["sign"] = args.residual_sign
    if args.residual_kp is not None: ro["kp"] = args.residual_kp
    if args.residual_tau_cap is not None: ro["tau_cap"] = args.residual_tau_cap

    record(args.condition, args.seed, args.scenario, args.gpu, args.seconds,
           args.out, args.stride, args.fps, args.width, args.height,
           residual_overrides=ro or None)


if __name__ == "__main__":
    main()
