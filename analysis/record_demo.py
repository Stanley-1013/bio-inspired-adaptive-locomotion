#!/usr/bin/env python3
"""
Record the integrated "compliance under sequential disturbances" demo of the
SATA reference policy: walk → side push → rough terrain → single-leg torque
limit → vertical impact → low friction → recover, as one continuous clip with
a follow-camera and post-hoc on-frame annotations (arrows / rings / subtitles).

Disturbances are REAL (forces applied in the env; leg-torque limit and friction
really changed; leg really recoloured). The arrows/labels are drawn on the
rendered RGB frame afterwards (see overlay.py) purely to make an external force
read as external rather than as the robot thrashing on its own.

Runs the reference policy unchanged; needs a display (DISPLAY=:0) because
legged_gym only renders with graphics enabled (headless=False).

Usage:
    DISPLAY=:0 python analysis/record_demo.py --seed 1 --gpu 0 \
        --out results/phase3-bio-claims-and-robustness/videos/demo_compliance.mp4
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import eval_under_conditions as ev
from eval_under_conditions import (
    CONDITION_INFO, SCENARIOS, find_run_dir, make_fake_args, apply_eval_overrides,
)
task_registry = ev.task_registry
from isaacgym import gymapi, gymtorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from overlay import annotate  # noqa: E402


def project_world_to_screen(gym, sim, cam, env, p_world, W, H):
    """Project a 3D world point to (x,y) pixel coords using the camera's
    view+projection matrices. Returns (x, y) or None if behind camera."""
    view = np.asarray(gym.get_camera_view_matrix(sim, env, cam)).reshape(4, 4)
    proj = np.asarray(gym.get_camera_proj_matrix(sim, env, cam)).reshape(4, 4)
    p = np.array([p_world[0], p_world[1], p_world[2], 1.0])
    clip = p @ view @ proj          # row-vector convention used by Isaac Gym
    if abs(clip[3]) < 1e-6 or clip[3] < 0:
        return None
    ndc = clip[:3] / clip[3]
    x = (ndc[0] * 0.5 + 0.5) * W
    y = (1.0 - (ndc[1] * 0.5 + 0.5)) * H
    return (float(x), float(y))


# ---- disturbance timeline (seconds) ---------------------------------------
# Each entry: (t_start, t_end, kind, params). Times are wall-clock sim seconds.
def build_timeline():
    # Forces are applied EVERY sim step within [t_start,t_end); keep that window
    # short (~0.1s) and the magnitude modest so it's a stagger-and-recover, not a
    # knockdown. The arrow is shown for a slightly longer label window so the
    # viewer can register it.
    return [
        (0.0, 4.0,  "walk",     dict(label="Walking — rough terrain")),
        (4.0, 4.12, "push",     dict(label="External push  →", fx=-90.0, fy=0.0,
                                     arrow_angle=0, color=(255, 60, 60))),
        (4.12, 4.8, "pushlabel", dict(label="External push  →",
                                      arrow_angle=0, color=(255, 60, 60))),
        (4.8, 7.0,  "recover",  dict(label="...recovering")),
        (7.0, 7.12, "push",     dict(label="←  External push", fx=85.0, fy=30.0,
                                     arrow_angle=180, color=(255, 60, 60))),
        (7.12, 7.8, "pushlabel", dict(label="←  External push",
                                      arrow_angle=180, color=(255, 60, 60))),
        (7.8, 10.0, "recover",  dict(label="...recovering")),
        (10.0, 13.0, "leglimit", dict(label="Front-left leg torque limited",
                                       leg="FL")),
        (13.0, 13.12, "stomp",   dict(label="Vertical impact  ↓", fz=-140.0)),
        (13.12, 13.8, "stomplabel", dict(label="Vertical impact  ↓")),
        (13.8, 16.0, "recover", dict(label="...recovering")),
        (16.0, 19.0, "lowfric", dict(label="Low-friction ground")),
        (19.0, 22.0, "walk",    dict(label="Recovered — walking on")),
    ]


def phase_at(timeline, t):
    for (a, b, kind, p) in timeline:
        if a <= t < b:
            return kind, p, (a, b)
    return timeline[-1][2], timeline[-1][3], (timeline[-1][0], timeline[-1][1])


# DOF / body index helpers (per-leg order FL,FR,RL,RR; calf = idx 2,5,8,11)
LEG_CALF_DOF = {"FL": 2, "FR": 5, "RL": 8, "RR": 11}
LEG_CALF_BODYNAME = {"FL": "FL_calf", "FR": "FR_calf", "RL": "RL_calf", "RR": "RR_calf"}


def record(seed, gpu, out_path, width, height, fps, stride):
    task = "go2_torque"
    run_dir = find_run_dir("ref", seed)
    args = make_fake_args(task=task, load_run=run_dir.name, num_envs=1, seed=seed, gpu=gpu)
    args.headless = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    apply_eval_overrides(env_cfg, SCENARIOS["nominal"], num_envs=1)
    # keep a steady forward command; real disturbances are injected below
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_dir.name
    train_cfg.runner.checkpoint = 3000

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    g, sim, e0 = env.gym, env.sim, env.envs[0]
    cam_props = gymapi.CameraProperties(); cam_props.width = width; cam_props.height = height
    cam = g.create_camera_sensor(e0, cam_props)

    # body index of the base and a force tensor [num_bodies, 3] for apply_*_force
    base_idx = 0
    num_bodies = g.get_actor_rigid_body_count(e0, env.actor_handles[0])
    default_color = gymapi.Vec3(0.8, 0.8, 0.85)
    hot_color = gymapi.Vec3(0.95, 0.15, 0.15)

    timeline = build_timeline()
    total_s = timeline[-1][1]
    n_steps = int(total_s / env.dt)

    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=8)
    print(f"  {n_steps} steps ({total_s}s), every {stride} → ~{n_steps//stride} frames")

    leg_recolored = None
    for i in range(n_steps):
        t = i * env.dt
        kind, p, (ta, tb) = phase_at(timeline, t)

        # --- apply REAL disturbance for this phase -------------------------
        if kind == "push":
            forces = torch.zeros((num_bodies, 3), device=env.device)
            forces[base_idx, 0] = p.get("fx", 0.0)
            forces[base_idx, 1] = p.get("fy", 0.0)
            g.apply_rigid_body_force_tensors(
                sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
        elif kind == "stomp":
            forces = torch.zeros((num_bodies, 3), device=env.device)
            forces[base_idx, 2] = p.get("fz", 0.0)
            g.apply_rigid_body_force_tensors(
                sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)
        elif kind == "leglimit":
            env.low_torque = True   # SATA flag: scales front-leg torque to 0.2x
            if leg_recolored != p["leg"]:
                bi = g.find_actor_rigid_body_handle(e0, env.actor_handles[0],
                                                    LEG_CALF_BODYNAME[p["leg"]])
                g.set_rigid_body_color(e0, env.actor_handles[0], bi,
                                       gymapi.MESH_VISUAL, hot_color)
                leg_recolored = p["leg"]
        else:
            if getattr(env, "low_torque", False):
                env.low_torque = False
            if leg_recolored is not None:
                bi = g.find_actor_rigid_body_handle(e0, env.actor_handles[0],
                                                    LEG_CALF_BODYNAME[leg_recolored])
                g.set_rigid_body_color(e0, env.actor_handles[0], bi,
                                       gymapi.MESH_VISUAL, default_color)
                leg_recolored = None

        # real friction change for the low-friction phase (set once on enter/exit)
        want_lowfric = (kind == "lowfric")
        if want_lowfric != getattr(env, "_lowfric_on", False):
            sp = g.get_actor_rigid_shape_properties(e0, env.actor_handles[0])
            for s in sp:
                s.friction = 0.15 if want_lowfric else 1.0
            g.set_actor_rigid_shape_properties(e0, env.actor_handles[0], sp)
            env._lowfric_on = want_lowfric

        # --- step the policy/env ------------------------------------------
        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        if i % stride != 0:
            continue

        # --- follow camera + render ---------------------------------------
        base = env.root_states[0, :3].cpu().numpy()
        g.set_camera_location(cam, e0,
                              gymapi.Vec3(base[0] - 1.9, base[1] - 1.9, base[2] + 1.0),
                              gymapi.Vec3(base[0], base[1], base[2] + 0.15))
        g.step_graphics(sim)
        g.render_all_camera_sensors(sim)
        img = g.get_camera_image(sim, e0, cam, gymapi.IMAGE_COLOR).reshape(height, width, 4)[:, :, :3]

        # --- overlays ------------------------------------------------------
        scr = project_world_to_screen(g, sim, cam, e0,
                                       (base[0], base[1], base[2] + 0.15), width, height)
        arrow = ring = None
        if kind in ("push", "pushlabel") and scr:
            arrow = dict(tip=scr, angle_deg=p.get("arrow_angle", 0),
                         color=p.get("color", (255, 60, 60)), length=110)
        elif kind in ("stomp", "stomplabel") and scr:
            arrow = dict(tip=(scr[0], scr[1] - 30), angle_deg=-90, color=(255, 120, 40), length=120)
            ring = dict(center=scr, r=38)
        frame = annotate(img, subtitle=p.get("label"), arrow=arrow, ring=ring)
        writer.append_data(np.ascontiguousarray(frame))

    writer.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    record(a.seed, a.gpu, a.out, a.width, a.height, a.fps, a.stride)


if __name__ == "__main__":
    main()
