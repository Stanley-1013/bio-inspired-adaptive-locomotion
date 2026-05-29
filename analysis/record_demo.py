#!/usr/bin/env python3
"""
Record ONE focused disturbance clip of the SATA reference policy, with
slow-motion + camera zoom during the event and post-hoc on-frame annotations
(arrow / ring / subtitle). One event per invocation so each can be calibrated
and reviewed independently.

Events (--event):
  push          external lateral push (force pulse), stagger + recover
  leg_pull      persistent external force pulling one calf (like a hand grabbing
                the leg); robot resists and keeps balance. NOT motor failure.
  stairs        reference policy walking a stairs-heavy terrain
  vertical      downward impact on the base, absorb + recover
  low_friction  friction lowered to a slippery value (ground tinted as a cue)

Disturbances are REAL sim changes; arrows/labels are drawn afterwards on the
rendered frame (3D->2D projected) so an external force reads as external.
Slow-mo = denser frame capture during the event window (more frames at constant
playback fps). Needs DISPLAY (headless disables rendering in legged_gym).

Usage:
  DISPLAY=:0 python analysis/record_demo.py --event push --seed 1 --gpu 0 \
      --out results/phase3-bio-claims-and-robustness/videos/ev_push.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import eval_under_conditions as ev
from eval_under_conditions import (
    SCENARIOS, find_run_dir, make_fake_args, apply_eval_overrides,
)
task_registry = ev.task_registry
from isaacgym import gymapi, gymtorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
from overlay import annotate  # noqa: E402

# per-leg DOF order FL,FR,RL,RR -> calf dof 2,5,8,11; calf body names:
CALF_BODY = {"FL": "FL_calf", "FR": "FR_calf", "RL": "RL_calf", "RR": "RR_calf"}

# Each event: total clip seconds, (event_start, event_end) within the clip,
# terrain, and a label. The event window is slow-mo'd and zoomed.
EVENTS = {
    "push":        dict(secs=8.0, win=(3.0, 5.5), terrain="rough",
                        label="External push", kind="push"),
    "leg_pull":    dict(secs=9.0, win=(3.0, 6.5), terrain="rough",
                        label="External force pulling the leg", kind="leg_pull"),
    # stairs uses the hard_terrain policy (trained on a stairs-heavy mix), not
    # the slope-trained reference, so it actually handles stairs rather than
    # being shown out-of-distribution.
    "stairs":      dict(secs=9.0, win=(2.0, 8.0), terrain="stairs",
                        label="Stairs (hard_terrain policy)", kind="terrain",
                        policy="hard_terrain", side_cam=True),
    "vertical":    dict(secs=8.0, win=(3.0, 5.5), terrain="rough",
                        label="Vertical impact", kind="vertical"),
}


def project(gym, sim, cam, env, p, W, H):
    view = np.asarray(gym.get_camera_view_matrix(sim, env, cam)).reshape(4, 4)
    proj = np.asarray(gym.get_camera_proj_matrix(sim, env, cam)).reshape(4, 4)
    clip = np.array([p[0], p[1], p[2], 1.0]) @ view @ proj
    if clip[3] <= 1e-6:
        return None
    ndc = clip[:3] / clip[3]
    return ((ndc[0] * 0.5 + 0.5) * W, (1.0 - (ndc[1] * 0.5 + 0.5)) * H)


def record(event, seed, gpu, out_path, width, height, fps):
    cfg_ev = EVENTS[event]
    # which policy/task to load: default reference, or an ablation (e.g. stairs
    # uses the hard_terrain policy so it actually walks stairs).
    pol = cfg_ev.get("policy", "ref")
    POL = {"ref": ("ref", "go2_torque"),
           "hard_terrain": ("hard_terrain", "go2_torque_hard_terrain")}
    prefix, task = POL[pol]
    run_dir = find_run_dir(prefix, seed)
    args = make_fake_args(task=task, load_run=run_dir.name, num_envs=1, seed=seed, gpu=gpu)
    args.headless = False
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    apply_eval_overrides(env_cfg, SCENARIOS["nominal"], num_envs=1)
    if cfg_ev["terrain"] == "stairs":
        # stairs-heavy mix: [smooth, rough, stairs_up, stairs_down, discrete]
        env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.5, 0.5, 0.0]
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = run_dir.name
    train_cfg.runner.checkpoint = 3000

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    policy = runner.get_inference_policy(device=env.device)

    g, sim, e0, ah = env.gym, env.sim, env.envs[0], env.actor_handles[0]
    cam_props = gymapi.CameraProperties(); cam_props.width = width; cam_props.height = height
    cam = g.create_camera_sensor(e0, cam_props)
    num_bodies = g.get_actor_rigid_body_count(e0, ah)
    calf_bi = g.find_actor_rigid_body_handle(e0, ah, CALF_BODY["FL"])
    hot = gymapi.Vec3(0.95, 0.15, 0.15); base_col = gymapi.Vec3(0.8, 0.8, 0.85)

    ev_a, ev_b = cfg_ev["win"]
    secs = cfg_ev["secs"]
    n_steps = int(secs / env.dt)
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264",
                                quality=8, macro_block_size=8)
    FAR, NEAR = 2.0, 1.05          # follow-cam distance; zoom in during event
    recolored = False; lowfric_on = False

    print(f"  event={event} secs={secs} win=({ev_a},{ev_b}) steps={n_steps}")
    for i in range(n_steps):
        t = i * env.dt
        in_event = ev_a <= t < ev_b
        kind = cfg_ev["kind"]

        # ---- real disturbance ----
        forces = None
        if in_event and kind == "push":
            # one ~0.15 s pulse near the start of the window
            if t < ev_a + 0.15:
                forces = torch.zeros((num_bodies, 3), device=env.device)
                forces[0, 1] = 220.0
        elif in_event and kind == "leg_pull":
            # persistent up+out pull on the FL calf, like a hand tugging it
            forces = torch.zeros((num_bodies, 3), device=env.device)
            forces[calf_bi, 1] = 55.0
            forces[calf_bi, 2] = 45.0
            if not recolored:
                g.set_rigid_body_color(e0, ah, calf_bi, gymapi.MESH_VISUAL, hot); recolored = True
        elif in_event and kind == "vertical":
            if t < ev_a + 0.15:
                forces = torch.zeros((num_bodies, 3), device=env.device)
                forces[0, 2] = -260.0
        if forces is not None:
            g.apply_rigid_body_force_tensors(sim, gymtorch.unwrap_tensor(forces), None, gymapi.ENV_SPACE)

        want_lowfric = in_event and kind == "low_friction"
        if want_lowfric != lowfric_on:
            sp = g.get_actor_rigid_shape_properties(e0, ah)
            for s in sp:
                s.friction = 0.12 if want_lowfric else 1.0
            g.set_actor_rigid_shape_properties(e0, ah, sp); lowfric_on = want_lowfric
        if not in_event and recolored:
            g.set_rigid_body_color(e0, ah, calf_bi, gymapi.MESH_VISUAL, base_col); recolored = False

        actions = policy(obs.detach())
        obs, _, _, _, _ = env.step(actions.detach())

        # ---- slow-mo: capture every step during event, every 3rd otherwise ----
        stride = 1 if in_event else 3
        if i % stride != 0:
            continue

        base = env.root_states[0, :3].cpu().numpy()
        dist = NEAR if in_event else FAR
        if cfg_ev.get("side_cam"):
            # low side-on view so stair risers/steps are visible in profile
            g.set_camera_location(cam, e0,
                                  gymapi.Vec3(base[0] - 0.2, base[1] - dist * 1.3, base[2] + 0.25),
                                  gymapi.Vec3(base[0], base[1], base[2]))
        else:
            g.set_camera_location(cam, e0,
                                  gymapi.Vec3(base[0] - dist, base[1] - dist, base[2] + dist * 0.55),
                                  gymapi.Vec3(base[0], base[1], base[2] + 0.1))
        g.step_graphics(sim); g.render_all_camera_sensors(sim)
        img = g.get_camera_image(sim, e0, cam, gymapi.IMAGE_COLOR).reshape(height, width, 4)[:, :, :3]

        # ---- overlays ----
        scr = project(g, sim, cam, e0, (base[0], base[1], base[2] + 0.1), width, height)
        arrow = ring = None
        label = cfg_ev["label"] + ("  (slow-mo)" if in_event else "")
        if in_event and kind == "push" and scr:
            arrow = dict(tip=scr, angle_deg=90, color=(255, 60, 60), length=120)
        elif in_event and kind == "vertical" and scr:
            arrow = dict(tip=(scr[0], scr[1] - 30), angle_deg=-90, color=(255, 120, 40), length=120)
            ring = dict(center=scr, r=40)
        elif in_event and kind == "leg_pull":
            cp = env.rigid_body_states[0, calf_bi, :3].cpu().numpy()
            cscr = project(g, sim, cam, e0, (cp[0], cp[1], cp[2]), width, height)
            if cscr:
                arrow = dict(tip=cscr, angle_deg=60, color=(255, 60, 60), length=110)
        tint = None
        frame = annotate(img, subtitle=label, arrow=arrow, ring=ring)
        writer.append_data(np.ascontiguousarray(frame))

    writer.close()
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True, choices=list(EVENTS.keys()))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    record(a.event, a.seed, a.gpu, a.out, a.width, a.height, a.fps)


if __name__ == "__main__":
    main()
