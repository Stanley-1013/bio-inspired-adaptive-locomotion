#!/usr/bin/env python3
"""
Phase 3 — evaluate every (condition, seed) checkpoint under controlled
scenarios (nominal + OOD perturbations), recording behavioural and
robustness metrics. Writes one CSV row per (policy, scenario).

Research principles enforced:
  - Each policy is evaluated in an env whose **bio-knob configuration matches
    its training-time config** (e.g., no_fatigue policy keeps `motor_fatigue=False`
    in eval). We do NOT do what `SATA/scripts/play.py` does, which forces
    activation_process / hill_model / motor_fatigue all True regardless.
  - `test.use_test=True` is set so `general_scale=1` from start
    (go2_torque.py:179) — full-grown actuators for all conditions.
  - Velocity command is fixed at (vx=1.0, vy=0, ω_yaw=0) via point ranges,
    not via the `control_type='T'` shortcut (which would override the
    no_growth ablation's training config in an unsafe way).
  - Domain randomisation is set to deterministic point values per scenario
    (no random sampling in eval); for OOD axes the value is OOD,
    for in-distribution axes it's the nominal midpoint.
  - Deterministic policy at inference (`get_inference_policy`) — no
    action noise — so each result reflects policy behaviour alone.

Usage:
    source ~/workspace/bio-inspired-adaptive-locomotion/scripts/sata-env.sh
    cd ~/workspace/bio-inspired-adaptive-locomotion
    python analysis/eval_under_conditions.py --condition reference --seed 1 \
        --scenario nominal --episodes 64
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# MUST be imported before torch (Isaac Gym constraint)
import isaacgym  # noqa: F401
from isaacgym import gymapi
from legged_gym.envs import *  # registers tasks  # noqa: F401, F403
from legged_gym.utils import task_registry

import numpy as np
import torch

# ---------- config ---------------------------------------------------------

LOGS_ROOT = Path.home() / "workspace/SATA/legged_gym/logs/SATA"

# (condition_label, run_name_prefix, task_name_in_registry)
CONDITION_INFO = {
    "reference":     ("ref",          "go2_torque"),
    "no_fatigue":    ("no_fatigue",   "go2_torque_no_fatigue"),
    "no_hill":       ("no_hill",      "go2_torque_no_hill"),
    "no_activation": ("no_activation","go2_torque_no_activation"),
    "no_growth":     ("no_growth",    "go2_torque_no_growth"),
    "hard_terrain":  ("hard_terrain", "go2_torque_hard_terrain"),
}


@dataclass
class Scenario:
    """One eval scenario — modifies one OOD axis from the in-distribution baseline."""
    name: str
    # domain-rand overrides
    randomize_base_mass: bool = True
    added_mass_range: tuple = (0.0, 0.0)         # point range = no randomness
    randomize_friction: bool = True
    friction_range: tuple = (0.875, 0.875)       # midpoint of train [0.5, 1.25]
    push_robots: bool = False
    push_interval_s: float = 4.0
    max_push_vel_xy: float = 0.0
    max_push_vel_ang: float = 0.0


# Scenario grid. Real Unitree Go2 payload spec: Air 7 kg / Pro 8 kg / EDU 12 kg;
# body mass ~15 kg; peak joint torque 45 N·m (sim uses a conservative 23.5 N·m).
# In-spec payloads (5, 8 kg) are the meaningful comparison points; 10/15 kg are
# beyond rated capacity and are kept only to characterise *how* each policy
# degrades (fall vs crouch), not as evidence of "robustness".
# Push: the original push_x_3 / push_x_5 (every 2 s) proved too severe — they
# crush every condition to ~20-35 reward, losing discriminative power. push_x_1p5
# is a training-boundary-magnitude push at the training interval (4 s) — gentle
# enough to discriminate.
SCENARIOS: dict[str, Scenario] = {
    "nominal":       Scenario("nominal"),
    "payload_5kg":   Scenario("payload_5kg",   added_mass_range=(5.0, 5.0)),
    "payload_8kg":   Scenario("payload_8kg",   added_mass_range=(8.0, 8.0)),    # Pro spec limit
    "payload_10kg":  Scenario("payload_10kg",  added_mass_range=(10.0, 10.0)),  # beyond-spec
    "payload_15kg":  Scenario("payload_15kg",  added_mass_range=(15.0, 15.0)),  # beyond-spec
    "push_x_1p5":    Scenario("push_x_1p5",    push_robots=True, max_push_vel_xy=1.5, push_interval_s=4.0),
    "push_x_3":      Scenario("push_x_3",      push_robots=True, max_push_vel_xy=3.0, push_interval_s=2.0),
    "push_x_5":      Scenario("push_x_5",      push_robots=True, max_push_vel_xy=5.0, push_interval_s=2.0),
}

# ---------- helpers --------------------------------------------------------

def find_run_dir(prefix: str, seed: int) -> Path:
    """Newest run dir matching *_<prefix>_s<seed>."""
    matches = sorted(glob.glob(str(LOGS_ROOT / f"*_{prefix}_s{seed}")))
    if not matches:
        raise FileNotFoundError(f"no run dir for prefix={prefix} seed={seed} under {LOGS_ROOT}")
    return Path(matches[-1])


def make_fake_args(task: str, load_run: str, num_envs: int = 64, seed: int = 0,
                   gpu: int = 0) -> argparse.Namespace:
    """Build the args object task_registry expects, without calling get_args()
    (which would parse our script's CLI). All sim-device defaults match
    Isaac Gym CPU/GPU pipeline conventions."""
    ns = argparse.Namespace(
        task=task,
        resume=True,
        experiment_name="SATA",
        run_name="",
        load_run=load_run,
        checkpoint=3000,
        headless=True,
        horovod=False,
        rl_device=f"cuda:{gpu}",
        num_envs=num_envs,
        seed=seed,
        max_iterations=1,
        # sim params (computed in helpers.parse_sim_params)
        physics_engine=gymapi.SIM_PHYSX,
        sim_device=f"cuda:{gpu}",
        sim_device_type="cuda",
        compute_device_id=gpu,
        graphics_device_id=gpu,
        num_threads=0,
        subscenes=0,
        slices=0,
        use_gpu=True,
        use_gpu_pipeline=True,
    )
    return ns


def apply_eval_overrides(cfg, scenario: Scenario, num_envs: int) -> None:
    """Apply our eval-specific overrides to env_cfg in place.
    Critically does NOT touch cfg.control.* — those keep the ablation's training settings."""

    cfg.env.num_envs = num_envs
    cfg.env.episode_length_s = 20.0          # longer than training (10s) to see late degradation

    # use_test = True forces general_scale=1 from iter 0 (go2_torque.py:179),
    # i.e. every condition runs with fully grown actuators in eval.
    cfg.test.use_test = True

    # Fixed forward command (vx=1, vy=0, ω=0) via point-range trick — does NOT
    # depend on control_type, so safe for all ablations including no_growth.
    cfg.commands.heading_command = False
    cfg.commands.resampling_time = 1e6       # effectively never resample mid-episode
    cfg.commands.ranges.lin_vel_x = [1.0, 1.0]
    cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    cfg.commands.ranges.heading = [0.0, 0.0]

    # No observation noise in eval (clean policy assessment).
    cfg.noise.add_noise = False

    # Domain-rand overrides per scenario. Anything we don't set here keeps the
    # training-time default, but we set all the noisy knobs explicitly so eval
    # is reproducible.
    cfg.domain_rand.randomize_friction = scenario.randomize_friction
    cfg.domain_rand.friction_range = list(scenario.friction_range)
    cfg.domain_rand.randomize_base_mass = scenario.randomize_base_mass
    cfg.domain_rand.added_mass_range = list(scenario.added_mass_range)
    cfg.domain_rand.push_robots = scenario.push_robots
    cfg.domain_rand.push_interval_s = scenario.push_interval_s
    cfg.domain_rand.max_push_vel_xy = scenario.max_push_vel_xy
    cfg.domain_rand.max_push_vel_ang = scenario.max_push_vel_ang
    cfg.domain_rand.loss_action_obs = False  # eval is clean


# ---------- the actual eval loop --------------------------------------------

def run_eval(condition: str, seed: int, scenario_name: str,
             episodes_target: int = 64, num_envs: int = 64, gpu: int = 0) -> dict:
    if condition not in CONDITION_INFO:
        raise ValueError(f"unknown condition '{condition}'")
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario '{scenario_name}'")
    prefix, task = CONDITION_INFO[condition]
    scenario = SCENARIOS[scenario_name]

    run_dir = find_run_dir(prefix, seed)
    load_run = run_dir.name

    print(f"  load_run={load_run}  task={task}  scenario={scenario_name}")

    args = make_fake_args(task=task, load_run=load_run, num_envs=num_envs, seed=seed, gpu=gpu)
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    apply_eval_overrides(env_cfg, scenario, num_envs=num_envs)

    train_cfg.runner.resume = True
    train_cfg.runner.load_run = load_run
    train_cfg.runner.checkpoint = 3000

    env, _ = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    ppo_runner, _ = task_registry.make_alg_runner(env=env, name=task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    # Per-env accumulators across the current episode
    n = num_envs
    device = env.device
    ep_steps = torch.zeros(n, dtype=torch.long, device=device)
    ep_reward = torch.zeros(n, device=device)
    ep_vx_abs_err = torch.zeros(n, device=device)
    ep_vy_abs_err = torch.zeros(n, device=device)
    ep_height_sum = torch.zeros(n, device=device)
    ep_below_thresh = torch.zeros(n, dtype=torch.long, device=device)  # steps below 0.22 m
    ep_energy = torch.zeros(n, device=device)
    ep_torque_peak = torch.zeros(n, device=device)
    ep_torque_peak_per_dof = torch.zeros((n, env.num_dof), device=device)  # NEW: per-DOF peak
    ep_energy_per_dof = torch.zeros((n, env.num_dof), device=device)       # NEW: per-DOF energy
    ep_jerk_sum = torch.zeros(n, device=device)
    ep_fatigue_max_per_dof = torch.zeros((n, env.num_dof), device=device)

    prev_action = torch.zeros((n, env.num_actions), device=device)
    have_prev = torch.zeros(n, dtype=torch.bool, device=device)

    # Per-episode results we'll average over after the run
    results = {
        "reward": [], "ep_length": [],
        "vx_abs_err": [], "vy_abs_err": [],
        "mean_height": [], "below_thresh_frac": [],
        "energy_per_step": [], "torque_peak": [],
        "torque_peak_std_acrossdofs": [],   # load-distribution proxy (works for no_fatigue)
        "energy_std_acrossdofs": [],        # load-distribution proxy (works for no_fatigue)
        "mean_jerk": [], "fatigue_max_acrossdofs": [],
        "fatigue_std_acrossdofs": [],
        "early_terminated": [],
        "survived": [],   # binary; matches SATA Table IV "success" convention
    }

    max_total_steps = 6000  # safety bound; ~30 s sim per env at 200 Hz
    step_idx = 0
    completed = 0
    print(f"  collecting >= {episodes_target} episodes (max {max_total_steps} sim steps)…")
    t0 = time.time()

    target_height = float(env_cfg.rewards.base_height_target)
    height_thresh = target_height - 0.08  # "low" if 8 cm below target

    while completed < episodes_target and step_idx < max_total_steps:
        actions = policy(obs.detach())
        prev = prev_action.clone()
        prev_action = actions.detach()
        # jerk = |Δaction| / dt (when we have a previous action for this episode)
        jerk = (actions - prev).abs().sum(dim=1) / env.dt
        jerk = torch.where(have_prev, jerk, torch.zeros_like(jerk))
        have_prev[:] = True

        obs, _, rews, dones, infos = env.step(actions.detach())

        # Per-step accumulation BEFORE handling dones, using values valid this step
        base_h = env.root_states[:, 2]
        vx = env.base_lin_vel[:, 0]
        vy = env.base_lin_vel[:, 1]
        cmd_x = env.commands[:, 0]
        cmd_y = env.commands[:, 1]
        torques = env.torques            # [n, num_dof]
        dof_vels = env.dof_vel
        power_per_dof = (torques * dof_vels).abs()
        energy_step = power_per_dof.sum(dim=1) * env.dt

        ep_steps += 1
        ep_reward += rews
        ep_height_sum += base_h
        ep_below_thresh += (base_h < height_thresh).long()
        ep_vx_abs_err += (vx - cmd_x).abs()
        ep_vy_abs_err += (vy - cmd_y).abs()
        ep_energy += energy_step
        ep_torque_peak = torch.maximum(ep_torque_peak, torques.abs().max(dim=1).values)
        ep_torque_peak_per_dof = torch.maximum(ep_torque_peak_per_dof, torques.abs())
        ep_energy_per_dof += power_per_dof * env.dt
        ep_jerk_sum += jerk
        ep_fatigue_max_per_dof = torch.maximum(ep_fatigue_max_per_dof, env.motor_fatigue.abs())

        # Whenever an env's episode just ended, snapshot its metrics into results.
        done_idx = torch.nonzero(dones, as_tuple=False).flatten()
        if done_idx.numel() > 0:
            steps = ep_steps[done_idx].float().clamp(min=1)
            results["reward"].extend(ep_reward[done_idx].cpu().tolist())
            results["ep_length"].extend(ep_steps[done_idx].cpu().tolist())
            results["vx_abs_err"].extend((ep_vx_abs_err[done_idx] / steps).cpu().tolist())
            results["vy_abs_err"].extend((ep_vy_abs_err[done_idx] / steps).cpu().tolist())
            results["mean_height"].extend((ep_height_sum[done_idx] / steps).cpu().tolist())
            results["below_thresh_frac"].extend((ep_below_thresh[done_idx].float() / steps).cpu().tolist())
            results["energy_per_step"].extend((ep_energy[done_idx] / steps).cpu().tolist())
            results["torque_peak"].extend(ep_torque_peak[done_idx].cpu().tolist())
            tp_per_dof = ep_torque_peak_per_dof[done_idx]   # [k, num_dof]
            results["torque_peak_std_acrossdofs"].extend(tp_per_dof.std(dim=1).cpu().tolist())
            energy_per_dof = ep_energy_per_dof[done_idx] / steps.unsqueeze(1)
            results["energy_std_acrossdofs"].extend(energy_per_dof.std(dim=1).cpu().tolist())
            results["mean_jerk"].extend((ep_jerk_sum[done_idx] / steps).cpu().tolist())
            fmax = ep_fatigue_max_per_dof[done_idx]   # [k, num_dof]
            results["fatigue_max_acrossdofs"].extend(fmax.max(dim=1).values.cpu().tolist())
            results["fatigue_std_acrossdofs"].extend(fmax.std(dim=1).cpu().tolist())
            max_steps = int(env.max_episode_length)
            early = (ep_steps[done_idx] < max_steps - 1).long()
            results["early_terminated"].extend(early.cpu().tolist())
            # NOTE: this `survived` flag is miscalibrated and uninformative —
            # max_steps = int(env.max_episode_length) ≈ 4000 (nominal-dt cap) is
            # never reached by the variable-control-rate rollout (a no-fall episode
            # completes at ~3578 steps), so `survived` ≈ 0 everywhere. Read falls
            # off `ep_length` relatively instead. Kept only for CSV-schema stability.
            survived = (ep_steps[done_idx] >= max_steps - 1).long()
            results["survived"].extend(survived.cpu().tolist())
            completed += int(done_idx.numel())

            # zero out accumulators for these envs (the env auto-reset them)
            ep_steps[done_idx] = 0
            ep_reward[done_idx] = 0
            ep_vx_abs_err[done_idx] = 0
            ep_vy_abs_err[done_idx] = 0
            ep_height_sum[done_idx] = 0
            ep_below_thresh[done_idx] = 0
            ep_energy[done_idx] = 0
            ep_torque_peak[done_idx] = 0
            ep_torque_peak_per_dof[done_idx] = 0
            ep_energy_per_dof[done_idx] = 0
            ep_jerk_sum[done_idx] = 0
            ep_fatigue_max_per_dof[done_idx] = 0
            have_prev[done_idx] = False

        step_idx += 1

    elapsed = time.time() - t0
    print(f"  done: {completed} episodes in {step_idx} steps ({elapsed:.1f}s)")

    # Aggregate
    agg = {
        "condition": condition, "seed": seed, "scenario": scenario_name,
        "n_episodes": completed,
        "sim_steps": step_idx,
        "wall_s": round(elapsed, 1),
    }
    for k, v in results.items():
        if not v:
            agg[k] = float("nan")
            agg[f"{k}_std"] = float("nan")
            continue
        arr = np.array(v, dtype=float)
        agg[k] = float(arr.mean())
        agg[f"{k}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return agg


def existing_cells(csv_path: Path) -> set:
    """Return set of (condition, seed, scenario) tuples already in the CSV."""
    done = set()
    if csv_path.exists():
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    done.add((row["condition"], int(row["seed"]), row["scenario"]))
                except (KeyError, ValueError):
                    pass
    return done


def append_csv_row(csv_path: Path, agg: dict, fieldnames: list[str]) -> None:
    """Append a single result row; write header if file is new."""
    new = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow({k: agg.get(k, "") for k in fieldnames})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=list(CONDITION_INFO.keys()),
                    help="single-cell mode: which condition to evaluate")
    ap.add_argument("--seed", type=int, help="single-cell mode: which seed")
    ap.add_argument("--scenario", choices=list(SCENARIOS.keys()),
                    help="single-cell mode: which scenario")
    ap.add_argument("--episodes", type=int, default=64)
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--gpu", type=int, default=0, help="GPU index for sim + policy")
    ap.add_argument("--out-csv", type=Path,
                    help="single-cell mode: append this cell's result row to this CSV")
    ap.add_argument("--batch-csv", type=Path,
                    help="batch mode: spawn one subprocess per (condition, seed, scenario) "
                         "cell and append to this CSV; existing rows skipped (resumable)")
    ap.add_argument("--conditions", nargs="+", default=list(CONDITION_INFO.keys()),
                    help="batch mode: restrict to a subset of conditions")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(1, 9)),
                    help="batch mode: restrict to a subset of seeds")
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS.keys()),
                    help="batch mode: restrict to a subset of scenarios")
    args = ap.parse_args()

    # Single-cell mode (also used as the subprocess unit of batch mode).
    # Isaac Gym only allows ONE sim per process, so batch mode below spawns a
    # fresh subprocess per cell rather than looping in-process.
    if args.batch_csv is None:
        if not (args.condition and args.seed is not None and args.scenario):
            ap.error("single-cell mode needs --condition --seed --scenario")
        agg = run_eval(args.condition, args.seed, args.scenario,
                       episodes_target=args.episodes, num_envs=args.num_envs, gpu=args.gpu)
        print()
        print("=== aggregate ===")
        for k, v in agg.items():
            print(f"  {k:30s} {v:.4f}" if isinstance(v, float) else f"  {k:30s} {v}")
        if args.out_csv is not None:
            base = list(agg.keys())
            if "error" not in base:
                base.append("error")
            # If CSV exists, reuse its header so columns align across cells.
            if args.out_csv.exists():
                with args.out_csv.open() as f:
                    base = next(csv.reader(f))
            append_csv_row(args.out_csv, agg, base)
            print(f"  -> appended to {args.out_csv}")
        return

    # Batch mode: spawn one subprocess per (condition, seed, scenario) cell.
    csv_path = args.batch_csv.resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_cells(csv_path)
    todo = [
        (c, s, sc)
        for c in args.conditions
        for s in args.seeds
        for sc in args.scenarios
        if (c, s, sc) not in done
    ]
    total = len(args.conditions) * len(args.seeds) * len(args.scenarios)
    print(f"Batch mode: csv={csv_path}")
    print(f"  grid total {total}; already done {len(done)}; to run {len(todo)}")

    for i, (cond, seed, scen) in enumerate(todo):
        print(f"\n[{i+1}/{len(todo)}] {cond} s{seed} {scen}  @ {time.strftime('%H:%M:%S')}", flush=True)
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--condition", cond, "--seed", str(seed), "--scenario", scen,
            "--episodes", str(args.episodes), "--num-envs", str(args.num_envs),
            "--gpu", str(args.gpu),
            "--out-csv", str(csv_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
        if proc.returncode != 0:
            print(f"  !! subprocess exit {proc.returncode}; tail of stderr:")
            print("\n".join("    " + ln for ln in proc.stderr.strip().splitlines()[-8:]))
            # Record an error row so we don't retry it forever on resume.
            base = list(csv.reader(csv_path.open()))[0] if csv_path.exists() else \
                ["condition", "seed", "scenario", "n_episodes", "error"]
            append_csv_row(csv_path, {
                "condition": cond, "seed": seed, "scenario": scen,
                "n_episodes": 0, "error": f"exit{proc.returncode}",
            }, base)
        else:
            # The subprocess already appended its own row.
            tail = [ln for ln in proc.stdout.strip().splitlines() if "appended to" in ln]
            print("  " + (tail[-1] if tail else "done"))

    print(f"\nbatch done; CSV: {csv_path}  rows now: {len(existing_cells(csv_path))}")


if __name__ == "__main__":
    main()
