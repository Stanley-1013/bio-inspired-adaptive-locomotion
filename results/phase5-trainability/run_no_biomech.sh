#!/usr/bin/env bash
# Phase 5 — trainability probe: reproduce the paper's "SATA w/o biomechanical
# model" ablation (§V-A1), which the paper reports is "completely unable to
# learn a coherent gait".
#
# Single GPU, five seeds run SERIALLY (one GPU only, per request). Full
# 3000 iters / 4096 envs to match the Phase 1 reference exactly, so the only
# difference vs reference is the biomechanical model being off.
#
# Detached / overnight-safe: launch with the companion launch line below so it
# survives shell + Claude disconnects. Expect ~75 min/seed → ~6.3 h total.
set -uo pipefail

GPU="${GPU:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ITERS="${ITERS:-3000}"
ENVS="${ENVS:-4096}"
TASK=go2_torque_no_biomech

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sata
cd "$HOME/workspace/SATA/legged_gym/legged_gym"

LOGDIR="$HOME/workspace/bio-inspired-adaptive-locomotion/results/phase5-trainability/logs"
mkdir -p "$LOGDIR"

echo "[$(date)] Phase 5 start — task=$TASK gpu=$GPU seeds=[$SEEDS] iters=$ITERS envs=$ENVS"
for s in $SEEDS; do
  echo "[$(date)] === seed $s START ==="
  CUDA_VISIBLE_DEVICES="$GPU" WANDB_MODE=disabled python scripts/train.py \
    --task="$TASK" --headless --num_envs="$ENVS" --max_iterations="$ITERS" \
    --seed="$s" --run_name="no_biomech_s${s}" \
    > "$LOGDIR/no_biomech_s${s}.log" 2>&1
  rc=$?
  echo "[$(date)] === seed $s DONE (exit $rc) ==="
done
echo "[$(date)] Phase 5 ALL DONE"
