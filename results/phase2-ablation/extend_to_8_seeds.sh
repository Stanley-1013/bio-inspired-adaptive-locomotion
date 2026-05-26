#!/usr/bin/env bash
# Extend Phase 2 to 8 seeds per condition (reference + 5 ablations).
# Adds seeds 4-8 on top of the existing 3-seed runs (which stay as-is).
# 6 conditions × 5 new seeds = 30 runs in 10 batches of 3 on GPUs 0/2/3.
# Per-run safety timeout 120 min (vs 90 in the original launcher) to absorb
# environmental jitter from shared NFS / co-tenant CPU + GPU.
#
# Launch (detached):
#   nohup setsid bash extend_to_8_seeds.sh </dev/null \
#     > ~/workspace/SATA/legged_gym/logs/phase2_extend.log 2>&1 &
#   disown
#
# Monitor: bash check_status.sh (already updated to read seeds 1-8 + reference)

set -uo pipefail

cd ~/workspace/SATA/legged_gym/legged_gym
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sata

LOG_DIR=~/workspace/SATA/legged_gym/logs/phase2_launches
mkdir -p "$LOG_DIR"

GPUS=(0 2 3)
RUN_TIMEOUT=7200          # 120 min (vs the original 90 that clipped no_fatigue_s1)
NEW_SEEDS=(4 5 6 7 8)

# Each entry: <short-name>:<task-name-in-registry>:<run-name-prefix>
# (run_name = prefix_s<seed>)
CONDITIONS_A=(
  "reference:go2_torque:ref"
  "no_fatigue:go2_torque_no_fatigue:no_fatigue"
  "no_hill:go2_torque_no_hill:no_hill"
)
CONDITIONS_B=(
  "no_activation:go2_torque_no_activation:no_activation"
  "no_growth:go2_torque_no_growth:no_growth"
  "hard_terrain:go2_torque_hard_terrain:hard_terrain"
)

run_one() {
  local gpu="$1" task="$2" seed="$3" run="$4"
  local logfile="$LOG_DIR/${run}.log"
  (
    CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled \
      timeout $RUN_TIMEOUT python scripts/train.py \
        --task=$task --headless \
        --num_envs=4096 --max_iterations=3000 \
        --seed=$seed --run_name=$run \
        > "$logfile" 2>&1
    echo "EXIT_CODE=$?" >> "$logfile"
  ) &
}

run_batch() {
  local entries=("$@") seed="$1"; shift   # first arg is seed, rest are conditions
  local pids=() names=()
  echo ""
  echo "--- batch start $(date)  seed=$seed  conditions=$@ ---"
  local i=0
  for cond in "$@"; do
    local short="${cond%%:*}"
    local rest="${cond#*:}"
    local task="${rest%%:*}"
    local prefix="${rest##*:}"
    local run="${prefix}_s${seed}"
    local gpu="${GPUS[$i]}"
    echo "  launching $run on cuda:$gpu  ->  $LOG_DIR/${run}.log"
    run_one "$gpu" "$task" "$seed" "$run"
    pids+=($!)
    names+=("$run")
    i=$((i + 1))
  done
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}"
    local rc=$?
    echo "  [$(date '+%H:%M:%S')] ${names[$k]} (pid ${pids[$k]}) exited $rc"
  done
  echo "--- batch end $(date) ---"
}

echo "=========================================================="
echo "Phase 2 extension to 8 seeds — launcher start $(date)"
echo "Pod: $(hostname)   GPUs: ${GPUS[*]}"
echo "Adding seeds: ${NEW_SEEDS[*]}  to  6 conditions  =  ${#NEW_SEEDS[@]} × 6 = 30 runs"
echo "Expected wall time: ~13 hours (10 batches × ~75 min)"
echo "=========================================================="

for seed in "${NEW_SEEDS[@]}"; do
  echo ""
  echo "### Round seed=$seed"
  run_batch "$seed" "${CONDITIONS_A[@]}"
  run_batch "$seed" "${CONDITIONS_B[@]}"
done

echo ""
echo "=========================================================="
echo "Phase 2 extension DONE at $(date)"
echo "Run check_status.sh for a 6-condition × 8-seed table."
echo "=========================================================="
