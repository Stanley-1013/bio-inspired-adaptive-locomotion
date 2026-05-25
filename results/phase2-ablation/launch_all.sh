#!/usr/bin/env bash
# Phase 2 ablation launcher with adaptive scheduling.
#
# Round A (diagnostic): every ablation × seed=1, 2 batches of (3 + 2) on GPUs 0/2/3.
# Inter-round triage:
#   - For each ablation, classify its seed-1 run as OK / DID_NOT_FINISH / FAILED.
#   - "OK" means: Python exited 0 AND the log shows "Learning iteration 2999".
#   - "FAILED" means: log contains Traceback / RuntimeError / CUDA OOM.
# Round B (confirmation): only ablations that came back OK get seed=2 and seed=3.
# Failed / not-finished ablations are skipped (record kept) so we don't burn 4 h
# of GPU repeating a broken config.
#
# Launch (detached from this shell / Claude session):
#   nohup setsid bash launch_all.sh </dev/null \
#     > ~/workspace/SATA/legged_gym/logs/phase2_launcher.log 2>&1 &
#   disown
#
# Monitor: bash check_status.sh
# Per-run training output: ~/workspace/SATA/legged_gym/logs/phase2_launches/<run>.log
# Per-run checkpoints + tensorboard: ~/workspace/SATA/legged_gym/logs/SATA/<timestamp>_<run>/

set -uo pipefail   # NOT -e; we want failing runs to be recorded, not abort the rest

cd ~/workspace/SATA/legged_gym/legged_gym

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate sata

LOG_DIR=~/workspace/SATA/legged_gym/logs/phase2_launches
mkdir -p "$LOG_DIR"

GPUS=(0 2 3)                  # GPU 1 reserved for the other tenant
RUN_TIMEOUT=5400              # 90-min safety net per run (reference is ~65 min)

ALL_ABLATIONS=(no_fatigue no_hill no_activation no_growth hard_terrain)

# -----------------------------------------------------------------------------
# Helper: run a batch of up to 3 (ablation, seed) entries in parallel,
# pinning each to one of $GPUS in order. Blocks until all in the batch exit.
# Args: list of "ablation:seed" tokens.
# -----------------------------------------------------------------------------
run_batch() {
  local entries=("$@")
  local pids=() names=()
  echo ""
  echo "--- batch start $(date) : ${entries[*]} ---"
  for j in "${!entries[@]}"; do
    local entry="${entries[$j]}"
    local gpu="${GPUS[$j]}"
    local ab="${entry%:*}"
    local seed="${entry#*:}"
    local task="go2_torque_${ab}"
    local run="${ab}_s${seed}"
    local logfile="$LOG_DIR/${run}.log"
    echo "  launching $run on cuda:$gpu  ->  $logfile"
    (
      CUDA_VISIBLE_DEVICES=$gpu WANDB_MODE=disabled \
        timeout $RUN_TIMEOUT python scripts/train.py \
          --task=$task --headless \
          --num_envs=4096 --max_iterations=3000 \
          --seed=$seed --run_name=$run \
          > "$logfile" 2>&1
      echo "EXIT_CODE=$?" >> "$logfile"
    ) &
    pids+=($!)
    names+=("$run")
  done
  for k in "${!pids[@]}"; do
    wait "${pids[$k]}"
    local rc=$?
    echo "  [$(date '+%H:%M:%S')] ${names[$k]} (pid ${pids[$k]}) exited $rc"
  done
  echo "--- batch end $(date) ---"
}

# -----------------------------------------------------------------------------
# Helper: classify one (ablation, seed)'s log into OK / FAILED / DID_NOT_FINISH.
# Echoes the label on stdout.
# -----------------------------------------------------------------------------
classify_run() {
  local ab="$1" seed="$2"
  local logfile="$LOG_DIR/${ab}_s${seed}.log"
  if [ ! -f "$logfile" ]; then echo "MISSING"; return; fi
  if grep -qE "Traceback|RuntimeError|CUDA out of memory|out of memory|Killed" "$logfile"; then
    echo "FAILED"; return
  fi
  if grep -q "Learning iteration 2999/3000" "$logfile"; then
    echo "OK"; return
  fi
  echo "DID_NOT_FINISH"
}

echo "=========================================================="
echo "Phase 2 launcher starting at $(date)"
echo "Pod: $(hostname),  GPUs: ${GPUS[*]}"
echo "Ablations: ${ALL_ABLATIONS[*]}"
echo "Plan: Round A (5× seed=1), triage, Round B (surviving × seeds 2,3)"
echo "=========================================================="

# -----------------------------------------------------------------------------
# Round A — diagnostic: every ablation, seed=1.
# 5 ablations / 3 GPUs → 2 batches.  Batch A2 has only 2 runs (GPU 3 idles briefly).
# -----------------------------------------------------------------------------
echo ""
echo "### Round A (diagnostic) at $(date)"

run_batch "no_fatigue:1" "no_hill:1"     "no_activation:1"
run_batch "no_growth:1"  "hard_terrain:1"

# -----------------------------------------------------------------------------
# Triage: pick which ablations get seeds 2 and 3.
# -----------------------------------------------------------------------------
echo ""
echo "### Triage after Round A at $(date)"
SURVIVED=()
for ab in "${ALL_ABLATIONS[@]}"; do
  status=$(classify_run "$ab" 1)
  case "$status" in
    OK)
      reward=$(grep "Mean reward:" "$LOG_DIR/${ab}_s1.log" 2>/dev/null | tail -1 | awk '{print $NF}')
      echo "  ${ab}_s1: OK  (final reward $reward) -> will run seeds 2,3"
      SURVIVED+=("$ab")
      ;;
    *)
      echo "  ${ab}_s1: $status  -> skipping seeds 2,3 (see $LOG_DIR/${ab}_s1.log)"
      ;;
  esac
done

if [ ${#SURVIVED[@]} -eq 0 ]; then
  echo ""
  echo "!!! All Round-A ablations failed/did-not-finish.  Round B aborted."
  echo "=========================================================="
  echo "Launcher exiting early at $(date)"
  echo "=========================================================="
  exit 0
fi

# -----------------------------------------------------------------------------
# Round B — confirmation: surviving ablations × seeds 2 and 3.
# Group into batches of up to 3.
# -----------------------------------------------------------------------------
echo ""
echo "### Round B (confirmation) at $(date)"
echo "Survivors: ${SURVIVED[*]}"

# Build the full Round B queue.
QUEUE=()
for ab in "${SURVIVED[@]}"; do
  for seed in 2 3; do
    QUEUE+=("${ab}:${seed}")
  done
done

# Slice queue into batches of 3.
total=${#QUEUE[@]}
batch_idx=0
i=0
while [ $i -lt $total ]; do
  batch_idx=$((batch_idx + 1))
  end=$((i + 3))
  [ $end -gt $total ] && end=$total
  echo ""
  echo "--- Round B batch $batch_idx ---"
  run_batch "${QUEUE[@]:$i:$((end - i))}"
  i=$end
done

echo ""
echo "=========================================================="
echo "Phase 2 launcher DONE at $(date)"
echo "Survivors: ${SURVIVED[*]}"
echo "Total batches: 2 (Round A) + $batch_idx (Round B)"
echo "Run check_status.sh for a summary."
echo "=========================================================="
