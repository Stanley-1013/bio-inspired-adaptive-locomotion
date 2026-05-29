#!/usr/bin/env bash
# Print a one-screen summary of Phase 2 launcher progress.
# Safe to run any time (while running, after done, before launch — handles all).

set -u

LOG_DIR=~/workspace/SATA/legged_gym/logs/phase2_launches
LAUNCHER_LOG=~/workspace/SATA/legged_gym/logs/phase2_launcher.log
EXTEND_LOG=~/workspace/SATA/legged_gym/logs/phase2_extend.log

# (label, run_name_prefix) for the table. "reference" is the unablated baseline
# (task = go2_torque). Seeds 4-8 logs land in $LOG_DIR; seeds 1-3 of "reference"
# came from Phase 1 (no per-run log file) so are hard-coded below.
CONDITIONS=(
  "reference:ref"
  "no_fatigue:no_fatigue"
  "no_hill:no_hill"
  "no_activation:no_activation"
  "no_growth:no_growth"
  "hard_terrain:hard_terrain"
)
SEEDS=(1 2 3 4 5 6 7 8)

# Known-good Phase 1 reference rewards (the 3 original seeds; logs not file-saved)
declare -A PHASE1_REF=( [1]=122 [2]=108 [3]=112 )
declare -A PHASE1_REF_WT=( [1]=63.6 [2]=64.7 [3]=64.0 )

echo "=== Phase 2 status @ $(date) ==="
echo ""

# Launcher process state — handle both the original and the extension launchers
running_launchers=$(pgrep -af "launch_all.sh|extend_to_8_seeds.sh" 2>/dev/null)
if [ -n "$running_launchers" ]; then
  echo "Launcher: RUNNING"
  echo "$running_launchers" | sed 's/^/  /'
else
  echo "Launcher: not running"
fi

# Active python train.py processes
echo ""
echo "Active training processes:"
ps -eo pid,etime,cmd | awk '/python.*train\.py.*go2_torque/ && !/awk/' | sed 's/^/  /' || echo "  (none)"

# GPU snapshot
echo ""
echo "GPU snapshot:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,temperature.gpu \
           --format=csv,noheader 2>/dev/null | sed 's/^/  /'

# Per-run status
echo ""
printf "%-15s %-5s %-15s %10s %10s\n" "condition" "seed" "status" "final_rew" "wall_min"
printf "%-15s %-5s %-15s %10s %10s\n" "---------" "----" "------" "---------" "--------"
for cond in "${CONDITIONS[@]}"; do
  label="${cond%%:*}"
  prefix="${cond##*:}"
  for seed in "${SEEDS[@]}"; do
    # Special case: reference seeds 1-3 came from Phase 1 (no log file)
    if [ "$label" = "reference" ] && [ -n "${PHASE1_REF[$seed]:-}" ]; then
      printf "%-15s %-5s %-15s %10s %10s\n" "$label" "$seed" "DONE_phase1" "${PHASE1_REF[$seed]}" "${PHASE1_REF_WT[$seed]}"
      continue
    fi
    run="${prefix}_s${seed}"
    log="$LOG_DIR/${run}.log"
    if [ ! -f "$log" ]; then
      printf "%-15s %-5s %-15s %10s %10s\n" "$label" "$seed" "NOT_STARTED" "-" "-"
      continue
    fi
    # Find status
    if grep -qE "Traceback|RuntimeError|CUDA out of memory|out of memory|Killed" "$log"; then
      status="FAILED"
    elif grep -q "Learning iteration 2999/3000" "$log"; then
      status="DONE"
    elif grep -q "Learning iteration" "$log"; then
      cur=$(grep "Learning iteration" "$log" | tail -1 | grep -oE "[0-9]+/3000" | head -1)
      status="RUNNING_${cur}"
    else
      status="STARTING"
    fi
    rew=$(grep "Mean reward:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
    [ -z "$rew" ] && rew="-"
    wt_s=$(grep "Total time:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' | tr -d 's')
    if [ -n "$wt_s" ]; then
      wt_min=$(printf "%.1f" "$(echo "$wt_s / 60" | bc -l 2>/dev/null)")
    else
      wt_min="-"
    fi
    printf "%-15s %-5s %-15s %10s %10s\n" "$label" "$seed" "$status" "$rew" "$wt_min"
  done
done

# Tail of launcher log for context (whichever launcher is most recently active)
echo ""
for lg in "$EXTEND_LOG" "$LAUNCHER_LOG"; do
  if [ -f "$lg" ]; then
    echo "Last 8 lines of $(basename "$lg"):"
    tail -n 8 "$lg" | sed 's/^/  /'
    echo ""
  fi
done
