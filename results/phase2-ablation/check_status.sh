#!/usr/bin/env bash
# Print a one-screen summary of Phase 2 launcher progress.
# Safe to run any time (while running, after done, before launch — handles all).

set -u

LOG_DIR=~/workspace/SATA/legged_gym/logs/phase2_launches
LAUNCHER_LOG=~/workspace/SATA/legged_gym/logs/phase2_launcher.log

ABLATIONS=(no_fatigue no_hill no_activation no_growth hard_terrain)
SEEDS=(1 2 3)

echo "=== Phase 2 status @ $(date) ==="
echo ""

# Launcher process state
if pgrep -af "launch_all.sh" >/dev/null 2>&1; then
  echo "Launcher: RUNNING"
  pgrep -af "launch_all.sh" | sed 's/^/  /'
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
printf "%-20s %-5s %-15s %10s %10s\n" "ablation" "seed" "status" "final_rew" "wall_min"
printf "%-20s %-5s %-15s %10s %10s\n" "--------" "----" "------" "---------" "--------"
for ab in "${ABLATIONS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run="${ab}_s${seed}"
    log="$LOG_DIR/${run}.log"
    if [ ! -f "$log" ]; then
      printf "%-20s %-5s %-15s %10s %10s\n" "$ab" "$seed" "NOT_STARTED" "-" "-"
      continue
    fi
    # Find status
    if grep -qE "Traceback|RuntimeError|CUDA out of memory|out of memory|Killed" "$log"; then
      status="FAILED"
    elif grep -q "Learning iteration 2999/3000" "$log"; then
      status="DONE"
    elif grep -q "Learning iteration" "$log"; then
      # mid-run; show last iter
      cur=$(grep "Learning iteration" "$log" | tail -1 | grep -oE "[0-9]+/3000" | head -1)
      status="RUNNING_${cur}"
    else
      status="STARTING"
    fi
    # Final reward (last printed "Mean reward")
    rew=$(grep "Mean reward:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}')
    [ -z "$rew" ] && rew="-"
    # Wall time (Total time)
    wt_s=$(grep "Total time:" "$log" 2>/dev/null | tail -1 | awk '{print $NF}' | tr -d 's')
    if [ -n "$wt_s" ]; then
      wt_min=$(printf "%.1f" "$(echo "$wt_s / 60" | bc -l 2>/dev/null)")
    else
      wt_min="-"
    fi
    printf "%-20s %-5s %-15s %10s %10s\n" "$ab" "$seed" "$status" "$rew" "$wt_min"
  done
done

# Tail of launcher log for context
echo ""
echo "Last 8 lines of launcher log ($LAUNCHER_LOG):"
if [ -f "$LAUNCHER_LOG" ]; then
  tail -n 8 "$LAUNCHER_LOG" | sed 's/^/  /'
else
  echo "  (launcher log not found yet)"
fi
